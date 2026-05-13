import os
import h5py
import numpy as np
import cv2
import json
import time

class DataRecorder:
    """
    Robust HDF5 Data Recorder for Robosuite and FOTS tactile integration.
    Produces datasets compatible with Robomimic IL training.
    """
    # Robomimic robosuite environments use EnvType.ROBOSUITE_TYPE == 1 (see robomimic.envs.env_base).
    _ROBOMIMIC_ROBOSUITE_TYPE = 1

    def __init__(self, output_dir, downsample_size=(128, 96), filename=None):
        """
        Args:
            output_dir (str): Directory where HDF5 files will be saved.
            downsample_size (tuple): (Width, Height) for tactile images.
            filename (str): Optional custom filename. Defaults to demo_<timestamp>.hdf5.
        """
        self.output_dir = output_dir
        self.downsample_size = downsample_size
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"[INFO] Created dataset directory: {self.output_dir}")
            
        if filename is None:
            filename = f"demo_{int(time.time())}.hdf5"
        self.filepath = os.path.join(self.output_dir, filename)
        
        # Buffer for current episode
        self.current_episode = {
            "obs": {},
            "actions": [],
            "rewards": [],
            "dones": []
        }
        self.demo_count = 0
        
        # Metadata to identify which object keys to store dynamically
        self.object_keys = []
        # Per-episode metadata
        self._nut_type = None
        self._render_type = None
        # Optional: serialized Robomimic `env_meta` written to data.attrs["env_args"] on first save
        self._robomimic_env_context = None

    def set_robomimic_env_context(self, env_meta_dict):
        """
        If set, the first time an episode is saved, `data` group gets attrs["env_args"] = json.dumps(...)
        with keys env_name, type, env_version, env_kwargs as expected by robomimic.utils.env_utils.

        Args:
            env_meta_dict (dict): Must be JSON-serializable (no numpy arrays / callables).
        """
        self._robomimic_env_context = dict(env_meta_dict)

    def start_episode(self):
        """Reset buffers for a new trajectory."""
        # Re-discover nut observables each episode: active nut switches name
        # (SquareNut_* vs RoundNut_*) when nut_type is None and the env randomizes per reset.
        self.object_keys = []
        self._nut_type = None
        self._render_type = None
        self.current_episode = {
            "obs": {},
            "actions": [],
            "rewards": [],
            "dones": []
        }

    def record_step(self, obs, action, reward, done):
        """
        Buffers a single step of the interaction.
        Args:
            obs (dict): Environment observation dictionary.
            action (np.array): Action taken by the agent.
            reward (float): Reward received.
            done (bool): Termination flag.
        """
        # 1. Capture Episode Metadata (only once per episode)
        nut_id = obs.get("_nut_type", -1)
        if self._nut_type is None:
            self._nut_type = int(nut_id)
        if self._render_type is None:
            self._render_type = int(obs.get("_render_type", -1))

        # 2. Key Standardization Logic
        # We rename keys like RoundNut_pos to object_pos to ensure consistent datasets
        nut_prefix = "SquareNut" if nut_id == 0 else "RoundNut" if nut_id == 1 else None
        
        target_keys = [
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
            "tactile_left", "tactile_right", "agentview_image", "robot0_eye_in_hand_image"
        ]
        
        # Build processed observation for this step
        processed_obs = {}
        
        # Add standardized object keys
        if nut_prefix:
            for k in obs.keys():
                if k.startswith(nut_prefix):
                    standard_k = k.replace(nut_prefix, "object")
                    processed_obs[standard_k] = obs[k]
                    if standard_k not in target_keys:
                        target_keys.append(standard_k)

        # Add explicit robot and vision keys
        for k in target_keys:
            if k in obs:
                processed_obs[k] = obs[k]
            elif k not in processed_obs:
                continue
                
        # Add nut_type as an observation array (for robot training)
        processed_obs["nut_type"] = np.array([self._nut_type], dtype=np.int32)
        if "nut_type" not in target_keys:
            target_keys.append("nut_type")

        # 3. Process and Buffer
        for k in target_keys:
            if k not in processed_obs:
                continue
                
            val = processed_obs[k]
            
            # Application-specific processing
            if "tactile" in k:
                # Downsample tactile images efficiently for IL training
                val = cv2.resize(val, self.downsample_size, interpolation=cv2.INTER_AREA)
            
            if k not in self.current_episode["obs"]:
                self.current_episode["obs"][k] = []
            
            self.current_episode["obs"][k].append(val)
            
        # 4. Store actions and signals
        self.current_episode["actions"].append(action)
        self.current_episode["rewards"].append(reward)
        self.current_episode["dones"].append(done)

    def save_episode(self, discard=False):
        """
        Writes the buffered episode to the HDF5 file.
        Returns:
            bool: True if saved successfully.
        """
        if discard:
            print("[INFO] Episode discarded.")
            self.start_episode()
            return False

        if len(self.current_episode["actions"]) == 0:
            return False

        with h5py.File(self.filepath, "a") as f:
            # Ensure the 'data' group exists
            if "data" not in f:
                data_grp = f.create_group("data")
            else:
                data_grp = f["data"]

            # Robomimic reads env reconstruction metadata from data.attrs["env_args"] (JSON string).
            if self._robomimic_env_context is not None and "env_args" not in data_grp.attrs:
                env_args = {
                    "env_name": self._robomimic_env_context["env_name"],
                    "type": self._ROBOMIMIC_ROBOSUITE_TYPE,
                    "env_version": self._robomimic_env_context.get(
                        "env_version", self._robomimic_env_context.get("repository_version", "")
                    ),
                    "env_kwargs": self._robomimic_env_context.get("env_kwargs", {}),
                }
                data_grp.attrs["env_args"] = json.dumps(env_args, default=str)
            
            # Find next demo index
            demo_id = f"demo_{self.demo_count}"
            while demo_id in data_grp:
                self.demo_count += 1
                demo_id = f"demo_{self.demo_count}"
            
            ep_grp = data_grp.create_group(demo_id)
            
            # Store primary keys
            ep_grp.create_dataset("actions", data=np.array(self.current_episode["actions"]))
            ep_grp.create_dataset("rewards", data=np.array(self.current_episode["rewards"]))
            dones = np.array(self.current_episode["dones"], dtype=bool)
            # Robosuite's done is horizon-only; demos usually end early with all False. Mark the
            # final transition as terminal so BC / sequence tooling that expects a boundary flag works.
            if dones.size > 0 and not np.any(dones):
                dones = dones.copy()
                dones[-1] = True
            ep_grp.create_dataset("dones", data=dones)
            
            # Store observations
            obs_grp = ep_grp.create_group("obs")
            for k, vals in self.current_episode["obs"].items():
                obs_grp.create_dataset(k, data=np.array(vals))
            
            # Add some metadata to the demo
            ep_grp.attrs["num_samples"] = len(self.current_episode["actions"])
            ep_grp.attrs["nut_type"] = self._nut_type
            ep_grp.attrs["render_type"] = self._render_type
            
            # Update global metadata on first save
            if "total" not in f.attrs:
                f.attrs["total"] = 0
                f.attrs["env_name"] = (
                    self._robomimic_env_context.get("env_name", "NutAssemblySingle")
                    if self._robomimic_env_context
                    else "NutAssemblySingle"
                )
            f.attrs["total"] += 1

        print(f"[SUCCESS] Saved {demo_id} to {self.filepath} ({len(self.current_episode['actions'])} steps)")
        self.demo_count += 1
        self.start_episode()
        return True
