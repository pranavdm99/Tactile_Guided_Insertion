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
    def __init__(self, output_dir, downsample_size=(64, 48), filename=None):
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

    def start_episode(self):
        """Reset buffers for a new trajectory."""
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
        # 1. Process and extract observations
        target_keys = [
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
            "tactile_left", "tactile_right", "agentview_image"
        ]
        
        # Dynamically discover object keys (RoundNut or SquareNut)
        if not self.object_keys:
            self.object_keys = [k for k in obs.keys() if ("Nut" in k and "image" not in k)]
        
        target_keys.extend(self.object_keys)
        
        for k in target_keys:
            if k not in obs:
                continue
                
            val = obs[k]
            
            # Application-specific processing
            if "tactile" in k:
                # Downsample tactile images efficiently for IL training
                val = cv2.resize(val, self.downsample_size, interpolation=cv2.INTER_AREA)
            
            if k not in self.current_episode["obs"]:
                self.current_episode["obs"][k] = []
            
            self.current_episode["obs"][k].append(val)
            
        # 2. Store actions and signals
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
            
            # Find next demo index
            demo_id = f"demo_{self.demo_count}"
            while demo_id in data_grp:
                self.demo_count += 1
                demo_id = f"demo_{self.demo_count}"
            
            ep_grp = data_grp.create_group(demo_id)
            
            # Store primary keys
            ep_grp.create_dataset("actions", data=np.array(self.current_episode["actions"]))
            ep_grp.create_dataset("rewards", data=np.array(self.current_episode["rewards"]))
            ep_grp.create_dataset("dones", data=np.array(self.current_episode["dones"]))
            
            # Store observations
            obs_grp = ep_grp.create_group("obs")
            for k, vals in self.current_episode["obs"].items():
                obs_grp.create_dataset(k, data=np.array(vals))
            
            # Add some metadata to the demo
            ep_grp.attrs["num_samples"] = len(self.current_episode["actions"])
            
            # Update global metadata on first save
            if "total" not in f.attrs:
                f.attrs["total"] = 0
                f.attrs["env_name"] = "NutAssemblySingle" # Hardcoded for now but can be dynamic
            f.attrs["total"] += 1

        print(f"[SUCCESS] Saved {demo_id} to {self.filepath} ({len(self.current_episode['actions'])} steps)")
        self.demo_count += 1
        self.start_episode()
        return True
