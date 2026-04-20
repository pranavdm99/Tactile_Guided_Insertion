import os
import time
import torch
import numpy as np
import cv2
import mujoco
from robosuite.wrappers import Wrapper
from fots_sim.utils.mlp_render import MLPRender
from fots_sim.mlp_model import MLP
from env_setup.tactile_depth_capture import (
    TactileDepthCapture, 
    meters_to_normalized_depth, 
    bandpass_gel_depth
)

class TactileObservationWrapper(Wrapper):
    """
    Robosuite Observation Wrapper that injects FOTS tactile imprints.
    Aligned for Robosuite 1.5.x and MuJoCo 3.x.
    
    NOTE: FOTS assets are (320, 240). MuJoCo renders as (H, W).
    To match assets, we use height=320, width=240.
    """
    def __init__(self, env, fidelity_mode=True, height=320, width=240, device=None):
        super().__init__(env)
        self.fidelity_mode = fidelity_mode
        self.height = height
        self.width = width
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize Depth Capture
        self.depth_capture = TactileDepthCapture(self.env.sim, height=height, width=width)
        self._model_id = id(self.env.sim.model)  # Track for stale renderer detection
        
        # 3. Discover Camera Names (handles robot0_gripper0_ prefixes in Robosuite 1.5.x)
        self.camera_names = self._discover_tactile_cameras()
        
        # 4. Initialize Masking: Move robot into a hidden group
        self._mask_robot_geoms()
        
        # 5. Setup Scene Options for Tactile Rendering
        # Cameras should ONLY see objects (Groups 0/1), not the gripper (Group 2)
        self.tactile_scene_option = mujoco.MjvOption()
        for i in range(6):
            # Show objects (0=collision, 1=visual)
            self.tactile_scene_option.geomgroup[i] = 1 if i in [0, 1] else 0
            self.tactile_scene_option.sitegroup[i] = 0
        
        # 4. Initialize FOTS Rendering Engine (if needed)
        self.fots_render = None
        if self.fidelity_mode:
            self._init_fots_engine()
            
        # 5. Dynamic Baselines (updated on reset)
        self.baseline_l = None
        self.baseline_r = None

    def _discover_tactile_cameras(self):
        """Find the mangled tactile camera names in the simulation model."""
        sim_model = self.env.sim.model
        if hasattr(sim_model, "model"): mj_model = sim_model.model
        elif hasattr(sim_model, "_model"): mj_model = sim_model._model
        else: mj_model = sim_model
        
        found = {"left": None, "right": None}
        for i in range(mj_model.ncam):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if "tactile_cam_left" in name: found["left"] = name
            if "tactile_cam_right" in name: found["right"] = name
        
        if found["left"] and found["right"]:
            print(f"[SUCCESS] FOTS Discovery: Linked to {found['left']} and {found['right']}")
        else:
            print(f"[WARNING] FOTS Discovery Failed! Found: {found}")
            if not found["left"]: found["left"] = "robot0_tactile_cam_left"
            if not found["right"]: found["right"] = "robot0_tactile_cam_right"
            
        return found

    def _mask_robot_geoms(self):
        """
        Migrates robot geoms to Group 2 so they can be hidden from tactile renderer.
        This must be called on every reset as Robosuite replaces the mjModel.
        """
        sim_model = self.env.sim.model
        mj_model = sim_model._model if hasattr(sim_model, "_model") else sim_model
        
        mask_keywords = ["robot", "finger", "hand", "gripper", "panda", "link", "franka"]
        for i in range(mj_model.ngeom):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and any(x in name.lower() for x in mask_keywords):
                # Move to group 2 (Hidden from tactile renderer)
                mj_model.geom_group[i] = 2

    def _init_fots_engine(self):
        base_dir = "/app/fots_sim"
        if not os.path.exists(base_dir): 
            base_dir = os.path.join(os.getcwd(), "fots_sim")
            
        bg_img = np.load(os.path.join(base_dir, "assets/digit_bg.npy"))
        bg_depth = np.load(os.path.join(base_dir, "utils/ini_depth_extent.npy"))
        bg_mlp = np.load(os.path.join(base_dir, "utils/ini_bg_mlp.npy"))
        
        model = MLP().to(self.device)
        model_path = os.path.join(base_dir, "models/mlp_n2c_r.pth")
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.eval()
        
        self.fots_render = MLPRender(background_img=bg_img, bg_depth=bg_depth, bg_render=bg_mlp, model=model)

    def _process_depth(self, z, side="left"):
        """
        Converts raw depth map into a normalized tactile deformation map [0, 1].
        """
        # 1. Orientation Sync (to match FOTS assets)
        # MuJoCo raw depth is upside down. 
        # Left finger needs 180-degree correction (fliplr + flipud)
        # Right finger is already 180-degrees rotated in XML, so it only needs horizontal sync
        if side == "left":
            z = np.fliplr(np.flipud(z))
        else:
            z = z # No flip needed for Right if it started 180-degrees from Left

        # DEBUG: Range Check (Should be ~0.0225m for gel surface)
        # print(f"DEBUG: {side} Raw depth range: {z.min():.4f} - {z.max():.4f}")

        # 2. Bandpass Filter: Clamp far objects to gel surface depth
        z_clamped = bandpass_gel_depth(z, z_ref_m=0.0225, far_cap_m=0.010)
        
        # 3. Global scene normalization (matches FOTS training distribution)
        return meters_to_normalized_depth(self.env.sim, z_clamped)

    def _get_tactile_obs(self):
        """Synthesize left and right tactile observations."""
        cams = [self.camera_names["left"], self.camera_names["right"]]
        # Render depth from tactile cameras (pass fresh sim reference)
        z_meters_list = self.depth_capture.render_depth_meters_batched(
            self.env.sim, cams, scene_option=self.tactile_scene_option
        )
        
        # 0. Denoising: Remove high-frequency "spikes" and jitter
        # Convert to numpy for filtering if not already
        z_meters_list = [np.nan_to_num(z, nan=10.0, posinf=10.0, neginf=10.0) for z in z_meters_list]
        z_meters_list = [cv2.medianBlur(z.astype(np.float32), 3) for z in z_meters_list]
        z_meters_list = [cv2.GaussianBlur(z, (5, 5), 0) for z in z_meters_list]
        
        
        obs = {}
        for i, side in enumerate(["left", "right"]):
            z_norm = self._process_depth(z_meters_list[i], side=side)
            
            # Debug: Check if depth is changing (remove after debugging)
            # if hasattr(self, '_debug_counter'):
            #     self._debug_counter += 1
            #     if self._debug_counter % 100 == 0:
            #         print(f'[DEBUG] {side} depth: min={z_meters_list[i].min():.4f}, max={z_meters_list[i].max():.4f}, mean={z_meters_list[i].mean():.4f}')
            
            if self.fidelity_mode and self.fots_render:
                # Fidelity Mode: Full MLP Synthesis (expects 0-1, returns RGB)
                baseline = self.baseline_l if side == "left" else self.baseline_r
                self.fots_render.bg_depth = baseline
                self.fots_render._pre_scaled_bg = self.fots_render.bg_depth * self.fots_render._scale
                rgb = self.fots_render.generate(z_norm)
                obs[f"tactile_{side}"] = rgb  # Keep as RGB
            else:
                # Fast Mode: High-Contrast Depth Visualization
                # Compute depth difference from baseline for better contrast
                baseline = self.baseline_l if side == "left" else self.baseline_r
                depth_diff = baseline - z_norm  # Positive = object closer than baseline
                
                # Enhance contrast: focus on the deformation range
                # Map [0, 0.3] depth difference to full [0, 255] color range
                depth_diff_enhanced = np.clip(depth_diff / 0.3, 0.0, 1.0)
                depth_vis = (depth_diff_enhanced * 255).astype(np.uint8)
                
                # Use jet colormap: blue=no contact, red=maximum contact
                jet_bgr = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                # Convert to RGB to match fidelity mode format
                obs[f"tactile_{side}"] = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
                
        return obs

    def reset(self):
        # 1. Standard Robosuite Reset
        obs = super().reset()
        
        # 2. Dynamic Masking Refresh: Robosuite replaces the model on reset!
        self._mask_robot_geoms()
        
        # 3. Hardware Cool-down: Allows native renderer to settle
        time.sleep(0.2)
        self.env.sim.forward() 
        
        cams = [self.camera_names["left"], self.camera_names["right"]]

        # 3. Warm-up (Clears any residual rendering artifacts)
        for _ in range(2):
           _ = self.depth_capture.render_depth_meters_batched(
               self.env.sim, cams, scene_option=self.tactile_scene_option
           )

        # 4. Update Baselines (No Contact state)
        z_meters_list = self.depth_capture.render_depth_meters_batched(
            self.env.sim, cams, scene_option=self.tactile_scene_option
        )
        self.baseline_l = self._process_depth(z_meters_list[0], side="left")
        self.baseline_r = self._process_depth(z_meters_list[1], side="right")
        
        # Inject tactile imprints
        tactile_obs = self._get_tactile_obs()
        obs.update(tactile_obs)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        # Inject tactile imprints
        tactile_obs = self._get_tactile_obs()
        obs.update(tactile_obs)
        return obs, reward, done, info
