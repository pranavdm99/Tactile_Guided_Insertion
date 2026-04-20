"""
Tactile depth capture using Robosuite's native shared renderer context.
Bypasses high-level sim.render() to ensure MjvOption masking is strictly enforced.
Aligned for MuJoCo 3.x and Robosuite 1.5.x.
"""

from __future__ import annotations
import numpy as np
import mujoco

class TactileDepthCapture:
    """Uses the environment's native renderer context to capture tactile depth."""

    def __init__(self, sim, height: int, width: int):
        self._height = height
        self._width = width

    def close(self):
        # Nothing to close - native renderer is managed by Robosuite
        pass

    def render_depth_meters_batched(self, sim, camera_names: list, scene_option: mujoco.MjvOption = None) -> list:
        """
        Render depth from multiple cameras using direct low-level MuJoCo calls.
        This ensures our MjvOption (robot masking) is STRICTLY enforced.
        """
        if not hasattr(sim, "_render_context_offscreen"):
            raise RuntimeError("Simulation object does not have _render_context_offscreen side-car.")
            
        rs_ctx = sim._render_context_offscreen
        # Robosuite wraps the MjrContext in rs_ctx.con
        mj_ctx = rs_ctx.con if hasattr(rs_ctx, "con") else rs_ctx
        
        mj_model = sim.model._model if hasattr(sim.model, "_model") else sim.model
        mj_data = sim.data._data if hasattr(sim.data, "_data") else sim.data
        
        # 1. Backup current renderer state
        old_geomgroup = np.array(rs_ctx.vopt.geomgroup)
        old_sitegroup = np.array(rs_ctx.vopt.sitegroup)
        
        try:
            # 2. Apply Tactile Scene Options
            if scene_option is not None:
                for i in range(6):
                    rs_ctx.vopt.geomgroup[i] = scene_option.geomgroup[i]
                    rs_ctx.vopt.sitegroup[i] = scene_option.sitegroup[i]

            # 3. Setup Viewport, Camera and Scene
            viewport = mujoco.MjrRect(0, 0, self._width, self._height)
            mjv_cam = mujoco.MjvCamera()
            
            out = []
            for name in camera_names:
                cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                if cam_id == -1:
                    raise ValueError(f"Camera '{name}' not found in model.")
                
                # Configure the camera
                mjv_cam.fixedcamid = cam_id
                mjv_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                
                # Update the scene (MuJoCo 3.x signature: m, d, opt, pert, cam, catmask, scn)
                mujoco.mjv_updateScene(
                    mj_model, 
                    mj_data, 
                    rs_ctx.vopt, 
                    None,      # MjvPerturb
                    mjv_cam, 
                    mujoco.mjtCatBit.mjCAT_ALL, 
                    rs_ctx.scn
                )
                
                # Direct low-level render (Bypasses Robosuite wrappers entirely)
                # Pass the raw MjrContext (mj_ctx)
                mujoco.mjr_render(viewport, rs_ctx.scn, mj_ctx)
                
                # Read Depth Buffer directly [0, 1] range
                z_raw = np.empty((self._height, self._width), dtype=np.float32)
                mujoco.mjr_readPixels(None, z_raw, viewport, mj_ctx)
                
                # 4. Linearization: Convert OpenGL [0, 1] buffer to Meters
                extent = mj_model.stat.extent
                near = float(mj_model.vis.map.znear * extent)
                far = float(mj_model.vis.map.zfar * extent)
                
                z_meters = near / (1.0 - z_raw * (1.0 - near / far))
                
                # Raw MuJoCo depth is upside down (bottom-to-top)
                # We return it raw and let TactileObservationWrapper handle final orientation
                out.append(z_meters)
                
            return out
            
        finally:
            # 5. Restore original renderer state
            for i in range(6):
                rs_ctx.vopt.geomgroup[i] = old_geomgroup[i]
                rs_ctx.vopt.sitegroup[i] = old_sitegroup[i]

def meters_to_normalized_depth(sim, z_meters: np.ndarray) -> np.ndarray:
    # Near/Far extraction from model
    mj_model = sim.model._model if hasattr(sim.model, "_model") else sim.model
    extent = mj_model.stat.extent
    near = float(mj_model.vis.map.znear * extent)
    far = float(mj_model.vis.map.zfar * extent)
    
    z = np.clip(np.asarray(z_meters, dtype=np.float64), near * 1.0001, far * 0.9999)
    denom = 1.0 - near / far
    d = (1.0 - near / z) / denom
    return np.clip(d.astype(np.float32), 0.0, 1.0)

def bandpass_gel_depth(z_meters: np.ndarray, z_ref_m: float, far_cap_m: float = 0.012) -> np.ndarray:
    z = np.asarray(z_meters, dtype=np.float32)
    cap = z_ref_m + far_cap_m
    return np.where(z > cap, z_ref_m, z)
