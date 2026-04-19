"""
Tactile depth capture using MuJoCo's native Renderer depth path.
Aligned for MuJoCo 3.x and Robosuite 1.5.x.
"""

from __future__ import annotations
import numpy as np
import mujoco

try:
    from robosuite.utils.binding_utils import _MjSim_render_lock
except ImportError:
    _MjSim_render_lock = None


def _get_raw_mj_model(sim):
    """Extract raw MjModel struct from Robosuite or native objects."""
    # Priority A: Robosuite 1.5.x wrapper
    if hasattr(sim, "model") and hasattr(sim.model, "_model"):
        return sim.model._model
    # Priority B: Already a native struct
    if isinstance(sim, mujoco._structs.MjModel):
        return sim
    # Priority C: Fallback to direct attribute
    return getattr(sim, "model", sim)

def _get_raw_mj_data(sim):
    """Extract raw MjData struct from Robosuite or native objects."""
    # Priority A: Robosuite 1.5.x wrapper
    if hasattr(sim, "data") and hasattr(sim.data, "_data"):
        return sim.data._data
    # Priority B: Already a native struct
    if isinstance(sim, mujoco._structs.MjData):
        return sim
    # Priority C: Fallback to direct attribute
    return getattr(sim, "data", sim)

def _near_far_meters(sim):
    mj_model = _get_raw_mj_model(sim)
    extent = mj_model.stat.extent
    near = float(mj_model.vis.map.znear * extent)
    far = float(mj_model.vis.map.zfar * extent)
    return near, far


def meters_to_normalized_depth(sim, z_meters: np.ndarray) -> np.ndarray:
    near, far = _near_far_meters(sim)
    z = np.clip(np.asarray(z_meters, dtype=np.float64), near * 1.0001, far * 0.9999)
    denom = 1.0 - near / far
    d = (1.0 - near / z) / denom
    return np.clip(d.astype(np.float32), 0.0, 1.0)


def bandpass_gel_depth(z_meters: np.ndarray, z_ref_m: float, far_cap_m: float = 0.012) -> np.ndarray:
    z = np.asarray(z_meters, dtype=np.float32)
    cap = z_ref_m + far_cap_m
    return np.where(z > cap, z_ref_m, z)


class TactileDepthCapture:
    """Holds a mujoco.Renderer sized for tactile cameras."""

    def __init__(self, sim, height: int, width: int, max_geom: int = 10000):
        # Don't store sim reference - it might become stale
        # Store renderer and let caller pass fresh sim data each time
        self._height = height
        self._width = width
        mj_model = _get_raw_mj_model(sim)
        self._renderer = mujoco.Renderer(mj_model, height=height, width=width, max_geom=max_geom)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __del__(self):
        try: self.close()
        except: pass

    def render_depth_meters_batched(self, sim, camera_names: list, scene_option: mujoco.MjvOption = None) -> list:
        """
        Render depth from multiple cameras.
        
        Args:
            sim: Fresh simulation object (passed each call to avoid stale references)
            camera_names: List of camera names to render
            scene_option: Optional MjvOption for controlling visibility
        """
        mj_data = _get_raw_mj_data(sim)
        r = self._renderer

        def _do():
            r.enable_depth_rendering()
            try:
                out = []
                for name in camera_names:
                    r.update_scene(mj_data, camera=name, scene_option=scene_option)
                    out.append(r.render())
                return out
            finally:
                r.disable_depth_rendering()

        if _MjSim_render_lock is not None:
            with _MjSim_render_lock:
                return _do()
        return _do()
