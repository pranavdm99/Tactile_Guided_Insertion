"""
Tactile depth capture using MuJoCo's native Renderer depth path.

Robosuite's sim.render(depth=True) uses mjr_readPixels(rgb, depth) after a
standard RGB render. For tactile sensing we need the same depth pipeline as
mujoco.Renderer: enable_depth_rendering() before update_scene/render so MuJoCo
uses segmented depth (mjRND_SEGMENT / mjRND_IDCOLOR) and reads metric depth.

See: mujoco.Renderer.enable_depth_rendering() and render() implementation.
"""

from __future__ import annotations

import numpy as np
import mujoco

try:
    from robosuite.utils.binding_utils import _MjSim_render_lock
except ImportError:
    _MjSim_render_lock = None


def _near_far_meters(sim):
    extent = sim.model.stat.extent
    near = float(sim.model.vis.map.znear * extent)
    far = float(sim.model.vis.map.zfar * extent)
    return near, far


def meters_to_normalized_depth(sim, z_meters: np.ndarray) -> np.ndarray:
    """Invert robosuite get_real_depth_map: z = near/(1 - d*(1-near/far))."""
    near, far = _near_far_meters(sim)
    z = np.clip(np.asarray(z_meters, dtype=np.float64), near * 1.0001, far * 0.9999)
    denom = 1.0 - near / far
    d = (1.0 - near / z) / denom
    return np.clip(d.astype(np.float32), 0.0, 1.0)


def bandpass_gel_depth(z_meters: np.ndarray, z_ref_m: float, far_cap_m: float = 0.012) -> np.ndarray:
    """
    Clamp geometry farther than z_ref + far_cap to z_ref (flat gel).
    Real DIGIT sees only the inner gel plane plus sub-mm indentations; rays that
    hit the table/nut meters away are treated as no added structure.
    """
    z = np.asarray(z_meters, dtype=np.float32)
    cap = z_ref_m + far_cap_m
    return np.where(z > cap, z_ref_m, z)


class TactileDepthCapture:
    """Holds a mujoco.Renderer sized for tactile cameras (one shared context)."""

    def __init__(self, sim, height: int, width: int, max_geom: int = 10000):
        self._sim = sim
        self._height = height
        self._width = width
        self._renderer = mujoco.Renderer(
            sim.model._model, height=height, width=width, max_geom=max_geom
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def render_depth_meters(self, camera_name: str) -> np.ndarray:
        """
        Metric depth (meters), shape (height, width). Order:
        enable_depth_rendering -> update_scene -> render -> disable_depth_rendering.
        """
        return self.render_depth_meters_batched([camera_name])[0]

    def render_depth_meters_batched(self, camera_names: list) -> list:
        """
        Same as repeated render_depth_meters(), but one enable/disable cycle and
        one lock acquisition for all cameras — noticeably cheaper per step than
        two isolated calls (segmented depth mode toggled once, less EGL churn).
        """
        data = self._sim.data._data
        r = self._renderer

        def _do():
            r.enable_depth_rendering()
            try:
                out = []
                for name in camera_names:
                    r.update_scene(data, camera=name)
                    out.append(r.render())
                return out
            finally:
                r.disable_depth_rendering()

        if _MjSim_render_lock is not None:
            with _MjSim_render_lock:
                return _do()
        return _do()
