import os
import sys
import numpy as np
import mujoco
import cv2
import torch
from fots_sim.utils.mlp_render import MLPRender
from fots_sim.mlp_model import MLP
from env_setup.tactile_depth_capture import (
    TactileDepthCapture, 
    meters_to_normalized_depth, 
    bandpass_gel_depth
)

def get_xml(shape):
    """Stand-alone MJCF for tactile primitive validation."""
    geom_xml = ""
    if shape == "sphere":
        geom_xml = '<geom name="sphere" type="sphere" size="0.015" rgba="0.8 0.2 0.2 1.0" group="1" condim="4" friction="1 0.05 0.01"/>'
    elif shape == "box":
        geom_xml = '<geom name="box" type="box" size="0.007 0.012 0.015" rgba="0.8 0.2 0.2 1.0" group="1" condim="4" friction="1 0.05 0.01"/>'
    elif shape == "cylinder":
        geom_xml = '<geom name="cylinder" type="cylinder" size="0.007 0.015" rgba="0.8 0.2 0.2 1.0" group="1" condim="4" friction="1 0.05 0.01"/>'
    
    return f"""
<mujoco model="fots_primitive_sandbox">
    <option integrator="Euler" timestep="0.004" solver="Newton" cone="elliptic"/>
    <visual><map znear="0.001" zfar="10.0"/><global offwidth="2560" offheight="1440"/></visual>
    <worldbody>
        <light pos="0 0 2" dir="0 0 -1" diffuse="1.2 1.2 1.2" ambient="0.4 0.4 0.4"/>
        <body name="sensor" pos="0 0 0.05">
            <geom name="gel_pad" type="box" size="0.015 0.015 0.001" rgba="0.3 0.4 0.5 1.0" group="1" solref="0.02 1"/>
            <camera name="tactile_cam" pos="0 0 -0.0225" euler="180 0 0" fovy="60"/>
        </body>
        <body name="object" pos="0 0 0.075">
            <joint type="slide" axis="1 0 0" name="obj_x"/>
            <joint type="slide" axis="0 1 0" name="obj_y"/>
            <joint type="slide" axis="0 0 1" name="obj_z"/>
            {geom_xml}
        </body>
        <geom name="floor" type="plane" size="1 1 0.01" pos="0 0 0" rgba="0.5 0.5 0.5 1"/>
        <camera name="world_cam" pos="0.3 0.3 0.3" mode="targetbody" target="sensor" fovy="60"/>
    </worldbody>
</mujoco>
"""

class ModelWrapper:
    def __init__(self, m): self._model = m
    def __getattr__(self, name): return getattr(self._model, name)
class DataWrapper:
    def __init__(self, d): self._data = d
    def __getattr__(self, name): return getattr(self._data, name)
class MockSim:
    def __init__(self, model, data):
        self.model = ModelWrapper(model)
        self.data = DataWrapper(data)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="FOTS Primitive Sandbox")
    parser.add_argument("--shape", type=str, default="sphere", choices=["sphere", "box", "cylinder"])
    args = parser.parse_args()

    # 1. Initialization
    xml_str = get_xml(args.shape)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    sim_helper = MockSim(model, data)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = "/app/fots_sim"
    if not os.path.exists(base_dir): base_dir = os.path.join(os.getcwd(), "fots_sim")
    
    # Load FOTS rendering assets
    background_img = np.load(os.path.join(base_dir, "assets/digit_bg.npy"))
    bg_ini_depth = np.load(os.path.join(base_dir, "utils/ini_depth_extent.npy"))
    bg_render_mlp = np.load(os.path.join(base_dir, "utils/ini_bg_mlp.npy"))
    
    mlp_model = MLP().to(device)
    mlp_model.load_state_dict(torch.load(os.path.join(base_dir, "models/mlp_n2c_r.pth"), map_location=device, weights_only=True))
    mlp_model.eval()
    
    fots_render = MLPRender(background_img=background_img, bg_depth=bg_ini_depth, bg_render=bg_render_mlp, model=mlp_model)
    depth_capture = TactileDepthCapture(sim_helper, height=900, width=675)
    world_renderer = mujoco.Renderer(model, height=900, width=900)
    
    # Baseline (No Contact)
    data.joint("obj_z").qpos[0] = 1.0 
    mujoco.mj_step(model, data)
    z_raw = depth_capture.render_depth_meters("tactile_cam")
    z_norm = meters_to_normalized_depth(sim_helper, np.fliplr(cv2.resize(z_raw, (240, 320), interpolation=cv2.INTER_AREA)))
    fots_render.bg_depth = z_norm
    fots_render._pre_scaled_bg = fots_render.bg_depth * fots_render._scale
    
    win_name = f"FOTS Primitive Sandbox: {args.shape.upper()}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("X-P", win_name, 100, 200, lambda x: None)
    cv2.createTrackbar("Y-P", win_name, 100, 200, lambda x: None)
    cv2.createTrackbar("Z-P", win_name, 100, 200, lambda x: None)

    while True:
        data.joint("obj_x").qpos[0] = (cv2.getTrackbarPos("X-P", win_name) - 100) * 0.0005
        data.joint("obj_y").qpos[0] = (cv2.getTrackbarPos("Y-P", win_name) - 100) * 0.0005
        # 0.100 is contact point
        data.joint("obj_z").qpos[0] = (cv2.getTrackbarPos("Z-P", win_name) - 100) * 0.0005
        
        mujoco.mj_step(model, data)
        world_renderer.update_scene(data, camera="world_cam")
        world_bgr = cv2.cvtColor(world_renderer.render(), cv2.COLOR_RGB2BGR)
        
        z_curr = depth_capture.render_depth_meters("tactile_cam")
        z_norm_curr = meters_to_normalized_depth(sim_helper, np.fliplr(cv2.resize(z_curr, (240, 320), interpolation=cv2.INTER_AREA)))
        
        tactile_rgb = fots_render.generate(z_norm_curr)
        tactile_bgr = cv2.cvtColor(cv2.resize(tactile_rgb, (675, 900), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)
        
        cv2.imshow(win_name, np.hstack([world_bgr, tactile_bgr]))
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cv2.destroyAllWindows()
    depth_capture.close()
    world_renderer.close()

if __name__ == "__main__": main()
