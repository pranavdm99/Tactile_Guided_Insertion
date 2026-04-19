import os
import sys
import traceback
import numpy as np
import mujoco
import cv2
import torch
import xml.etree.ElementTree as ET
import re
from fots_sim.utils.mlp_render import MLPRender
from fots_sim.mlp_model import MLP
from env_setup.custom_grippers import FOTSPandaGripper
from env_setup.tactile_depth_capture import (
    TactileDepthCapture, 
    meters_to_normalized_depth, 
    bandpass_gel_depth
)

def get_full_xml(shape="box"):
    """
    Constructs a standalone MuJoCo XML with the FOTSPandaGripper and a test object.
    """
    gripper = FOTSPandaGripper()
    gripper_root = gripper.root
    
    def get_all_children_xml(tag_name):
        xml_snippets = []
        for node in gripper_root.findall(tag_name):
            for child in node:
                xml_snippets.append(ET.tostring(child, encoding='unicode'))
        return "".join(xml_snippets)

    asset_str = get_all_children_xml("asset")
    world_str = get_all_children_xml("worldbody")
    actuator_str = get_all_children_xml("actuator")
    contact_str = get_all_children_xml("contact")

    # Final XML Scrubber: Ensure all gripper parts are in Group 4 for optical clarity
    root = ET.fromstring(f"<root>{world_str}</root>")
    for geom in root.iter("geom"):
        gname = geom.get("name", "").lower()
        if any(x in gname for x in ["finger", "hand", "gripper", "pad", "link"]):
            geom.set("group", "4")
    world_str = ET.tostring(root, encoding='unicode')[6:-7]

    geom_xml = ""
    col_str = 'contype="1" conaffinity="1"'
    if shape == "sphere":
        geom_xml = f'<geom name="prim" type="sphere" size="0.015" rgba="0.8 0.2 0.2 1.0" group="1" condim="4" friction="1 0.05 0.01" {col_str}/>'
    elif shape == "box":
        geom_xml = f'<geom name="prim" type="box" size="0.007 0.012 0.015" rgba="0.8 0.2 0.2 1.0" group="1" condim="4" friction="1 0.05 0.01" {col_str}/>'
    elif shape == "cylinder":
        geom_xml = f'<geom name="prim" type="cylinder" size="0.007 0.015" rgba="0.8 0.2 0.2 1.0" group="1" condim="4" friction="1 0.05 0.01" {col_str}/>'

    return f"""
<mujoco model="franka_gripper_fots_verified">
    <option integrator="Euler" timestep="0.004" solver="Newton" cone="elliptic"/>
    <visual><map znear="0.001" zfar="10.0"/><global offwidth="2560" offheight="1440"/></visual>
    <asset>
        <texture type="2d" name="texplane" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2" width="512" height="512"/>
        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
        {asset_str}
    </asset>
    <worldbody>
        <light pos="0 0 2" dir="0 0 -1" diffuse="1.2 1.2 1.2" ambient="0.4 0.4 0.4"/>
        <geom name="floor" type="plane" size="0.5 0.5 0.01" pos="0 0 0" material="matplane" group="0"/>
        <body name="gripper_base" pos="0 0 0.1"> {world_str} </body>
        <body name="object_mocap" mocap="true" pos="0 0 0.5"> {geom_xml} </body>
        <camera name="world_cam" pos="0.5 0.5 0.5" mode="targetbody" target="gripper_base" fovy="60"/>
    </worldbody>
    <actuator> {actuator_str} </actuator>
    <contact> {contact_str} </contact>
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
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--shape", type=str, default="box", choices=["sphere", "box", "cylinder"])
        parser.add_argument("--headless", action="store_true")
        args = parser.parse_args()

        # 1. Initialization
        xml_str = get_full_xml(args.shape)
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
        depth_capture = TactileDepthCapture(sim_helper, height=320, width=240)
        world_renderer = mujoco.Renderer(model, height=900, width=900)
        
        # 2. Rendering Options (Stricter Site Filtering)
        vopt_tactile = mujoco.MjvOption()
        for i in range(6): 
            vopt_tactile.geomgroup[i] = 0
            vopt_tactile.sitegroup[i] = 0
        vopt_tactile.geomgroup[1] = 1 # Show objects 
        
        vopt_world = mujoco.MjvOption()
        vopt_world.geomgroup[4] = 1 # Show gripper
        vopt_world.geomgroup[1] = 1 # Show objects
        
        def process(z, side="left"):
            # Symmetrical orientation Sync
            if side == "left": z = np.fliplr(z)
            else: z = np.flipud(z)
            z_clamped = bandpass_gel_depth(z, 0.0225, far_cap_m=0.010)
            return meters_to_normalized_depth(sim_helper, z_clamped)

        def feed(cam, baseline, side="left"):
            z = depth_capture.render_depth_meters_batched(sim_helper, [cam], scene_option=vopt_tactile)[0]
            curr = process(z, side=side)
            fots_render.bg_depth = baseline
            fots_render._pre_scaled_bg = fots_render.bg_depth * fots_render._scale
            rgb = fots_render.generate(curr)
            return cv2.cvtColor(cv2.resize(rgb, (675, 900), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)

        # Baseline (No Contact)
        data.mocap_pos[0][2] = -5.0
        mujoco.mj_step(model, data)

        # Discovery: Find the actual camera names (which have prefixes)
        cam_l, cam_r = None, None
        for i in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if "tactile_cam_left" in name: cam_l = name
            if "tactile_cam_right" in name: cam_r = name
        
        if not cam_l or not cam_r:
            print("[ERROR] Tactile cameras not found in model!")
            return

        baselines = depth_capture.render_depth_meters_batched(sim_helper, [cam_l, cam_r], scene_option=vopt_tactile)
        b_l = process(baselines[0], "left")
        b_r = process(baselines[1], "right")

        if args.headless:
            # Verified Centered Contact
            data.mocap_pos[0] = [0, 0, 0.205]
            data.ctrl[0], data.ctrl[1] = -0.01, 0.01
            for _ in range(100): mujoco.mj_step(model, data)
            combined = np.hstack([feed(cam_l, b_l, "left"), feed(cam_r, b_r, "right")])
            cv2.imwrite("/app/tactile_verification.png", combined)
            print("[SUCCESS] Headless verification saved to /app/tactile_verification.png")
            return

        win = "FOTS Franka Sandbox"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("GRIP", win, 100, 200, lambda x: None)
        cv2.createTrackbar("OBJ_X", win, 100, 200, lambda x: None)
        cv2.createTrackbar("OBJ_Y", win, 100, 200, lambda x: None)
        cv2.createTrackbar("OBJ_Z", win, 300, 300, lambda x: None)

        print("[SUCCESS] Franka FOTS Sandbox Active. GRIP > 100 for imprints.")

        while True:
            gv = (100 - cv2.getTrackbarPos("GRIP", win)) / 2500.0
            data.ctrl[0], data.ctrl[1] = gv, -gv
            ox = (cv2.getTrackbarPos("OBJ_X", win) - 100) * 0.001
            oy = (cv2.getTrackbarPos("OBJ_Y", win) - 100) * 0.001
            oz = cv2.getTrackbarPos("OBJ_Z", win) / 1000.0
            data.mocap_pos[0] = [ox, oy, oz]
            
            for _ in range(5): mujoco.mj_step(model, data)
            world_renderer.update_scene(data, camera="world_cam", scene_option=vopt_world)
            w_bgr = cv2.cvtColor(world_renderer.render(), cv2.COLOR_RGB2BGR)
            t_l = feed(cam_l, b_l, "left")
            t_r = feed(cam_r, b_r, "right")
            cv2.imshow(win, np.hstack([w_bgr, t_l, t_r]))
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__": main()
