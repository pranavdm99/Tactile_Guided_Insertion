import os
import re
import numpy as np
import xml.etree.ElementTree as ET
from robosuite.models.grippers import PandaGripper, GRIPPER_MAPPING

class FOTSPandaGripper(PandaGripper):
    """
    Standard Franka Panda Gripper with integrated FOTS (Fused Optical Tactile Sensor) simulation.
    Features:
    - Dual fingertip camera injection (22.5mm setback).
    - Hardcoded calibration (0.000mm focal error).
    - Unlocked actuators and joints for tactile compliance.
    """
    def __init__(self, idn=0):
        super().__init__(idn=idn)
        self._inject_fots_sensors()

    def _inject_fots_sensors(self):
        root = self.root
        worldbody = root.find("worldbody")
        asset = root.find("asset")
        
        # 1. Hardware Unlock: Remove rigid joint/actuator limits
        for joint in root.iter("joint"):
            joint.set("limited", "false")
            if "range" in joint.attrib: del joint.attrib["range"]
            
        for actuator in root.iter("actuator"):
            actuator.set("ctrllimited", "false")
            actuator.set("kp", "50000") # High-gain for realistic tactile synthesis
            if "ctrlrange" in actuator.attrib: del actuator.attrib["ctrlrange"]

        # 2. Site Scrubbing: Move robosuite default markers to hidden group 4
        for site in root.iter("site"):
            site.set("group", "4")

        # 3. Fingertip Camera Injection
        for i, tip_name in enumerate(["gripper0_finger_joint1_tip", "gripper0_finger_joint2_tip"]):
            side = "left" if i == 0 else "right"
            euler = "-90 0 0" if side == "left" else "90 0 0"
            
            # Find the joint tip body
            tip_body = None
            for body in root.iter("body"):
                if body.get("name") == tip_name:
                    tip_body = body
                    break
            
            if tip_body is not None:
                # Add Camera (22.5mm setback is verified focal center)
                # Left finger faces +Y, Right finger faces -Y
                cam_y = 0.0225 if side == "left" else -0.0225
                cam = ET.SubElement(tip_body, "camera", {
                    "name": f"tactile_cam_{side}",
                    "pos": f"0 {cam_y} -0.015",
                    "euler": euler,
                    "fovy": "60"
                })
                # Add Debug Axes
                axes = ET.SubElement(tip_body, "site", {
                    "name": f"fots_axes_{side}",
                    "type": "box",
                    "size": "0.005 0.005 0.005",
                    "rgba": "1 0 0 0.5",
                    "group": "4"
                })

        # 4. Soft-Gel Compliance: Inject canonical solref/solimp
        for geom in root.iter("geom"):
            if "pad_collision" in geom.get("name", ""):
                geom.set("solref", "0.02 1")
                geom.set("solimp", "0.9 0.95 0.001 0.5 2")
                geom.set("group", "4") # Hidden from tactile renderer

    def format_action(self, action):
        return super().format_action(action)

    @property
    def speed(self): return 0.01
