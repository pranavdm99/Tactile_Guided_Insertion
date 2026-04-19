import os
import numpy as np
from robosuite.models.grippers import GripperModel

class FOTSPandaGripper(GripperModel):
    """
    Custom Franka Gripper with integrated FOTS (Fused Optical Tactile Sensor) cameras.
    Loads the MJCF from env_setup/models/fots_panda_gripper.xml.
    
    Mimics the standard PandaGripper with 1-DOF parallel jaw control,
    where both fingers move symmetrically (mirrored).
    """
    def __init__(self, idn=0):
        # Locate the MJCF file relative to this script
        base_path = os.path.dirname(__file__)
        xml_path = os.path.join(base_path, "models/fots_panda_gripper.xml")
        
        super().__init__(xml_path, idn=idn)
        
        # Initialize current action state for velocity-based control
        self.current_action = np.zeros(2)

    @property
    def init_qpos(self):
        return [0.02, -0.02]

    @property
    def _important_sites(self):
        return {
            "grip_site": "grip_site",
            "grip_cylinder": "grip_site_cylinder",
            "ee": "ee",
            "ee_x": "ee_x",
            "ee_y": "ee_y",
            "ee_z": "ee_z",
        }

    def format_action(self, action):
        """
        Maps continuous action into mirrored control signals for both fingers.
        Follows the same pattern as standard PandaGripper with velocity-based control.
        
        Args:
            action (np.array): Single-element array where:
                    -1 => open
                    +1 => closed
        
        Returns:
            2-element array with mirrored control signals for both fingers
        
        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof, f"Expected action dim {self.dof}, got {len(action)}"
        
        # Velocity-based control: accumulate action with speed factor
        # Both fingers move in opposite directions (mirrored)
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action),
            -1.0,
            1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.1

    @property
    def dof(self):
        return 1  # Single DOF: mirrored parallel jaw gripper
