"""
FOTS Integration Setup
Registers custom grippers and provides environment creation utilities.
"""

from robosuite.models.grippers import GRIPPER_MAPPING
from env_setup.custom_grippers import FOTSPandaGripper

# Register custom gripper once at import time
if "FOTSPandaGripper" not in GRIPPER_MAPPING:
    GRIPPER_MAPPING["FOTSPandaGripper"] = FOTSPandaGripper
    print("[INFO] FOTSPandaGripper registered successfully")
