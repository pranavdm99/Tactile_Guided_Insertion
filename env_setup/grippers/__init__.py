from robosuite.models.grippers import GRIPPER_MAPPING
from env_setup.grippers.fots_panda import FOTSPandaGripper

# Register custom gripper locally if not already present
if "FOTSPandaGripper" not in GRIPPER_MAPPING:
    GRIPPER_MAPPING["FOTSPandaGripper"] = FOTSPandaGripper
