import numpy as np
import robosuite as suite
import robosuite.controllers as controllers
from robosuite.environments.manipulation.nut_assembly import NutAssembly
import env_setup  # Register FOTSPandaGripper
from env_setup.tactile_wrapper import TactileObservationWrapper

class FOTSNutAssemblySingle(NutAssembly):
    """
    Custom single-nut environment that correctly handles fixed vs random nut selection.
    
    Key fix: NutAssemblySingle hardcodes single_object_mode=1 which ALWAYS randomizes
    the nut on every reset, silently overriding any nut_type you pass. We subclass
    NutAssembly directly to correctly choose between:
      - single_object_mode=2 (fixed nut) when nut_type is provided
      - single_object_mode=1 (random nut) when nut_type is None
    
    Also dynamically hides the 'incorrect' peg and optionally randomizes peg position.
    """
    def __init__(self, randomize_peg=False, nut_type=None, **kwargs):
        self.randomize_peg = randomize_peg
        
        # Choose correct mode based on whether a nut_type is specified
        if nut_type is not None:
            # Fixed mode: always use the specified nut
            mode = 2
        else:
            # Random mode: randomly pick one nut per reset
            mode = 1
        
        super().__init__(single_object_mode=mode, nut_type=nut_type, **kwargs)

    def _load_model(self):
        super()._load_model()
        # Initial hiding if nut_id is already known (fixed mode)
        if hasattr(self, "nut_id"):
            self._apply_peg_hiding_to_xml()

    def _apply_peg_hiding_to_xml(self):
        """Hides the incorrect peg in the XML model before sim initialization."""
        arena = self.model.mujoco_arena
        # Reset both to defaults first
        arena.peg1_body.set("pos", "0.23 0.1 0.85")
        arena.peg2_body.set("pos", "0.23 -0.1 0.85")
        
        if self.nut_id == 0: # Square nut selected -> hide round peg (peg2)
            arena.peg2_body.set("pos", "0 0 -10")
        elif self.nut_id == 1: # Round nut selected -> hide square peg (peg1)
            arena.peg1_body.set("pos", "0 0 -10")

    def _reset_internal(self):
        super()._reset_internal()
        # After super() resolves the nut choice (for random mode), sync pegs
        self._sync_peg_visibility()
        
        # Optional: Randomize the active peg's position
        if self.randomize_peg:
            self._randomize_active_peg()

    def _sync_peg_visibility(self):
        """Hides the incorrect peg in the active simulation model."""
        peg1_id = self.sim.model.body_name2id("peg1")
        peg2_id = self.sim.model.body_name2id("peg2")
        
        # Default positions (from PegsArena XML)
        pos1 = np.array([0.23, 0.1, 0.85])
        pos2 = np.array([0.23, -0.1, 0.85])
        hidden_pos = np.array([0, 0, -10.0])
        
        if self.nut_id == 0: # Square nut -> show peg1 (square), hide peg2 (round)
            self.sim.model.body_pos[peg1_id] = pos1
            self.sim.model.body_pos[peg2_id] = hidden_pos
        elif self.nut_id == 1: # Round nut -> show peg2 (round), hide peg1 (square)
            self.sim.model.body_pos[peg1_id] = hidden_pos
            self.sim.model.body_pos[peg2_id] = pos2

    def _randomize_active_peg(self):
        """Moves the active peg to a random location on the table surface."""
        # 0 = SquareNut/peg1, 1 = RoundNut/peg2
        active_peg_id = self.sim.model.body_name2id("peg1" if self.nut_id == 0 else "peg2")
        
        # Safe region within robot workspace on table surface
        # Table center is at (0,0,0.82), surface at z=0.87
        # Peg base Z=0.85 keeps it partially embedded in the table
        new_x = self.rng.uniform(0.15, 0.35)
        new_y = self.rng.uniform(-0.25, 0.25)
        
        self.sim.model.body_pos[active_peg_id] = [new_x, new_y, 0.85]
        self.sim.forward()

def make_fots_env(
    env_name="NutAssemblySingle",
    fidelity_mode=True,
    control_type="OSC_POSE",
    render_width=240,
    render_height=320,
    randomize_peg=False,
    nut_type=None,
    **kwargs
):
    """
    Factory function for Robosuite 1.5.x + MuJoCo 3.x.
    Creates an environment with FOTSPandaGripper and integrated tactile sensing.
    """
    # 1. Load Controller Config
    controller_config = suite.load_composite_controller_config(robot="Panda")
    if control_type and control_type != "OSC_POSE":
         controller_config = suite.load_composite_controller_config(controller=control_type, robot="Panda")
    
    # 2. Create Base Environment
    env_cls = FOTSNutAssemblySingle if env_name == "NutAssemblySingle" else suite.environments.REGISTERED_ENVS[env_name]
    
    suite_kwargs = dict(
        robots="Panda",
        gripper_types="FOTSPandaGripper",
        controller_configs=controller_config,
        has_renderer=kwargs.pop("has_renderer", False),
        has_offscreen_renderer=kwargs.pop("has_offscreen_renderer", True),
        use_camera_obs=kwargs.pop("use_camera_obs", True),
        randomize_peg=randomize_peg,
        nut_type=nut_type,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=[256, 256],
        camera_widths=[256, 256],
        reward_shaping=True,   # Dense staged rewards: grasp→lift→hover→insert
    )
    suite_kwargs.update(kwargs)
    
    env = env_cls(**suite_kwargs)
    
    # 3. Apply Tactile Observation Wrapper
    env = TactileObservationWrapper(
        env, 
        fidelity_mode=fidelity_mode, 
        height=render_height, 
        width=render_width
    )
    
    return env
