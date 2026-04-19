#!/usr/bin/env python3
"""
FOTS-Enabled Teleoperation script for Robosuite Panda with tactile sensing.
Allows manual control using Mouse (XY Translation, Scroll Z) and Keyboard (Arrows/PgUpDn Rotation).
Displays real-time tactile imprints from FOTS sensors.
"""

import argparse
import numpy as np
import cv2
import robosuite as suite
from pynput import keyboard, mouse
from robosuite import load_composite_controller_config
from env_setup.make_env import make_fots_env

class HybridDevice:
    """
    Combines Keyboard and Mouse input for 6-DOF robot control.
    """
    def __init__(self, pos_sensitivity=0.2, rot_sensitivity=0.05, scroll_sensitivity=0.5):
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.scroll_sensitivity = scroll_sensitivity

        # State
        self.dpos = np.zeros(3)
        self.raw_drotation = np.zeros(3)
        self.grasp = False
        self.reset_state = False
        self.quit = False

        # Mouse tracking
        self.last_mouse_pos = None

        # Listeners
        self.kb_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.m_listener = mouse.Listener(on_move=self.on_move, on_scroll=self.on_scroll)

    def start(self):
        self.kb_listener.start()
        self.m_listener.start()

    def stop(self):
        self.kb_listener.stop()
        self.m_listener.stop()

    def on_move(self, x, y):
        if self.last_mouse_pos is None:
            self.last_mouse_pos = (x, y)
            return
        
        dx = x - self.last_mouse_pos[0]
        dy = y - self.last_mouse_pos[1]
        self.last_mouse_pos = (x, y)

        self.dpos[0] += dy * self.pos_sensitivity
        self.dpos[1] += dx * self.pos_sensitivity

    def on_scroll(self, x, y, dx, dy):
        self.dpos[2] += dy * self.scroll_sensitivity

    def on_press(self, key):
        try:
            if key == keyboard.Key.up:
                self.raw_drotation[0] += self.rot_sensitivity
            elif key == keyboard.Key.down:
                self.raw_drotation[0] -= self.rot_sensitivity
            elif key == keyboard.Key.left:
                self.raw_drotation[2] += self.rot_sensitivity
            elif key == keyboard.Key.right:
                self.raw_drotation[2] -= self.rot_sensitivity
            elif key == keyboard.Key.page_up:
                self.raw_drotation[1] -= self.rot_sensitivity
            elif key == keyboard.Key.page_down:
                self.raw_drotation[1] += self.rot_sensitivity
            elif key == keyboard.Key.backspace:
                self.reset_state = True
            elif key == keyboard.Key.esc:
                self.quit = True
        except Exception:
            pass

    def on_release(self, key):
        if key == keyboard.Key.enter:
            self.grasp = not self.grasp

    def get_controller_state(self):
        state = {
            "dpos": self.dpos.copy(),
            "raw_drotation": self.raw_drotation.copy(),
            "grasp": self.grasp,
            "reset": self.reset_state,
            "quit": self.quit
        }
        self.dpos.fill(0)
        self.raw_drotation.fill(0)
        self.reset_state = False
        return state

def main():
    parser = argparse.ArgumentParser(description="Panda FOTS Teleop (Mouse + Keyboard + Tactile)")
    parser.add_argument("--env", type=str, default="NutAssemblySingle", help="Environment name")
    parser.add_argument("--horizon", type=int, default=1000, help="Max number of steps")
    parser.add_argument("--camera", type=str, default="agentview", help="Camera name")
    parser.add_argument("--fast", action="store_true", help="Use fast depth visualization instead of FOTS MLP")
    parser.add_argument("--no-tactile-display", action="store_true", help="Disable tactile image display window")
    parser.add_argument("--nut", type=str, default=None, choices=["round", "square"], help="Nut type to use")
    args = parser.parse_args()

    # 1. Initialize environment with FOTS Panda Gripper
    print("[INFO] Initializing FOTS Environment with Tactile Sensing...")
    
    env = make_fots_env(
        env_name=args.env,
        fidelity_mode=not args.fast,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        horizon=args.horizon,
        render_camera=args.camera,
        nut_type=args.nut,
    )

    # 2. Initialize Hybrid Device
    device = HybridDevice()
    device.start()
    
    # 3. Setup tactile display window if enabled
    tactile_win = None
    if not args.no_tactile_display:
        tactile_win = "FOTS Tactile Sensors"
        cv2.namedWindow(tactile_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(tactile_win, 1350, 900)  # Width for 2 sensors side-by-side
    
    print("[INFO] Teleoperation Started (FOTS-Enabled Robosuite).")
    print("Controls: Mouse=Pos, Arrows/PgUpDn=Rot, Enter=Grasp, Backspace=Reset, ESC=Quit")
    if not args.no_tactile_display:
        print(f"[INFO] Tactile Mode: {'Fast Depth' if args.fast else 'FOTS MLP Rendering'}")

    obs = env.reset()
    env.render()
    
    # Enable geometry group 4 (gripper visualization) AFTER first render
    # This ensures the viewer is initialized before we modify vopt
    # Note: env is wrapped, so base env is accessed via env.env
    base_env = env.env if hasattr(env, 'env') else env
    
    # For on-screen mjviewer renderer (uses mujoco.viewer with .opt attribute)
    if hasattr(base_env, 'viewer') and base_env.viewer is not None:
        # The MjviewerRenderer creates the actual viewer lazily in update()
        # Access via base_env.viewer.viewer (the inner mujoco viewer)
        if hasattr(base_env.viewer, 'viewer') and base_env.viewer.viewer is not None:
            # Use .opt (not .vopt) for the mujoco passive viewer
            if hasattr(base_env.viewer.viewer, 'opt'):
                base_env.viewer.viewer.opt.geomgroup[4] = 1
                print("[INFO] Enabled geometry group 4 (gripper visualization) in mjviewer")
    
    # For offscreen renderer (used for camera observations and persists across resets)
    if hasattr(base_env, 'sim') and hasattr(base_env.sim, '_render_context_offscreen'):
        if base_env.sim._render_context_offscreen is not None:
            ctx = base_env.sim._render_context_offscreen
            if hasattr(ctx, 'vopt'):
                ctx.vopt.geomgroup[4] = 1
                print("[INFO] Enabled geometry group 4 in offscreen renderer (persists across resets)")

    try:
        while not device.quit:
            state = device.get_controller_state()
            
            if state["reset"]:
                print("[INFO] Resetting...")
                obs = env.reset()
                env.render()
                # Re-enable geometry group 4 after reset
                if hasattr(base_env, 'viewer') and base_env.viewer is not None:
                    if hasattr(base_env.viewer, 'viewer') and base_env.viewer.viewer is not None:
                        if hasattr(base_env.viewer.viewer, 'opt'):
                            base_env.viewer.viewer.opt.geomgroup[4] = 1
                continue

            # Standard 7-DOF action: [dx, dy, dz, ax, ay, az, grasp]
            # FOTS gripper mirrors both fingers from single control (1-DOF parallel jaw)
            # +1 = closed, -1 = open
            action = np.zeros(env.action_dim)
            action[:3] = state["dpos"]
            action[3:6] = state["raw_drotation"]
            action[6] = 1.0 if state["grasp"] else -1.0

            obs, reward, done, info = env.step(action)
            env.render()
            
            # Re-enable geometry group 4 after each render (in case viewer was recreated)
            if hasattr(base_env, 'viewer') and base_env.viewer is not None:
                if hasattr(base_env.viewer, 'viewer') and base_env.viewer.viewer is not None:
                    if hasattr(base_env.viewer.viewer, 'opt'):
                        base_env.viewer.viewer.opt.geomgroup[4] = 1
            
            # Display tactile observations if available
            if not args.no_tactile_display and "tactile_left" in obs and "tactile_right" in obs:
                # Convert from RGB (wrapper format) to BGR (OpenCV display format)
                tactile_left_bgr = cv2.cvtColor(obs["tactile_left"], cv2.COLOR_RGB2BGR)
                tactile_right_bgr = cv2.cvtColor(obs["tactile_right"], cv2.COLOR_RGB2BGR)
                
                # Resize for better visibility (from 320x240 to 900x675 each)
                left_display = cv2.resize(tactile_left_bgr, (675, 900), interpolation=cv2.INTER_CUBIC)
                right_display = cv2.resize(tactile_right_bgr, (675, 900), interpolation=cv2.INTER_CUBIC)
                
                # Stack horizontally and display
                combined = np.hstack([left_display, right_display])
                cv2.imshow(tactile_win, combined)
                cv2.waitKey(1)

            if done:
                print("[INFO] Episode done. Resetting...")
                obs = env.reset()
                env.render()

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[INFO] Stopping Teleop...")
        device.stop()
        if tactile_win:
            cv2.destroyAllWindows()
        env.close()

if __name__ == "__main__":
    main()

