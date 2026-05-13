import os
import argparse
import numpy as np
import cv2
import pygame
import robosuite as suite
from datetime import datetime
from env_setup.make_env import make_fots_env
from env_setup.utils.data_recorder import DataRecorder

class F310Controller:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise Exception("❌ F310 not detected! Make sure it is plugged in and switched to 'X' mode.")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()
        
        # --- SPEED TUNING PARAMETERS ---
        # Increase these to make the robot move/rotate faster!
        self.pos_sensitivity = 0.5  # Up from 0.15
        self.rot_sensitivity = 0.4  # Up from 0.2
        
        # State tracking
        self.grasp_val = -1.0  # -1 = Open, 1 = Closed
        self.recording = False
        self.recording_toggled = False
        self.reset_state = False
        self.quit = False
        
        print(f"🎮 Connected: {self.joy.get_name()}")

    def get_action(self):
        pygame.event.pump()
        
       # --- TRANSLATION ---
        dx = self.joy.get_axis(1) * self.pos_sensitivity   # L-Stick Y -> Move X
        dy = self.joy.get_axis(0) * self.pos_sensitivity   # L-Stick X -> Move Y
        
        # Move Z-axis (Up/Down) to Right Joystick Up/Down (Axis 4)
        # Note: Negative because pushing UP on a joystick usually reads as -1.0
        dz = -self.joy.get_axis(4) * self.pos_sensitivity             
        
        # --- ROTATION ---
        d_roll = 0.0
        d_pitch = 0.0  # Pitch is 0 now that Right Stick Up/Down is used for Z translation
        d_yaw = -self.joy.get_axis(3) * self.rot_sensitivity  # R-Stick Left/Right -> Yaw
        # --- GRIPPER ---
        try:
            lt = self.joy.get_axis(2)
            rt = self.joy.get_axis(5)
            if rt > 0.0:
                self.grasp_val = 1.0   # Close
            elif lt > 0.0:
                self.grasp_val = -1.0  # Open
        except:
            pass
            
        # --- BUTTONS ---
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 2:  # 'X' Button
                    self.recording = not self.recording
                    self.recording_toggled = True
                elif event.button == 0:  # 'A' Button
                    self.reset_state = True
                elif event.button == 7:  # 'Start' Button
                    self.quit = True
                    
        return np.array([dx, dy, dz, d_roll, d_pitch, d_yaw]), self.grasp_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="NutAssemblySingle")
    parser.add_argument("--out", type=str, default="/app/datasets")
    parser.add_argument("--fast", action="store_true", help="Use fast depth visualization to boost FPS")
    args = parser.parse_args()

    print("[INFO] Initializing Fast Environment...")
    env = make_fots_env(
        env_name=args.env, 
        fidelity_mode=not args.fast,  
        has_renderer=False,           
        has_offscreen_renderer=True,  
        use_camera_obs=True,
        render_gpu_device_id=0
    )
    
    base_env = env.env if hasattr(env, "env") else env
    
    recorder = DataRecorder(output_dir=args.out)
    recorder.set_robomimic_env_context({
        "env_name": args.env,
        "env_version": suite.__version__,
        "env_kwargs": {
            "env_name": args.env,
            "robots": "Panda",
            "gripper_types": "FOTSPandaGripper",
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "control_freq": getattr(base_env, "control_freq", 20),
        },
    })
    
    controller = F310Controller()
    obs = env.reset()

    cv2.namedWindow("Main Robot View", cv2.WINDOW_NORMAL)
    cv2.namedWindow("FOTS Tactile Sensors", cv2.WINDOW_NORMAL)

    print("\n[INFO] 🎮 Gamepad Teleop Ready!")
    print("---------------------------------")
    print(" L-Stick: Move X/Y | D-Pad: Move Z")
    print(" R-Stick: Rotate   | RT/LT: Gripper")
    print(" [X] Record        | [A] Reset")
    print(" [Start] Quit      |")
    print("---------------------------------\n")

    try:
        while not controller.quit:
            if controller.reset_state:
                print("[INFO] Resetting environment...")
                obs = env.reset()
                controller.reset_state = False
                if controller.recording:
                    recorder.save_episode(discard=True)
                    controller.recording = False
                continue

            action_pos_rot, grasp_val = controller.get_action()
            action = np.zeros(7)
            action[:6] = action_pos_rot
            action[6] = grasp_val
            
            obs_before = obs
            obs, reward, done, info = env.step(action)

            # Recording Logic
            if controller.recording:
                if controller.recording_toggled:
                    print("[INFO] 🔴 RECORDING STARTED")
                    recorder.start_episode()
                    controller.recording_toggled = False
                recorder.record_step(obs_before, action, reward, done)
            elif controller.recording_toggled:
                print("[INFO] ⏹️ RECORDING STOPPED. Saving data...")
                recorder.save_episode()
                controller.recording_toggled = False

            # --- VISUALIZATION ---
            if "agentview_image" in obs:
                img = cv2.cvtColor(obs["agentview_image"], cv2.COLOR_RGB2BGR)
                img = cv2.flip(img, 0)
                
                # FAST RESIZE: Scales up the raw pixels so it fills the screen without burning CPU
                img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_LINEAR)
                
                if controller.recording:
                    cv2.putText(img, "REC", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.circle(img, (8, 28), 8, (0, 0, 255), -1)
                    
                cv2.imshow("Main Robot View", img)

            if "tactile_left" in obs and "tactile_right" in obs:
                t_l = cv2.cvtColor(obs["tactile_left"], cv2.COLOR_RGB2BGR)
                t_r = cv2.cvtColor(obs["tactile_right"], cv2.COLOR_RGB2BGR)
                t_l = cv2.flip(t_l, 0)
                t_r = cv2.flip(t_r, 0)
                
                # Scale up tactile images
                t_l = cv2.resize(t_l, (450, 600), interpolation=cv2.INTER_LINEAR)
                t_r = cv2.resize(t_r, (450, 600), interpolation=cv2.INTER_LINEAR)
                
                cv2.imshow("FOTS Tactile Sensors", np.hstack([t_l, t_r]))

            cv2.waitKey(1)

            if done:
                print("[INFO] Episode complete! Resetting...")
                obs = env.reset()
                
    finally:
        if controller.recording:
            print("[INFO] Saving final recorded episode...")
            recorder.save_episode()
            
        cv2.destroyAllWindows()
        env.close()
        pygame.quit()

if __name__ == "__main__":
    main()