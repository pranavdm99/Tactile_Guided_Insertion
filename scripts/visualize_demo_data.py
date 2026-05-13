import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_demo(filepath, demo_id):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return

    with h5py.File(filepath, "r") as f:
        demo_path = f"data/{demo_id}"
        if demo_path not in f:
            print(f"Error: Demo {demo_id} not found in {filepath}")
            return
            
        demo = f[demo_path]
        actions = demo["actions"][:]
        rewards = demo["rewards"][:]
        dones = demo["dones"][:]
        
        # Extract tactile stats if available
        tactile_l = None
        if "tactile_left" in demo["obs"]:
            tactile_l = demo["obs/tactile_left"][:]
            tactile_r = demo["obs/tactile_right"][:]
            
        n_steps = actions.shape[0]
        time = np.arange(n_steps)
        
        # Create multi-panel figure
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        fig.suptitle(f"Demo Analysis: {demo_id} in {os.path.basename(filepath)}", fontsize=16)
        
        # 1. Actions Plot
        ax = axes[0]
        ax.plot(time, actions[:, 0], label="PosX")
        ax.plot(time, actions[:, 1], label="PosY")
        ax.plot(time, actions[:, 2], label="PosZ")
        ax.plot(time, actions[:, 6], label="Grasp", linewidth=3, linestyle="--")
        ax.set_ylabel("Normalized Actions")
        ax.set_title("End-Effector Control & Grasp")
        ax.legend(loc="upper right", ncol=4)
        ax.grid(True, alpha=0.3)
        
        # 2. Orientation (Optional but useful)
        ax = axes[1]
        ax.plot(time, actions[:, 3], label="OriX")
        ax.plot(time, actions[:, 4], label="OriY")
        ax.plot(time, actions[:, 5], label="OriZ")
        ax.set_ylabel("Rotation Delta")
        ax.set_title("EEF Orientation Actions")
        ax.legend(loc="upper right", ncol=3)
        ax.grid(True, alpha=0.3)
        
        # 3. Rewards and Dones
        ax = axes[2]
        ax.step(time, rewards, label="Reward", color="green", where="post")
        ax_d = ax.twinx()
        ax_d.step(time, dones, label="Done", color="red", alpha=0.5, where="post")
        ax.set_ylabel("Reward")
        ax_d.set_ylabel("Done (Binary)")
        ax.set_title("Reward & Termination")
        ax.grid(True, alpha=0.3)
        
        # 4. Tactile Intensity (The "Signal" check)
        ax = axes[3]
        if tactile_l is not None:
            # Calculate average intensity per frame (dynamic axis selection for 1-channel vs 3-channel)
            calc_axes = tuple(range(1, tactile_l.ndim))
            l_intensity = np.mean(tactile_l, axis=calc_axes)
            r_intensity = np.mean(tactile_r, axis=calc_axes)
            # Subtract baseline (first frame) to show delta
            l_delta = l_intensity - l_intensity[0]
            r_delta = r_intensity - r_intensity[0]
            
            ax.plot(time, l_delta, label="Left Intensity \u0394", color="blue")
            ax.plot(time, r_delta, label="Right Intensity \u0394", color="cyan")
            ax.set_ylabel("Avg Pixel Delta")
            ax.set_title("Tactile Contact Signal (Average Intensity Delta)")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No Tactile Data Found", ha="center")
            
        ax.set_xlabel("Timestep")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save output
        output_name = f"plots/{demo_id}_analysis.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(output_name)
        print(f"[SUCCESS] Plot saved to {output_name}")
        plt.close()
        return output_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to HDF5 dataset")
    parser.add_argument("--demo", type=str, default="demo_0", help="Demo ID to plot")
    args = parser.parse_args()
    
    plot_demo(args.file, args.demo)
