#!/usr/bin/env python3
"""
FOTS Reconstruction Verification Tool
Loads a snapshot captured during teleoperation and verifies that 
raw depth can be accurately converted back to photorealistic FOTS imprints.
"""
import os
import numpy as np
import cv2
import torch
import sys

# Ensure we can import from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fots_sim.mlp_model import MLP
from fots_sim.utils.mlp_render import MLPRender

def main():
    # 1. Load Snapshot
    snapshot_file = os.path.join(os.path.dirname(__file__), "snapshot_depth.npz")
    if not os.path.exists(snapshot_file):
        print(f"[ERROR] Snapshot not found at {snapshot_file}. Press 'S' during teleop first.")
        return
    
    print(f"[INFO] Loading snapshot from {snapshot_file}...")
    data = np.load(snapshot_file)
    depth_l = data["tactile_left"]
    depth_r = data["tactile_right"]
    baseline_l = data["baseline_left"]
    baseline_r = data["baseline_right"]
    
    # 2. Initialize FOTS Engine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing FOTS Engine on {device}...")
    
    # Paths (relative to script location)
    base_dir = os.path.join(os.path.dirname(__file__), "..", "fots_sim")
    
    try:
        bg_img = np.load(os.path.join(base_dir, "assets/digit_bg.npy"))
        bg_depth = np.load(os.path.join(base_dir, "utils/ini_depth_extent.npy"))
        bg_mlp = np.load(os.path.join(base_dir, "utils/ini_bg_mlp.npy"))
        
        model = MLP().to(device)
        model_path = os.path.join(base_dir, "models/mlp_n2c_r.pth")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        fots_render = MLPRender(background_img=bg_img, bg_depth=bg_depth, bg_render=bg_mlp, model=model)
    except Exception as e:
        print(f"[ERROR] Failed to initialize FOTS engine: {e}")
        return
    
    # 3. Perform FOTS Reconstruction
    print("[INFO] Reconstructing tactile imprints from raw depth...")
    
    def render_fots(depth, baseline):
        fots_render.bg_depth = baseline
        fots_render._pre_scaled_bg = fots_render.bg_depth * fots_render._scale
        return fots_render.generate(depth)

    fots_l = render_fots(depth_l, baseline_l)
    fots_r = render_fots(depth_r, baseline_r)
    
    # 4. Prepare Comparison Window (4-panel)
    # Top Panel: Raw Depth (with Heatmap)
    def to_heatmap(z, baseline):
        diff = baseline - z
        # Use our standard 1cm sensitivity range
        z_v = (np.clip(diff / 0.01, 0.0, 1.0) * 255).astype(np.uint8)
        return cv2.applyColorMap(z_v, cv2.COLORMAP_JET)

    heat_l = to_heatmap(depth_l, baseline_l)
    heat_r = to_heatmap(depth_r, baseline_r)
    
    # Bottom Panel: FOTS Render
    # Convert FOTS (RGB) to BGR for OpenCV display
    fots_l_bgr = cv2.cvtColor(fots_l, cv2.COLOR_RGB2BGR)
    fots_r_bgr = cv2.cvtColor(fots_r, cv2.COLOR_RGB2BGR)
    
    # Combine Panels
    # Raw Depth row
    top_row = np.hstack([heat_l, heat_r])
    # FOTS Render row
    btm_row = np.hstack([fots_l_bgr, fots_r_bgr])
    
    # Full 4-panel stack
    combined = np.vstack([top_row, btm_row])
    
    # Add Annotations
    cv2.putText(combined, "TOP: RAW DEPTH HEATMAP", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "BOTTOM: FOTS RECONSTRUCTION", (20, combined.shape[0]//2 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display Result
    win_name = "FOTS Reconstruction Verification"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1200, 1200)
    cv2.imshow(win_name, combined)
    
    print("\n[SUCCESS] Displaying comparison window.")
    print("  -> Top Panel: What you saw in '--fast' teleop.")
    print("  -> Bottom Panel: What FOTS generated from that raw data.")
    print("\nPress any key in the window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
