#!/usr/bin/env python3
import h5py
import numpy as np
import cv2
import argparse
import os
import sys

# Optional FOTS imports (initialized only if needed for reconstruction)
try:
    import torch
    # Ensure project root is in path
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from fots_sim.mlp_model import MLP
    from fots_sim.utils.mlp_render import MLPRender
except ImportError:
    torch = None

def validate_hdf5(filepath):
    """
    Validates a Robomimic-style HDF5 dataset.
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return

    print(f"--- Validating Dataset: {filepath} ---")
    with h5py.File(filepath, "r") as f:
        # 1. Global Metadata
        total_demos = f.attrs.get("total", len(f["data"].keys()))
        print(f"Total Demos: {total_demos}")
        print(f"Env Name: {f.attrs.get('env_name', 'Unknown')}")
        
        data = f["data"]
        demo_keys = list(data.keys())
        if not demo_keys:
            print("[ERROR] No demos found in 'data' group.")
            return

        # 2. Sample arbitrary demo for key checking
        sample_demo = data[demo_keys[0]]
        print("\nStructure of sample demo (demo_0):")
        print(f"  Actions: {sample_demo['actions'].shape}")
        print(f"  Rewards: {sample_demo['rewards'].shape}")
        print(f"  Dones:   {sample_demo['dones'].shape}")
        
        print("  Observations:")
        obs = sample_demo["obs"]
        for k in sorted(obs.keys()):
            # Highlight standardized object keys
            prefix = "⭐ " if k.startswith("object_") else "  "
            print(f"    {prefix} {k}: {obs[k].shape} ({obs[k].dtype})")

        # 4. Per-Demo Attributes
        print("\nDemo Attributes:")
        for k, v in sample_demo.attrs.items():
            desc = ""
            if k == "nut_type": desc = "(0=Square, 1=Round)"
            elif k == "render_type": desc = "(0=Fast, 1=Fidelity)"
            print(f"  - {k}: {v} {desc}")

        # 3. Deep validation: check for blank tactile images
        print("\nVerifying Tactile Signal...")
        blank_count = 0
        total_steps = 0
        
        for d_key in demo_keys:
            d = data[d_key]
            t_l = d["obs"]["tactile_left"][:]
            t_r = d["obs"]["tactile_right"][:]
            total_steps += t_l.shape[0]
            
            # Check if all pixels are the same (unlikely in real sensor data)
            for step in range(t_l.shape[0]):
                pixel_val_l = t_l[step].flatten()[0]
                pixel_val_r = t_r[step].flatten()[0]
                if np.all(t_l[step] == pixel_val_l) and np.all(t_r[step] == pixel_val_r):
                    blank_count += 1
        
        if blank_count == 0:
            print(f"[SUCCESS] No blank tactile frames detected across {total_steps} steps.")
        else:
            print(f"[WARNING] Detected {blank_count} potential blank frames.")

def draw_timeseries_panel(actions, rewards, dones, current_idx, width=400, height=1000):
    """
    Renders a live-updating plotting panel for demo data.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    n_steps = actions.shape[0]
    
    # 1. Coordinate Mapping
    pix_per_step = width / n_steps
    
    # Header
    cv2.putText(panel, "ANALYTICS PANEL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # 2. Plot Actions (Top 400px) - PosX(R), PosY(G), PosZ(B), Grasp(W)
    a_h = 400
    a_y_off = 50
    cv2.putText(panel, "Actions (7-DOF)", (10, a_y_off + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Draw zero-line
    cv2.line(panel, (0, a_y_off + a_h//2), (width, a_y_off + a_h//2), (50, 50, 50), 1)

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] # BGR: R, G, B
    for dim in range(3):
        pts = []
        for s in range(n_steps):
            x = int(s * pix_per_step)
            # Map -1..1 to 0..a_h
            y = int(a_y_off + a_h/2 - actions[s, dim] * (a_h/2.2))
            pts.append((x, y))
        
        # Draw previous segments
        for s in range(1, current_idx + 1):
            cv2.line(panel, pts[s-1], pts[s], colors[dim], 1 if s > current_idx else 2)

    # Grasp (Dashed White)
    for s in range(1, current_idx + 1):
        x1, y1 = int((s-1)*pix_per_step), int(a_y_off + a_h/2 - actions[s-1, 6] * (a_h/2.2))
        x2, y2 = int(s*pix_per_step), int(a_y_off + a_h/2 - actions[s, 6] * (a_h/2.2))
        cv2.line(panel, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # 3. Plot Rewards (Middle 300px)
    r_h = 300
    r_y_off = 500
    cv2.putText(panel, "Rewards (Green)", (10, r_y_off + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    if np.max(rewards) > 0:
        for s in range(1, current_idx + 1):
            x1, y1 = int((s-1)*pix_per_step), int(r_y_off + r_h - (rewards[s-1]/np.max(rewards)) * r_h)
            x2, y2 = int(s*pix_per_step), int(r_y_off + r_h - (rewards[s]/np.max(rewards)) * r_h)
            cv2.line(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 4. Plot Dones (Bottom 100px)
    d_y_off = 850
    cv2.putText(panel, "Episode Done (Red)", (10, d_y_off + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    for s in range(1, current_idx + 1):
        if dones[s-1]:
            x = int(s * pix_per_step)
            cv2.line(panel, (x, d_y_off + 50), (x, d_y_off + 100), (0, 0, 255), 3)

    # 5. Playhead
    px = int(current_idx * pix_per_step)
    cv2.line(panel, (px, 50), (px, height-50), (0, 100, 255), 2)
    
    return panel

def init_fots_engine():
    """Lazily initializes the FOTS rendering engine."""
    if torch is None:
        print("[ERROR] PyTorch or FOTS modules not found. Cannot reconstruct.")
        return None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing FOTS Engine on {device}...")
    
    base_dir = os.path.join(os.path.dirname(__file__), "..", "fots_sim")
    if not os.path.exists(base_dir):
        # Fallback for container absolute path
        base_dir = "/app/fots_sim"

    try:
        bg_img = np.load(os.path.join(base_dir, "assets/digit_bg.npy"))
        bg_depth = np.load(os.path.join(base_dir, "utils/ini_depth_extent.npy"))
        bg_mlp = np.load(os.path.join(base_dir, "utils/ini_bg_mlp.npy"))
        
        model = MLP().to(device)
        model_path = os.path.join(base_dir, "models/mlp_n2c_r.pth")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        return MLPRender(background_img=bg_img, bg_depth=bg_depth, bg_render=bg_mlp, model=model)
    except Exception as e:
        print(f"[ERROR] Failed to initialize FOTS engine: {e}")
        return None

def play_demos(filepath, fps=20, use_recon=True):
    """
    Plays back the recorded demonstrations with overlays and analytics.
    """
    W_VIS = 600
    W_PLOT = 400
    H_TOTAL = 800
    H_AGENT = 450
    H_TACTILE = 350

    with h5py.File(filepath, "r") as f:
        data = f["data"]
        for d_key in sorted(data.keys(), key=lambda x: int(x.split("_")[1])):
            print(f"Playing {d_key}...")
            demo = data[d_key]
            obs = demo["obs"]
            
            # Data for plotting
            actions = demo["actions"][:]
            rewards = demo["rewards"][:]
            dones = demo["dones"][:]
            
            # Observations
            t_l = obs["tactile_left"][:]
            t_r = obs["tactile_right"][:]
            v_key = "agentview_image"
            has_vis = v_key in obs
            
            num_steps = t_l.shape[0]
            
            # 2. Check if we need FOTS reconstruction
            is_fast_mode = (t_l.dtype == np.float32)
            fots_render = None
            if is_fast_mode and use_recon:
                if not hasattr(play_demos, "_fots_engine"):
                    play_demos._fots_engine = init_fots_engine()
                fots_render = play_demos._fots_engine
            
            # Use the first frame as the baseline for visualization for this demo
            # (Raw depth only)
            baseline_l = t_l[0].copy() if is_fast_mode else None
            baseline_r = t_r[0].copy() if is_fast_mode else None
            
            for i in range(num_steps):
                # 1. Prepare tactile views
                tl_frame = t_l[i]
                tr_frame = t_r[i]
                
                if is_fast_mode:
                    if fots_render:
                        # Photorealistic Reconstruction
                        def render_fots(depth, baseline):
                            fots_render.bg_depth = baseline
                            fots_render._pre_scaled_bg = fots_render.bg_depth * fots_render._scale
                            return fots_render.generate(depth)
                        
                        left_view_rgb = render_fots(tl_frame, baseline_l)
                        right_view_rgb = render_fots(tr_frame, baseline_r)
                        left_view_bgr = cv2.cvtColor(left_view_rgb, cv2.COLOR_RGB2BGR)
                        right_view_bgr = cv2.cvtColor(right_view_rgb, cv2.COLOR_RGB2BGR)
                    else:
                        # Faster Heatmap view (Fallback or --no-recon)
                        diff_l = baseline_l - tl_frame
                        diff_r = baseline_r - tr_frame
                        
                        def to_heatmap(diff):
                            z_u8 = (np.clip(diff / 0.01, 0.0, 1.0) * 255).astype(np.uint8)
                            return cv2.applyColorMap(z_u8, cv2.COLORMAP_JET)
                        
                        left_view_bgr = to_heatmap(diff_l)
                        right_view_bgr = to_heatmap(diff_r)
                else:
                    # Already RGB (uint8)
                    left_view_bgr = cv2.cvtColor(tl_frame, cv2.COLOR_RGB2BGR)
                    right_view_bgr = cv2.cvtColor(tr_frame, cv2.COLOR_RGB2BGR)

                left_view = cv2.resize(left_view_bgr, (W_VIS//2, H_TACTILE), interpolation=cv2.INTER_NEAREST)
                right_view = cv2.resize(right_view_bgr, (W_VIS//2, H_TACTILE), interpolation=cv2.INTER_NEAREST)
                tactile_row = np.hstack([left_view, right_view])
                
                playback_canvas = np.zeros((H_TOTAL, W_VIS, 3), dtype=np.uint8)
                if has_vis:
                    vis_frame = obs[v_key][i]
                    vis_frame = np.flip(vis_frame, axis=0) # MuJoCo flip
                    vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                    vis_frame_res = cv2.resize(vis_frame_bgr, (W_VIS, H_AGENT))
                    playback_canvas[:H_AGENT, :W_VIS] = vis_frame_res
                
                playback_canvas[H_AGENT:, :W_VIS] = tactile_row
                cv2.putText(playback_canvas, f"{d_key} | Step: {i}/{num_steps}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 2. Create Analytics Panel
                analytics_panel = draw_timeseries_panel(actions, rewards, dones, i, width=W_PLOT, height=H_TOTAL)
                
                # 3. Combine Panels
                final_canvas = np.hstack([playback_canvas, analytics_panel])
                
                cv2.imshow("Dataset Validation & Analytics", final_canvas)
                if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                    return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to HDF5 dataset")
    parser.add_argument("--play", action="store_true", help="Play back the demos")
    parser.add_argument("--fps", type=int, default=20, help="Playback speed (FPS)")
    parser.add_argument("--no-recon", action="store_true", help="Disable photorealistic reconstruction during playback")
    args = parser.parse_args()

    validate_hdf5(args.file)
    if args.play:
        play_demos(args.file, fps=args.fps, use_recon=not args.no_recon)
