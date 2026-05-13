"""
Strategic checkpoint sweep informed by Run 2 training loss analysis.

Training log (train_19219503.out) shows:
  - Val loss explodes after epoch ~100 — useless for checkpoint selection
  - Train loss improves steadily, with a noticeable jump after LR decay at epoch 2000
  - Epochs 1000-1050 confirmed 0/5 success in initial sweep
  - Best known checkpoint: epoch 2800 with ~20% success

Strategy:
  - Spot-check 4 early epochs (1000, 1250, 1500, 1750) to confirm the trend
  - Dense sweep of post-LR-decay zone (2000-3000 every 50 epochs, 21 checkpoints)
  Total: ~25 checkpoints instead of 41 — saves ~40% runtime

Usage:
    conda activate robomimic_venv
    python sweep_rollouts.py
"""

import os
import re
import json
import subprocess
import sys

PYTHON = sys.executable
SCRIPT = "robomimic/scripts/rollout_fots.py"
RUNS_DIR = "runs"
SWEEP_ROLLOUTS = 5
FINAL_ROLLOUTS = 10
HORIZON = 800
SEED = 42

# Strategic checkpoint selection based on training loss analysis
def get_epoch_checkpoints():
    pattern = re.compile(r"^model_epoch_(\d+)\.pth$")
    available = {}
    for f in os.listdir(RUNS_DIR):
        m = pattern.match(f)
        if m:
            epoch = int(m.group(1))
            available[epoch] = os.path.join(RUNS_DIR, f)

    # Spot-checks in pre-LR-decay zone (train loss ~-43 to -47, expected 0 success)
    early_spot = [1000, 1250, 1500, 1750]
    # Dense sweep of post-LR-decay zone (train loss -47 to -49.4, best performance expected)
    late_dense = list(range(2000, 3001, 50))

    wanted = sorted(set(early_spot + late_dense))
    checkpoints = [(ep, available[ep]) for ep in wanted if ep in available]
    return checkpoints


def run_rollout(agent, n_rollouts, horizon, seed, video_path=None):
    cmd = [
        PYTHON, SCRIPT,
        "--agent", agent,
        "--n_rollouts", str(n_rollouts),
        "--horizon", str(horizon),
        "--seed", str(seed),
    ]
    if video_path:
        cmd += ["--video_path", video_path]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse average stats from output
    stats = {}
    output = result.stdout + result.stderr
    in_avg = False
    for line in output.splitlines():
        if "Average Rollout Stats" in line:
            in_avg = True
        if in_avg and "Num_Success" in line:
            stats["Num_Success"] = int(re.search(r"(\d+)", line).group(1))
        if in_avg and '"Success"' in line:
            stats["Success"] = float(re.search(r"([\d.]+)", line).group(1))
        if in_avg and '"Return"' in line:
            stats["Return"] = float(re.search(r"([\d.]+)", line).group(1))

    if result.returncode != 0 and not stats:
        print(f"  ERROR:\n{result.stderr[-500:]}")

    return stats


def main():
    checkpoints = get_epoch_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints to sweep")
    print(f"  Early spot-checks: {[e for e,_ in checkpoints if e < 2000]}")
    print(f"  Post-LR-decay (2000-3000): {len([e for e,_ in checkpoints if e >= 2000])} checkpoints\n")

    results = []

    for epoch, path in checkpoints:
        print(f"Epoch {epoch:4d} | ", end="", flush=True)
        stats = run_rollout(path, SWEEP_ROLLOUTS, HORIZON, SEED)
        num_success = stats.get("Num_Success", 0)
        avg_return = stats.get("Return", 0.0)
        print(f"Success {num_success}/{SWEEP_ROLLOUTS} | Return {avg_return:.1f}")
        results.append({
            "epoch": epoch,
            "path": path,
            "num_success": num_success,
            "avg_return": avg_return,
        })

    # Save sweep results
    results_path = os.path.join(RUNS_DIR, "sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep results saved to {results_path}")

    # Print top 5 from coarse sweep
    top5 = sorted(results, key=lambda x: (x["num_success"], x["avg_return"]), reverse=True)[:5]
    print(f"\n{'='*50}")
    print("Top 5 from coarse sweep:")
    for r in top5:
        print(f"  Epoch {r['epoch']:4d} — {r['num_success']}/{SWEEP_ROLLOUTS} success, return {r['avg_return']:.1f}")
    print(f"{'='*50}")

    # Confirmation pass: re-test top 3 candidates with more rollouts to reduce ranking noise.
    # With 5-rollout sweep a 20% success checkpoint has 33% chance of showing 0/5,
    # so we confirm the top 3 before picking the winner.
    CONFIRM_ROLLOUTS = 20
    top3 = top5[:3]
    print(f"\nConfirmation pass: {CONFIRM_ROLLOUTS} rollouts each on top-3 candidates...")
    confirmed = []
    for r in top3:
        print(f"Epoch {r['epoch']:4d} | ", end="", flush=True)
        stats = run_rollout(r["path"], CONFIRM_ROLLOUTS, HORIZON, SEED + 2)
        num_success = stats.get("Num_Success", 0)
        avg_return = stats.get("Return", 0.0)
        print(f"Success {num_success}/{CONFIRM_ROLLOUTS} | Return {avg_return:.1f}")
        confirmed.append({**r, "confirm_num_success": num_success, "confirm_avg_return": avg_return})

    # Save updated results with confirmation stats
    for r in confirmed:
        for orig in results:
            if orig["epoch"] == r["epoch"]:
                orig.update({"confirm_num_success": r["confirm_num_success"],
                             "confirm_avg_return": r["confirm_avg_return"]})
    results_path = os.path.join(RUNS_DIR, "sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    best = max(confirmed, key=lambda x: (x["confirm_num_success"], x["confirm_avg_return"]))
    print(f"\n{'='*50}")
    print(f"BEST CHECKPOINT: epoch {best['epoch']}")
    print(f"  Coarse:  {best['num_success']}/{SWEEP_ROLLOUTS}")
    print(f"  Confirm: {best['confirm_num_success']}/{CONFIRM_ROLLOUTS}")
    print(f"  Return:  {best['confirm_avg_return']:.1f}")
    print(f"{'='*50}")

    # Final run with video on confirmed best
    video_path = os.path.join(RUNS_DIR, f"best_epoch{best['epoch']}_rollout.mp4")
    print(f"\nRunning final {FINAL_ROLLOUTS}-rollout eval on epoch {best['epoch']} with video...")
    final_stats = run_rollout(best["path"], FINAL_ROLLOUTS, HORIZON, SEED + 3, video_path=video_path)
    print(f"Final stats: {final_stats}")
    print(f"Video saved: {video_path}")


if __name__ == "__main__":
    main()
