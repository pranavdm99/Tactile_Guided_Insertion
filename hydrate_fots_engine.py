import os
import shutil
import logging
import subprocess
import argparse

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def run_command(command, cwd=None):
    try:
        logging.info(f"Executing: {' '.join(command)}")
        subprocess.check_call(command, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {e}")
        return False

def hydrate(repo_url="https://github.com/pranavdm99/FOTS.git", branch="FOTS-mujoco"):
    """
    Extracts relevant FOTS simulation components from FOTS_repo into the decoupled fots_sim engine.
    If FOTS_repo is missing, it automatically clones from the specified URL and branch.
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    src_repo = os.path.join(root_dir, "FOTS_repo")
    dest_engine = os.path.join(root_dir, "fots_sim")

    # 1. Automatic Cloning if missing
    if not os.path.exists(src_repo):
        logging.info(f"Source repository not found. Cloning from {repo_url}...")
        if not run_command(["git", "clone", repo_url, "FOTS_repo"], cwd=root_dir):
            return False
        
        if branch != "main":
            logging.info(f"Checking out branch: {branch}...")
            if not run_command(["git", "checkout", branch], cwd=src_repo):
                return False

    # 2. Define the mapping (Source Relative to FOTS_repo -> Dest Relative to fots_sim)
    mapping = {
        "src/train/mlp_model.py": "mlp_model.py",
        "models/mlp_n2c_r.pth": "models/mlp_n2c_r.pth",
        "planar_shadow.py": "planar_shadow.py",
        "utils/mlp_render.py": "utils/mlp_render.py",
        "utils/prepost_mlp.py": "utils/prepost_mlp.py",
        "utils/utils_data/ini_bg_mlp.npy": "utils/ini_bg_mlp.npy",
        "utils/utils_data/ini_depth_extent.npy": "utils/ini_depth_extent.npy",
        "assets/gel/digit_bg.npy": "assets/digit_bg.npy",
    }

    logging.info("Starting FOTS Engine Hydration...")

    success_count = 0
    for src_rel, dest_rel in mapping.items():
        src_path = os.path.join(src_repo, src_rel)
        dest_path = os.path.join(dest_engine, dest_rel)

        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)
            logging.info(f"✓ Extracted: {src_rel} -> {dest_rel}")
            success_count += 1
        else:
            logging.warning(f"✘ Missing from source: {src_rel}")

    logging.info(f"Hydration Complete. {success_count}/{len(mapping)} files synchronized.")
    
    # 3. Optional Clean-up: Remove the source repo to keep the project lean
    if os.path.exists(src_repo):
        logging.info("Cleaning up legacy FOTS_repo...")
        shutil.rmtree(src_repo)
        logging.info("✓ Legacy repository removed.")

    logging.info("The 'fots_sim' engine is now self-contained.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOTS Engine Hydration Tool")
    parser.add_argument("--url", type=str, default="https://github.com/pranavdm99/FOTS.git", help="Remote repository URL")
    parser.add_argument("--branch", type=str, default="FOTS-mujoco", help="Git branch to checkout")
    args = parser.parse_args()
    
    hydrate(repo_url=args.url, branch=args.branch)
