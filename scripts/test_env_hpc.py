import torch
import mujoco
import os
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def test_gpu():
    print("\n--- GPU Check ---")
    available = torch.cuda.is_available()
    print(f"PyTorch CUDA Available: {available}")
    if available:
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
    else:
        print("WARNING: No GPU detected. This will slow down training significantly.")
    return available

def test_mujoco():
    print("\n--- MuJoCo Check ---")
    print(f"MuJoCo Version: {mujoco.__version__}")
    try:
        # Try a simple headless render test
        # We don't need a full env, just a model
        xml_path = os.path.join(project_root, "env_setup/grippers/models/fots_panda_gripper.xml")
        if os.path.exists(xml_path):
            model = mujoco.MjModel.from_xml_path(xml_path)
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            pixels = renderer.render()
            print(f"MuJoCo Render Success: {pixels.shape} image generated.")
        else:
            print(f"Skipping render test: {xml_path} not found.")
    except Exception as e:
        print(f"MuJoCo Render FAILED: {e}")
        return False
    return True

def test_imports():
    print("\n--- Imports Check ---")
    try:
        import robosuite
        import wandb
        print("robosuite and wandb imported successfully.")
    except ImportError as e:
        print(f"Import FAILED: {e}")
        return False
    return True

if __name__ == "__main__":
    s1 = test_gpu()
    s2 = test_mujoco()
    s3 = test_imports()
    
    if all([s1, s2, s3]):
        print("\n✅ Environment Smoke Test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Environment Smoke Test FAILED")
        sys.exit(1)
