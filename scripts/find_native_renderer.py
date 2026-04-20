import robosuite as suite
import mujoco
import numpy as np

# Use a standard robosuite env to find the renderer handle
env = suite.make(
    env_name="NutAssembly",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
)

print(f"Sim type: {type(env.sim)}")
# Check properties of MjSim or similar
sim = env.sim
for attr in dir(sim):
    if "render" in attr.lower():
        val = getattr(sim, attr)
        print(f"Attr: {attr}, Type: {type(val)}")

# Try to render a camera using the sim's own method
try:
    # Use standard robosuite rendering
    img = sim.render(width=320, height=240, camera_name="agentview")
    print(f"[SUCCESS] Native Sim Render worked. Shape: {img.shape}")
except Exception as e:
    print(f"[FAIL] Native Sim Render failed: {e}")

env.close()
