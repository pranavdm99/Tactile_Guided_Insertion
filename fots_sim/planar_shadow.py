import cv2
import numpy as np

def shadow_matrix(S, light_type):
    if light_type == "spot":
        m = np.mat([[S[2], 0, -S[0], 0],
                    [0, S[2], -S[1], 0],
                    [0, 0, 0 ,0],
                    [0, 0, -1, S[2]]])
    else:
        m = np.mat([[1, 0, -S[0]/S[2], 0],
                    [0, 1, -S[1]/S[2], 0],
                    [0, 0, 0, 0]])
    return m

def planar_shadow(light, depth, light_type):
    """
    Resolution-agnostic planar shadow generation.
    Detects H, W from input depth map shape.
    """
    H, W = depth.shape
    m = shadow_matrix(light, light_type)
    
    # Threshold for contact detection to cast shadow
    # In FOTS scaled units, 5 is a typical threshold for small deformations
    idx = np.nonzero(depth > 5)

    if len(idx[0]) == 0:
        return np.ones((H, W))

    P = np.mat([idx[0], idx[1], depth[idx], np.ones_like(idx[0])])
    Q = np.dot(m, P)
    if light_type == "spot":
        # Add epsilon to prevent division by zero
        Q = Q / (Q[3] + 1e-6)
    
    # Sanitize NaNs and Infs before the cast to uint16
    Q = np.nan_to_num(Q, nan=0.0, posinf=65535, neginf=0)
    shadow = Q[:2].astype(np.uint16)
    
    # limit x,y to the actual dimensions of the input depth map
    shadow_x = np.asarray(shadow[0])
    shadow_y = np.asarray(shadow[1])
    shadow_x[shadow_x < 0] = 0
    shadow_x[shadow_x > H - 1] = H - 1
    shadow_y[shadow_y < 0] = 0
    shadow_y[shadow_y > W - 1] = W - 1
    
    # generate shadow mask
    mask_img = np.zeros((H, W))
    mask_img[shadow_x, shadow_y] = 1.0
    
    # Dilate the shadow for a softer look
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask_img, kernel, iterations=2)

    return 1.0 - mask

if __name__ == "__main__":
    # Test script for manual verification
    light = [80, 0, 100.0]
    # Create a dummy contact
    height_map = np.zeros((48, 64))
    height_map[20:25, 30:35] = 20
    
    mask = planar_shadow(light, height_map, "spot")
    print(f"Test shadow mask shape: {mask.shape}")
    if mask.shape == (48, 64):
        print("SUCCESS: Standalone resolution-agnostic test passed.")