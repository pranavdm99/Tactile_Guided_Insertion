import cv2
import torch
import numpy as np
import copy
seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preproc_mlp(normal) -> torch.Tensor:
    """ Preprocess image for input to model.

    Args: normal: Normal map array
    Return: tensor of shape (R*C,5) where R=320 and C=240 for DIGIT images
    5-columns are: X,Y,Nx,Ny,Nz

    """
    # Ensure we always have a full (320x240) coordinate grid regardless of NaNs
    R, C = normal.shape[:2]
    yy, xx = np.meshgrid(np.arange(R), np.arange(C), indexing='ij')
    xy_coords = np.stack([xx.flatten(), yy.flatten()], axis=1)
    
    # Flatten normal map and sanitize NaNs/Infs
    nxyz = np.reshape(normal, (R * C, 3))
    nxyz = np.nan_to_num(nxyz, nan=0.0, posinf=1.0, neginf=-1.0)
    
    value_base = np.hstack([xy_coords.astype(np.float32), nxyz.astype(np.float32)])
    
    # Normalize X, Y coordinates to [0, 1] relative to sensor resolution
    value_base[:, 0] /= float(C)
    value_base[:, 1] /= float(R)
    
    test_tensor = torch.tensor(value_base, dtype=torch.float32).to(device)
    return test_tensor


def post_proc_mlp(model_output: torch.Tensor):
    """ Postprocess model output to get normal map.

    Args: model_output: torch.Tensor of shape (1,3)
    Return: two torch.Tensor of shape (1,3)

    """
    test_np = model_output.reshape(320, 240, 3)
    normal = copy.deepcopy(test_np)  # surface normal image
    test_np = torch.tensor(test_np,
                           dtype=torch.float32)  # convert to torch tensor for later processing in gradient computation
    test_np = test_np.permute(2, 0, 1)  # swap axes to (3,320,240)
    test_np = test_np # convert to uint8 for visualization
    return test_np, normal


