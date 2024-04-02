import numpy as np
import torch
import pytorch3d.renderer.cameras as cameras

def patch_ndc2_full_ndc(patch_ndc_proj_matrix, patch_size, image_size, patch_center):
    """
    Given NDC projection matrix (+X left, +Y up) with respect to patch in an image, image size, patch size, and patch center,
    returns NDC projection matrix (+X left, +Y up) with respect to full image.
    
    We represent the NDC to screen conversion as a 4x4 matrix K with projection matrix

    K = [
            [sx,   0,    0,  cx],
            [0,   sy,    0,  cy],
            [0,   0,    1,   0],
            [0,   0,    0,   1],
    ]

    K_scaled:
    sx = full_width / patch_width
    sy = full_height / patch_height
    
    cx = (patch_center_x * sx) - (full_width / 2)
    cy = (patch_center_y * sy) - (full_height / 2)
    
    
    Args:
        patch_ndc_proj_matrix (torch.Tensor): The NDC projection matrix with respect to patch, of shape (batch_size, 4, 4)
        patch_size (tuple): The size of the patch, (patch_height, patch_width). shape: (batch_size, 2)
        image_size (tuple): The size of the full image, (full_height, full_width). shape: (batch_size, 2)
        patch_center (tuple): The center of the patch with respect to the full image center (full image center is (0, 0)). shape: (batch_size, 2)

    Returns:
        torch.Tensor: The NDC projection matrix with respect to full image. shape: (batch_size, 4, 4)

    """
    
    # if no batch dim then add it
    if len(patch_ndc_proj_matrix.shape) == 2:
        patch_ndc_proj_matrix = patch_ndc_proj_matrix.unsqueeze(0) # shape: (1, 4, 4)
    
    if len(patch_size.shape) == 1:
        patch_size = patch_size.unsqueeze(0)
        # repeat patch size if one value is given
        patch_size = patch_size.repeat(patch_ndc_proj_matrix.shape[0], 1) # shape: (batch_size, 2)
    
    if len(image_size.shape) == 1:
        image_size = image_size.unsqueeze(0)
        patch_size = patch_size.repeat(patch_ndc_proj_matrix.shape[0], 1) # shape: (batch_size, 2)
    
    if len(patch_center.shape) == 1:
        patch_center = patch_center.unsqueeze(0)
        patch_center = patch_center.repeat(patch_ndc_proj_matrix.shape[0], 1)
        
    H_patch, W_patch = patch_size[:, 0], patch_size[:, 1]
    H_full, W_full = image_size[:, 0], image_size[:, 1]
        
    full_ndc_proj_matrix = torch.eye(4).unsqueeze(0).repeat(patch_ndc_proj_matrix.shape[0], 1, 1)
    
    # scaling
    sx_scale = W_full / W_patch 
    sy_scale = H_full / H_patch
    
    full_ndc_proj_matrix[:, 0, 0] = sx_scale
    full_ndc_proj_matrix[:, 1, 1] = sy_scale
    
    # translation
    full_ndc_proj_matrix[:, 0, 3] = (patch_center[:, 0] * sx_scale) - (W_full / 2)
    full_ndc_proj_matrix[:, 1, 3] = (patch_center[:, 1] * sy_scale) - (H_full / 2)
    
    return full_ndc_proj_matrix
    
    
def full_ndc2_full_ndc(patch_center, patch_height, patch_width, full_height, full_width):
    """
    Given NDC projection matrix (+X left, +Y up) with respect to the full image, of image size,
    returns NDC projection matrix (+X left, +Y up) with respect to patch with patch size, and patch center given
    
    We represent the NDC to screen conversion as a 4x4 matrix K with projection matrix

    K = [
            [sx,   0,    0,  cx],
            [0,   sy,    0,  cy],
            [0,   0,    1,   0],
            [0,   0,    0,   1],
    ]

    
    Args:
        full_ndc_proj_matrix (torch.Tensor): The NDC projection matrix with respect to full image.
        patch_size (tuple): The size of the patch, (patch_height, patch_width).
        image_size (tuple): The size of the full image, (full_height, full_width).
        patch_center (tuple): The center of the patch with respect to the full image center (full image center is (0, 0)).

    Returns:
        torch.Tensor: The NDC projection matrix with respect to patch.
    """
    return torch.inverse(patch_ndc2_full_ndc(patch_center, patch_height, patch_width, full_height, full_width))
    
    

def full_ndc2full_screen(cameras,
                         with_xyflip=False,
                            image_size=None,):
    
    ndc2screen = cameras.get_ndc_to_screen_transform(cameras, with_xyflip, image_size)
    return torch.tensor(ndc2screen.get_matrix().detach().cpu().numpy(), dtype=torch.float32)
    

def full_screen2_full_ndc(cameras,
                          with_xyflip=False,
                          image_size=None,):
    screen2ndc = cameras.get_screen_to_ndc_transform(cameras, with_xyflip, image_size)
    return torch.tensor(screen2ndc.get_matrix().detach().cpu().numpy(), dtype=torch.float32)
    
def full_screen2_camera_coords(screen_coords, K):
    """
    Convert screen coordinates to camera coordinates.
    
    Args:
        screen_coords (torch.Tensor): The screen coordinates. shape: (batch_size, 2)
        K (torch.Tensor): The intrinsic camera matrix. shape: b, 4, 4
        
    Returns:
        torch.Tensor: The camera coordinates. shape: (batch_size, 3)
    """
    K_inv = torch.inverse(K)
    
    # add 1 to the z coordinate
    screen_coords_3d = torch.cat([screen_coords, torch.ones_like(screen_coords[:, 0:1])], dim=1)
    
    # make the screen coordinates homogenous by adding 1
    screen_coords_homogeneous = torch.cat([screen_coords, torch.ones_like(screen_coords[:, 0:1])], dim=1)
    
    camera_coords_homogeneous = torch.matmul(K_inv, screen_coords_homogeneous.t()).t() # shape: (batch_size, 4)
    
    x_camera, y_camera, z_camera = camera_coords_homogeneous[:, 0], camera_coords_homogeneous[:, 1], camera_coords_homogeneous[:, 2]
    
    return torch.stack([x_camera, y_camera, z_camera], dim=1) # shape: (batch_size, 3)
      
def camera_coords2_full_screen(camera_coords, K):
    """
    Convert camera coordinates to screen coordinates.
    
    Args:
        camera_coords (torch.Tensor): The camera coordinates.
        K (torch.Tensor): The intrinsic camera matrix.
        
    Returns:
        torch.Tensor: The screen coordinates.
    """
    x_camera, y_camera, z_camera = camera_coords[:, 0], camera_coords[:, 1], camera_coords[:, 2]
    screen_coords = torch.stack([x_camera, y_camera, z_camera], dim=1)
    
    screen_coords = torch.cat([screen_coords, torch.ones_like(screen_coords[:, 0:1])], dim=1)
    
    screen_coords = torch.matmul(K, screen_coords.t()).t()
    
    x_screen = screen_coords[:, 0] / screen_coords[:, 2]
    y_screen = screen_coords[:, 1] / screen_coords[:, 2]
    
    return torch.stack([x_screen, y_screen], dim=1)
    

def patch_ndc2_camera_coords(patch_center, patch_height, patch_width, K, full_height, full_width):
    """
    Converts patch coordinates in normalized device coordinates (NDC) to camera coordinates.
    
    Args:
        patch_center (tuple): The center coordinates of the patch in NDC.
        patch_height (float): The height of the patch in NDC.
        patch_width (float): The width of the patch in NDC.
        K (torch.Tensor): The intrinsic camera matrix.
        full_height (float): The height of the full image in NDC.
        full_width (float): The width of the full image in NDC.
    
    Returns:
        torch.Tensor: The camera coordinates.
    """
    full_ndc_proj_matrix = patch_ndc2_full_ndc(patch_center, patch_height, patch_width, full_height, full_width)
    full_ndc2full_screen_proj_matrix = full_ndc2full_screen(full_ndc_proj_matrix)
    full_screen2_camera_coords_proj_matrix = full_screen2_camera_coords(full_ndc2full_screen_proj_matrix, K)
    return full_screen2_camera_coords_proj_matrix

def camera_coords2_patch_ndc(camera_coords, patch_center, patch_height, patch_width, K, full_height, full_width):
    """
    Converts camera coordinates to patch coordinates in normalized device coordinates (NDC).
    
    Args:
        camera_coords (torch.Tensor): The camera coordinates.
        patch_center (tuple): The center coordinates of the patch in NDC.
        patch_height (float): The height of the patch in NDC.
        patch_width (float): The width of the patch in NDC.
        K (torch.Tensor): The intrinsic camera matrix.
        full_height (float): The height of the full image in NDC.
        full_width (float): The width of the full image in NDC.
    
    Returns:
        torch.Tensor: The patch coordinates in NDC.
    """
    full_ndc = full_screen2_full_ndc(patch_center, patch_height, patch_width, full_height, full_width)
    full_ndc2_camera_coords = full_screen2_camera_coords(full_ndc, K)
    return full_ndc2_camera_coords