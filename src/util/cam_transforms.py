import numpy as np
import torch
import pytorch3d.renderer.cameras as cameras

def get_patch_ndc_proj_matrix(patch_size, image_size, patch_center):
    """Get NDC proj matrix with patch center at patch_center in NDC space
    
    We represent the NDC to screen conversion as a 4x4 matrix K with projection matrix

    K = [
            [sx,   0,    0,  cx],
            [0,   sy,    0,  cy],
            [0,   0,    1,   0],
            [0,   0,    0,   1],
    ]
    
    patch size is the size of the patch in pixels # batch_size, 2
    image size is the size of the image in pixels # batch_size, 2
    
    patch_center is the center of the patch in image coordinates ie pixel coordinates # batch_size, 2
    
    K is the camera intrinsics matrix # batch_size, 4, 4
    
    """
    
    # when patch_ndc_proj_matrix is multiplied by 4d homogenous screen coords, they must be in the range [-1, 1]
    
    sx = 2 * patch_size[:, 1] / image_size[:, 1]
    sy = 2 * patch_size[:, 0] / image_size[:, 0]
    
    # offset from center in x with origin at top left, center coords H/2, W/2
    cx = (2 * patch_center[:, 0] / image_size[:, 1]) - 1 
    cy = (2 * patch_center[:, 1] / image_size[:, 0]) - 1
    
    batch_size = patch_size.shape[0]
    patch_ndc_proj_matrix = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    
    patch_ndc_proj_matrix[:, 0, 0] = sx
    patch_ndc_proj_matrix[:, 1, 1] = sy
    patch_ndc_proj_matrix[:, 0, 2] = cx
    patch_ndc_proj_matrix[:, 1, 2] = cy
    
    return patch_ndc_proj_matrix

def patch_ndc2_full_ndc(patch_size, image_size, patch_center):
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
        K: camera
        patch_size (tuple): The size of the patch, (patch_height, patch_width). shape: (batch_size, 2)
        image_size (tuple): The size of the full image, (full_height, full_width). shape: (batch_size, 2)
        patch_center (tuple): The center of the patch with respect to the full image center (full image center is (0, 0)). shape: (batch_size, 2)

    Returns:
        torch.Tensor: The NDC projection matrix with respect to full image. shape: (batch_size, 4, 4)

    """
    
    patch_ndc_proj_matrix = get_patch_ndc_proj_matrix(patch_size, image_size, patch_center)
    
    # scale the patch_ndc_proj_matrix to full image size such that when screen coords are multiplied by this matrix, they are in the range [-1, 1]
    full_ndc_proj_matrix = torch.eye(4).unsqueeze(0).repeat(patch_size.shape[0], 1, 1)
    
    H_full, W_full = image_size[:, 0], image_size[:, 1]
    H_patch, W_patch = patch_size[:, 0], patch_size[:, 1]
    
    cx_patch, cy_patch = patch_center[:, 0], patch_center[:, 1] # patch center in x and y in pixels, origin at top left
    sx_patch, sy_patch = patch_ndc_proj_matrix[:, 0, 0], patch_ndc_proj_matrix[:, 1, 1] # patch scale in x and y from NDC to screen patch

    sx_full = (W_patch / W_full) * sx_patch
    sy_full = (H_patch / H_full) * sy_patch
    
    cx_full = (cx_patch * sx_patch) - (W_full / 2)
    cy_full = (cy_patch * sy_patch) - (H_full / 2)
    
    full_ndc_proj_matrix[:, 0, 0] = sx_full
    full_ndc_proj_matrix[:, 1, 1] = sy_full
    full_ndc_proj_matrix[:, 0, 2] = cx_full
    full_ndc_proj_matrix[:, 1, 2] = cy_full
    
    return full_ndc_proj_matrix
 
def full_ndc2_patch_ndc(patch_ndc_proj_matrix, patch_size, image_size, patch_center):
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
    
    full_ndc_proj_matrix = patch_ndc2_full_ndc(patch_ndc_proj_matrix, patch_size, image_size, patch_center)
    patch_ndc_proj_matrix = torch.inverse(full_ndc_proj_matrix)
    return patch_ndc_proj_matrix
       
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
    
def full_screen2_camera_coords(pose_screen, K):
    """
    Convert screen coordinates to camera coordinates.
    
    Args:
        screen_coords (torch.Tensor): The se3 pose matrix in screen coords. shape: (batch_size, 4, 4)
        K (torch.Tensor): The intrinsic camera matrix. shape: b, 4, 4
        
    Returns:
        torch.Tensor: The camera coordinates. shape: (batch_size, 3)
    """
    K_inv = torch.inverse(K) # shape: (batch_size, 4, 4)
    
    screen2cam_transform = torch.matmul(K_inv, pose_screen.t()).t() # shape: (batch_size, 4, 4)
    
    camera_coords = screen2cam_transform[:, :3, 3] #  Extracting the resulting camera coordinates (translation part)
    
    return camera_coords
      
def camera_coords2_full_screen(camera_coords, K):
    """
    Convert camera coordinates to screen coordinates.
    
    Args:
        camera_coords (torch.Tensor): The camera coordinates. shape: (batch_size, 3)
        K (torch.Tensor): The intrinsic camera matrix. shape: b, 4, 4
        
    Returns:
        torch.Tensor: The transform in screen coordinates. shape: (batch_size, 4, 4)
    """
    
    camera2screen_transform = torch.eye(4).unsqueeze(0).repeat(camera_coords.shape[0], 1, 1)
    camera2screen_transform[:, :3, 3] = camera_coords
    
    camera2screen_transform = torch.matmul(K, camera2screen_transform.t()).t() # shape: (batch_size, 4, 4)
    
    return camera2screen_transform

def camera_coords_from_patch_NDC(patch_se3_pose, patch_size, image_size, patch_center,
                                 cameras, with_xyflip=False):
    
    patch_ndc2full_ndc = patch_ndc2_full_ndc(patch_size, image_size, patch_center)
    full_ndc2full_screen = full_ndc2full_screen(cameras, with_xyflip, image_size)
    
    patch_ndc2full_screen = torch.matmul(full_ndc2full_screen, patch_ndc2full_ndc)
    
    patch_screen2camera = torch.matmul(patch_se3_pose, patch_ndc2full_screen)
    
    camera_coords = full_screen2_camera_coords(patch_screen2camera, cameras.K)

    return camera_coords
