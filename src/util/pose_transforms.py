from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, se3_log_map, se3_exp_map
import torch
import logging

def euler_angles_translation2transformation_matrix(euler_angle, translation, convention):
    """Convert euler angles and translation to transformation matrix. They are batched tensors."""
    # check if batch dimension is present, if not add it
    if len(euler_angle.shape) == 1:
        euler_angle = euler_angle.unsqueeze(0)
        translation = translation.unsqueeze(0)
        
    # make sure euler_angles.dim() == 0 and euler_angles.shape[-1] != 3:
    euler_angle = euler_angle.reshape(-1, 3)
    translation = translation.reshape(-1, 3)
    
    batch_size = euler_angle.shape[0] 
    R = euler_angles_to_matrix(euler_angle, convention) # R shape: torch.Size([1, 3, 3])
    t = translation # t shape: torch.Size([1, 3])
    transformation = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1) # transformation shape: torch.Size([1, 4, 4])
    transformation[:, :3, :3] = R
    transformation[:, :3, 3] = t
    return transformation

def euler_angles_translation2se3_log_map(euler_angle, translation, convention):
    """
    Convert euler angles and translation to a batch of 6-dimensional SE(3) logarithms of the SE(3) matrices.
    
    """    
    transformation = euler_angles_translation2transformation_matrix(euler_angle, translation, convention)    
    transformation_t = transformation.permute(0, 2, 1)    
    se3_log = se3_log_map(transformation_t)   
    se3_log = torch.tensor(se3_log, dtype=torch.float32)    
    return se3_log

def transformation_matrix2euler_angles_translation(transformation, convention):
    """Convert transformation matrix to euler angles and translation."""

    R = transformation[:, :3, :3].permute(0, 2, 1)
    t = transformation.permute(0, 2, 1)[:, :3, 3]

    euler_angle = matrix_to_euler_angles(R, convention)
    translation = t
    return euler_angle, translation

def se3_log_map2euler_angles_translation(se3_log, convention, angle_max=-torch.pi, angle_min=torch.pi):
    """
    Convert a batch of 6-dimensional SE(3) logarithms of the SE(3) matrices to euler angles and translation.
    default: # make all angles be in the range of [-pi, pi]
    """
    # put zeroes in the first three elements
    transformation = se3_exp_map(se3_log)
    
    euler_angle, t = transformation_matrix2euler_angles_translation(transformation, convention)
    
    # make all angles be in the range [angle_min, angle_max]
    euler_angle = torch.where(euler_angle > angle_max, euler_angle - 2 * torch.pi, euler_angle)
    euler_angle = torch.where(euler_angle < angle_min, euler_angle + 2 * torch.pi, euler_angle)
    translation = t
    return euler_angle, translation