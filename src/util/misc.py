# src/util/misc.py
import os
import sys
import logging
import json
import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, se3_log_map, se3_exp_map

def log_opts(opts):
    """Log the options."""
    logging.info("Options: %s", json.dumps(vars(opts), indent=2))

def set_submodule_paths(submodule_dir):
    """Set the paths for the submodules."""
    for submodule in os.listdir(submodule_dir):
        submodule_path = os.path.join(submodule_dir, submodule)
        if os.path.isdir(submodule_path):
            sys.path.append(submodule_path)
     
def set_cache_directories(opts):
    """Set environment variables for cache directories."""
    os.environ["TRANSFORMERS_CACHE"] = opts.transformers_cache
    os.environ["TORCH_HOME"] = opts.torch_home
  
def euler_angles_translation2transformation_matrix(euler_angle, translation, convention):
    """Convert euler angles and translation to transformation matrix. They are batched tensors."""
    batch_size = euler_angle.shape[0]

    R = euler_angles_to_matrix(euler_angle, convention) # shape: (batch_size, 3, 3)
    t = translation
    
    transformation = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    transformation[:, :3, :3] = R
    transformation[:, :3, 3] = t
    
    return transformation

def euler_angles_translation2se3_log_map(translation, euler_angle, convention):
    """
    Convert euler angles and translation to  to a batch of 6-dimensional SE(3) logarithms of the SE(3) matrices.
    
    """
    transformation = euler_angles_translation2transformation_matrix(euler_angle, translation, convention)    
    
    transformation_t = transformation.permute(0, 2, 1)
    se3_log = se3_log_map(transformation_t)
    
    return se3_log

def transformation_matrix2euler_angles_translation(transformation, convention):
    """Convert transformation matrix to euler angles and translation."""

    R = transformation[:, :3, :3].permute(0, 2, 1)
    t = transformation.permute(0, 2, 1)[:, :3, 3]

    euler_angle = matrix_to_euler_angles(R, convention)   # 
    return euler_angle, t

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
    
    return euler_angle, t
