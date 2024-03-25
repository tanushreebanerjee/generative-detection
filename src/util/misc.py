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
  
def euler_angles_translation2transformation_matrix(translation, euler_angle):
    """Convert euler angles and translation to transformation matrix."""
    
    R = euler_angles_to_matrix(euler_angle)
    t = translation
    
    transformation = torch.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    
    return transformation

def euler_angles_translation2se3_log_map(translation, euler_angle):
    """
    Convert euler angles and translation to  to a batch of 6-dimensional SE(3) logarithms of the SE(3) matrices.
    
    """
    transformation = euler_angles_translation2transformation_matrix(translation, euler_angle)
    se3_log = se3_log_map(transformation)
    return se3_log

def transformation_matrix2euler_angles_translation(transformation):
    """Convert transformation matrix to euler angles and translation."""
    
    R = transformation[:3, :3]
    t = transformation[:3, 3]
    
    euler_angle = matrix_to_euler_angles(R)    
    return t, euler_angle

def se3_log_map2euler_angles_translation(se3_log):
    """
    Convert a batch of 6-dimensional SE(3) logarithms of the SE(3) matrices to euler angles and translation.
    
    """
    transformation = se3_exp_map(se3_log)
    t, euler_angle = transformation_matrix2euler_angles_translation(transformation)
    return t, euler_angle
