# src/util/misc.py
import os
import sys
import logging
import json
import pytransform3d

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

def euler_translation_to_screw(euler_angles, translation):
    """Convert Euler angles and translation to screw."""
    
    # compute active rotation matrix from any Euler angles
    R = pytransform3d.rotations.matrix_from_euler(euler_angles, axes="sxyz")

    # Make transformation from rotation matrix and translation.
    transform = pytransform3d.transformations.transform_from(R, translation)
    
    # Compute dual quaternion from transformation matrix.
    dq = pytransform3d.transformations.dual_quaternion_from_transform(transform)

    # Compute screw parameters from dual quaternion.
    screw_parameters = pytransform3d.transformations.screw_parameters_from_dual_quaternion(dq)
    
    assert screw_parameters.shape == (6,), "Screw parameters should have shape (6,)."
    return screw_parameters