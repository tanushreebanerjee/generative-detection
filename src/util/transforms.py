import numpy as np
import warnings
from src.transforms.conversions import matrix_from_euler_batch, transform_from_batch
# implementing all functions as numpy functions to allow for batch processing

def dual_quaternion_from_transform_batch(transform):
    pass

def screw_parameters_from_dual_quaternion_batch(dq):
    pass

def screw_axis_from_screw_parameters_batch(q, s_axis, h):
    pass

def euler_translation_to_screw(euler_angles, translation):
    """
    Converts Euler angles and translation to screw axis representation. (in batches)
    
    Args: 
        euler_angles : array-like, shape (N, 3)
            Extracted rotation angles in radians about the axes i, j, k in this
            order. The first and last angle are normalized to [-pi, pi]. The middle
            angle is normalized to either [0, pi] (proper Euler angles) or
            [-pi/2, pi/2] (Cardan / Tait-Bryan angles).
        
        translation : array-like, shape (N, 3)
                Translation
    
    Returns:
        screw_axis : array, shape (N, 6)
            Screw axis described by 6 values
            (omega_1, omega_2, omega_3, v_1, v_2, v_3),
            where the first 3 components are related to rotation and the last 3
            components are related to translation.
    
    Raises:
        AssertionError: If the length of the screw axis tuple is not 6.
    """
    
    # assert shape of inputs 
    assert euler_angles.shape[1] == 3, f"Expected euler_angles to have shape (N, 3), but got {euler_angles.shape}"
    assert translation.shape[1] == 3, f"Expected translation to have shape (N, 3), but got {translation.shape}"
    
    # compute active rotation matrix from any Euler angles
    R = matrix_from_euler_batch(e=euler_angles, i=0, j=1, k=2, extrinsic=True)

    # Make transformation from rotation matrix and translation.
    transform = transform_from_batch(R, translation)
    
    # Compute dual quaternion from transformation matrix.
    dq = dual_quaternion_from_transform_batch(transform)

    # Compute screw parameters from dual quaternion.
    
    q, s_axis, h, _ = screw_parameters_from_dual_quaternion_batch(dq)
    
    screw_axis = screw_axis_from_screw_parameters_batch(q, s_axis, h)
    
    assert screw_axis.shape[1] == 6, f"Screw axis should be an array of 6 elements, but got {screw_axis} of length {screw_axis.shape[1]}"
    return screw_axis