from pytransform3d.rotations import _conversions as conversions
from pytransform3d.transformations import transform_from, dual_quaternion_from_transform, screw_parameters_from_dual_quaternion, screw_axis_from_screw_parameters

def euler_translation_to_screw(euler_angles, translation):
    """
    Converts Euler angles and translation to screw axis representation.
    
    Args:
        euler_angles (tuple): Tuple of Euler angles (in radians) representing rotation.
        translation (tuple): Tuple of translation values (x, y, z).
    
    Returns:
        screw_axis : array, shape (6,)
            Screw axis described by 6 values
            (omega_1, omega_2, omega_3, v_1, v_2, v_3),
            where the first 3 components are related to rotation and the last 3
            components are related to translation.
    
    Raises:
        AssertionError: If the length of the screw axis tuple is not 6.
    """
    
    # compute active rotation matrix from any Euler angles
    R = conversions.matrix_from_euler(euler_angles, 0, 1, 2, True)

    # Make transformation from rotation matrix and translation.
    transform = transform_from(R, translation)
    
    # Compute dual quaternion from transformation matrix.
    dq = dual_quaternion_from_transform(transform)

    # Compute screw parameters from dual quaternion.
    
    q, s_axis, h, _ = screw_parameters_from_dual_quaternion(dq)
    
    screw_axis = screw_axis_from_screw_parameters(q, s_axis, h)
    
    # check if its an array of 6 elements
    assert len(screw_axis) == 6, f"Screw axis should be an array of 6 elements, but got {screw_axis}"
    return screw_axis