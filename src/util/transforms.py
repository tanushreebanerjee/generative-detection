from pytransform3d.rotations import _conversions as conversions
from pytransform3d.transformations import transform_from, dual_quaternion_from_transform, screw_parameters_from_dual_quaternion

def euler_translation_to_screw(euler_angles, translation):
    """Convert Euler angles and translation to screw."""
    
    # compute active rotation matrix from any Euler angles
    R = conversions.matrix_from_euler(euler_angles, 0, 1, 2, True)

    # Make transformation from rotation matrix and translation.
    transform = transform_from(R, translation)
    
    # Compute dual quaternion from transformation matrix.
    dq = dual_quaternion_from_transform(transform)

    # Compute screw parameters from dual quaternion.
    screw_parameters = screw_parameters_from_dual_quaternion(dq)
    
    assert screw_parameters.shape == (6,), "Screw parameters should have shape (6,)."
    return screw_parameters