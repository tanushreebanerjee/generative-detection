import numpy as np
from src.transforms.utils import batch_check_transform, batch_quaternion_from_matrix, \
    batch_concatenate_quaternions, batch_check_dual_quaternion, batch_norm_axis_angle, \
        batch_check_quaternion, batch_check_screw_parameters

# Source:
# Code adapted from https://github.com/dfki-ric/pytransform3d

ROT_MATRIX_SHAPE = (3, 3)

def check_axis_index(name, i):
    """Checks axis index.

    Parameters
    ----------
    name : str
        Name of the axis. Required for the error message.

    i : int from [0, 1, 2]
        Index of the axis (0: x, 1: y, 2: z)

    Raises
    ------
    ValueError
        If basis is invalid
    """
    if i not in [0, 1, 2]:
        raise ValueError("Axis index %s (%d) must be in [0, 1, 2]" % (name, i))

def batch_active_matrix_from_angle(basis, angle):
    """
    Compute active rotation matrix from rotation about basis vector.

    With the angle α and s = sin(α), c=cos(α), we construct rotation matrices about the basis vectors as follows:
    
    .. math::

        \boldsymbol{R}_x(\alpha) =
        \left(
        \begin{array}{ccc}
        1 & 0 & 0\\
        0 & c & -s\\
        0 & s & c
        \end{array}
        \right)

    .. math::

        \boldsymbol{R}_y(\alpha) =
        \left(
        \begin{array}{ccc}
        c & 0 & s\\
        0 & 1 & 0\\
        -s & 0 & c
        \end{array}
        \right)

    .. math::

        \boldsymbol{R}_z(\alpha) =
        \left(
        \begin{array}{ccc}
        c & -s & 0\\
        s & c & 0\\
        0 & 0 & 1
        \end{array}
        \right)

    Parameters
    ----------
    basis : int from [0, 1, 2]
        The rotation axis (0: x, 1: y, 2: z)

    angle : array-like, shape (batch_size, 1)
        Batched rotation angles

    Returns
    -------
    R : array, shape (batch_size, *ROT_MATRIX_SHAPE)
        Batched rotation matrices
    """
    c = np.cos(angle) # shape (batch_size, 1)
    s = np.sin(angle) # shape (batch_size, 1)
    R = np.zeros((len(angle), *ROT_MATRIX_SHAPE)) # shape (batch_size, 3, 3)
    
    if basis == 0:
        R[:, 0, 0] = 1.0
        R[:, 1, 1] = c
        R[:, 1, 2] = -s
        R[:, 2, 1] = s
        R[:, 2, 2] = c
    elif basis == 1:
        R[:, 0, 0] = c
        R[:, 0, 2] = s
        R[:, 1, 1] = 1.0
        R[:, 2, 0] = -s
        R[:, 2, 2] = c
    elif basis == 2:
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        R[:, 2, 2] = 1.0
    else:
        raise ValueError("Basis must be in [0, 1, 2]")
    
    return R
    
def batch_matrix_from_euler(e, i, j, k, extrinsic):
    """
    General method to compute active rotation matrix from any Euler angles.

    Parameters
    ----------
    e : array-like, shape (n, 3)
        Batched rotation angles in radians about the axes i, j, k in this
        order. The first and last angles are normalized to [-pi, pi]. The middle
        angle is normalized to either [0, pi] (proper Euler angles) or
        [-pi/2, pi/2] (Cardan / Tait-Bryan angles).

    i : int from [0, 1, 2]
        The first rotation axis (0: x, 1: y, 2: z)

    j : int from [0, 1, 2]
        The second rotation axis (0: x, 1: y, 2: z)

    k : int from [0, 1, 2]
        The third rotation axis (0: x, 1: y, 2: z)

    extrinsic : bool
        Do we use extrinsic transformations? Intrinsic otherwise.

    Returns
    -------
    R : array, shape (n, *ROT_MATRIX_SHAPE)
        Batched active rotation matrices
        
    Raises
    ------
    ValueError
        If basis is invalid
    """
    check_axis_index("i", i)
    check_axis_index("j", j)
    check_axis_index("k", k)
    
    try:
        alpha, beta, gamma = np.split(e, 3, axis=1)
    except ValueError:
        raise ValueError("Euler angles must have shape (n, 3)")

    if not extrinsic:
        i, k = k, i
        alpha, gamma = gamma, alpha

    R = batch_active_matrix_from_angle(k, gamma).dot(
        batch_active_matrix_from_angle(j, beta)).dot(
        batch_active_matrix_from_angle(i, alpha))

    return R

def batch_translate_transform(A2B, p, strict_check=True, check=True):
    """Sets the translation of a transform.

    Parameters
    ----------
    A2B : array-like, shape (batch_size, 4, 4)
        Batched transforms from frame A to frame B

    p : array-like, shape (batch_size, 3)
        Batched translations

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    A2B : array, shape (batch_size, 4, 4)
        Batched transforms from frame A to frame B
    """
    if check:
        A2B = batch_check_transform(A2B, strict_check=strict_check)
    out = A2B.copy()
    out[:, :3, -1] = p
    return out

def batch_transform_from(R, p, strict_check=True):
    """Make batched transforms from rotation matrices and translations.

    .. math::

        \boldsymbol{T}_{BA} = \left(
        \begin{array}{cc}
        \boldsymbol{R} & \boldsymbol{p}\\
        \boldsymbol{0} & 1
        \end{array}
        \right) \in SE(3)

    Parameters
    ----------
    R : array-like, shape (batch_size, 3, 3)
        Batched rotation matrices

    p : array-like, shape (batch_size, 3)
        Batched translations

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    A2B : array, shape (batch_size, 4, 4)
        Batched transforms from frame A to frame B
    """
    A2B = batch_rotate_transform(
        np.tile(np.eye(4), (len(R), 1, 1)), R, strict_check=strict_check, check=False) 
        # np.eye(4), R, strict_check=strict_check, check=False)
    A2B = batch_translate_transform(
        A2B, p, strict_check=strict_check, check=False)
        # A2B, p, strict_check=strict_check, check=False)
    return A2B

def batch_rotate_transform(A2B, R, strict_check=True, check=True):
    """Sets the rotation of a transform for batched inputs.

    Parameters
    ----------
    A2B : array-like, shape (batch_size, 4, 4)
        Batched transforms from frame A to frame B

    R : array-like, shape (batch_size, 3, 3)
        Batched rotation matrices

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    A2B : array, shape (batch_size, 4, 4)
        Transforms from frame A to frame B with rotated orientations
    """
    if check:
        A2B = batch_check_transform(A2B, strict_check=strict_check)
    out = A2B.copy()
    out[..., :3, :3] = R # out[:3, :3] = R
    return out

def batch_dual_quaternion_from_transform(transform):
    """Compute dual quaternion from transformation matrix.

    Parameters
    ----------
    A2B : array-like, shape (batch_size, 4, 4)
        Batched transforms from frame A to frame B

    Returns
    -------
    dq : array, (batch_size, 8)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)
    """
    A2B = batch_check_transform(A2B)
    real = batch_quaternion_from_matrix(A2B[..., :3, :3])
    dual = 0.5 * batch_concatenate_quaternions(
        np.hstack((np.zeros((len(real), 1)), A2B[..., :3, 3:])), real)
        # np.r_[0, A2B[:3, 3]], real)
    return np.hstack((real, dual))

def batch_axis_angle_from_quaternion(q):
    """Compute axis-angle from quaternion.

    This operation is called logarithmic map.

    We usually assume active rotations.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi) so that the mapping is unique.
    """
    q = batch_check_quaternion(q)
    p = q[:, 1:]
    p_norm = np.linalg.norm(p, axis=1)
    
    res = np.zeros((q.shape[0], 4), dtype=np.float64)
    mask_zero_norm = p_norm < np.finfo(float).eps # if p_norm < np.finfo(float).eps:
    
    # return np.array([1.0, 0.0, 0.0, 0.0])

    res[mask_zero_norm] = np.array([1.0, 0.0, 0.0, 0.0])
    
    axis = p / p_norm[:, None]
    
    w_clamped = np.clip(q[:, 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w_clamped)
    
    res[~mask_zero_norm] = batch_norm_axis_angle(np.hstack((axis, angle[:, None])))
    
    # if p_norm < np.finfo(float).eps:
    #     return np.array([1.0, 0.0, 0.0, 0.0])

    # axis = p / p_norm
    # w_clamped = max(min(q[0], 1.0), -1.0)
    # angle = (2.0 * np.arccos(w_clamped),)
    # return batch_norm_axis_angle(np.hstack((axis, angle)))
    return res
    
def batch_q_conj(Q):
    """Conjugate of quaternions.

    The conjugate of a unit quaternion inverts the rotation represented by
    this unit quaternion. The conjugate of a quaternion q is often denoted
    as q*.

    Parameters
    ----------
    Q : array-like, shape (..., 4)
        Unit quaternions to represent rotations: (w, x, y, z)

    Returns
    -------
    Q_c : array, shape (..., 4,)
        Conjugates (w, -x, -y, -z)
    """
    Q = np.asarray(Q)
    out = np.empty_like(Q)
    out[..., 0] = Q[..., 0]
    out[..., 1:] = -Q[..., 1:]
    return out

def batch_screw_parameters_from_dual_quaternion(dq):
    """Compute screw parameters from dual quaternion.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    q : array, shape (3,)
        Vector to a point on the screw axis

    s_axis : array, shape (3,)
        Direction vector of the screw axis

    h : float
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    theta : float
        Parameter of the transformation: theta is the angle of rotation
        and h * theta the translation.
    """
    dq = batch_check_dual_quaternion(dq, unit=True)
    q = np.zeros((dq.shape[0], 3))
    s_axis = np.zeros((dq.shape[0], 3))
    h = np.zeros(dq.shape[0])
    theta = np.zeros(dq.shape[0])
    moment = np.zeros((dq.shape[0], 3))
    distance = np.zeros(dq.shape[0])
    
    real = dq[:, :4]
    dual = dq[:, 4:]
    
    a = batch_axis_angle_from_quaternion(real)
    s_axis = a[:, :3]
    theta = a[:, 3]

    translation = 2 * batch_concatenate_quaternions(dual, batch_q_conj(real))[:, 1:]
    
    #     if abs(theta) < np.finfo(float).eps:
    mask_abs_theta_neg = np.abs(theta) < np.finfo(float).eps # pure translation
    d = np.linalg.norm(translation, axis=1)
    # if d < np.finfo(float).eps:
    mask_d_neg = d < np.finfo(float).eps 
    mask_curr = mask_abs_theta_neg & mask_d_neg
    s_axis[mask_curr] = np.array([1, 0, 0])
    
    # else:
    mask_curr = mask_abs_theta_neg & ~mask_d_neg
    s_axis[mask_curr] = translation[mask_curr] / d[mask_curr][:, None]
    
    
    q[mask_abs_theta_neg] = np.zeros(3)
    dual[mask_abs_theta_neg] = q[mask_abs_theta_neg]
    theta[mask_abs_theta_neg] = d[mask_abs_theta_neg]
    h[mask_abs_theta_neg] = np.inf 
    
    distance[~mask_abs_theta_neg] = np.dot(translation[~mask_abs_theta_neg], s_axis[~mask_abs_theta_neg])
    moment[~mask_abs_theta_neg] = 0.5 * (np.cross(translation[~mask_abs_theta_neg], s_axis[~mask_abs_theta_neg]) +
                                            (translation[~mask_abs_theta_neg] - distance[~mask_abs_theta_neg][:, None] * s_axis[~mask_abs_theta_neg])
                                            / np.tan(0.5 * theta[~mask_abs_theta_neg]))
    dual[~mask_abs_theta_neg] = np.cross(s_axis[~mask_abs_theta_neg], moment[~mask_abs_theta_neg])
    h[~mask_abs_theta_neg] = distance[~mask_abs_theta_neg] / theta[~mask_abs_theta_neg]

    # if abs(theta) < np.finfo(float).eps:
    #     # pure translation
    #     d = np.linalg.norm(translation)
    #     if d < np.finfo(float).eps:
    #         s_axis = np.array([1, 0, 0])
    #     else:
    #         s_axis = translation / d
    #     q = np.zeros(3)
    #     theta = d
    #     h = np.inf
    #     return q, s_axis, h, theta

    # distance = np.dot(translation, s_axis)
    # moment = 0.5 * (np.cross(translation, s_axis) +
    #                 (translation - distance * s_axis)
    #                 / np.tan(0.5 * theta))
    # dual = np.cross(s_axis, moment)
    # h = distance / theta
    return dual, s_axis, h, theta

def batch_screw_axis_from_screw_parameters(q, s_axis, h):
    """Compute screw axis representation from screw parameters.

    Parameters
    ----------
    q : array-like, shape (3,)
        Vector to a point on the screw axis

    s_axis : array-like, shape (3,)
        Direction vector of the screw axis

    h : float
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    Returns
    -------
    screw_axis : array, shape (6,)
        Screw axis described by 6 values
        (omega_1, omega_2, omega_3, v_1, v_2, v_3),
        where the first 3 components are related to rotation and the last 3
        components are related to translation.
    """
    screw_axis = np.zeros((q.shape[0], 6))
    q, s_axis, h = batch_check_screw_parameters(q, s_axis, h)

    # if np.isinf(h):  # pure translation
    mask_pure_translation = np.isinf(h) 
    
    #     return np.r_[0.0, 0.0, 0.0, s_axis]
    screw_axis[mask_pure_translation, 3:] = s_axis[mask_pure_translation] 
    
    # return np.r_[s_axis, np.cross(q, s_axis) + h * s_axis]
    screw_axis[~mask_pure_translation] = np.hstack((s_axis[~mask_pure_translation], np.cross(q[~mask_pure_translation], s_axis[~mask_pure_translation]) + h[~mask_pure_translation][:, np.newaxis] * s_axis[~mask_pure_translation]))
    
    return screw_axis

def batch_screw_axis_from_euler_angles_and_translation(euler_angles, translation):
    """Compute screw axis representation from Euler angles and translation.
    
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
    euler_angles = np.asarray(euler_angles)
    translation = np.asarray(translation)
    
    # assert shape of inputs 
    assert euler_angles.shape[1] == 3, f"Expected euler_angles to have shape (N, 3), but got {euler_angles.shape}"
    assert translation.shape[1] == 3, f"Expected translation to have shape (N, 3), but got {translation.shape}"
    assert euler_angles.shape[0] != translation.shape[0], f"Number of rows in euler_angles and translation must be the same."
    
    # compute active rotation matrix from any Euler angles
    R = batch_matrix_from_euler(e=euler_angles, i=0, j=1, k=2, extrinsic=True)

    # Make transformation from rotation matrix and translation.
    transform = batch_transform_from(R, translation)
    
    # Compute dual quaternion from transformation matrix.
    dq = batch_dual_quaternion_from_transform(transform)

    # Compute screw parameters from dual quaternion.
    q, s_axis, h, _ = batch_screw_parameters_from_dual_quaternion(dq)
    
    # Compute screw axis representation from screw parameters.
    screw_axis = batch_screw_axis_from_screw_parameters(q, s_axis, h)
    
    assert screw_axis.shape[1] == 6, f"Screw axis should be an array of 6 elements, but got {screw_axis} of length {screw_axis.shape[1]}"
    return screw_axis