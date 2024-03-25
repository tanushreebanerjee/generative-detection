import warnings
import numpy as np
# Source:
# Code adapted from https://github.com/dfki-ric/pytransform3d

# constants
R_SHAPE = (3, 3)
HOMOGENEOUS_SHAPE = (4, 4)
TWO_PI = 2.0 * np.pi

def batch_check_matrix(R, tolerance=1e-6, strict_check=True):
    """Input validation of a rotation matrix.

    We check whether R multiplied by its inverse is approximately the identity
    matrix

    .. math::

        \boldsymbol{R}\boldsymbol{R}^T = \boldsymbol{I}

    and whether the determinant is positive

    .. math::

        det(\boldsymbol{R}) > 0

    Parameters
    ----------
    R : array-like, shape (batch_size, 3, 3)
        Rotation matrix

    tolerance : float, optional (default: 1e-6)
        Tolerance threshold for checks. Default tolerance is the same as in
        assert_rotation_matrix(R).

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    R : array, shape (batch_size, 3, 3)
        Validated rotation matrices

    Raises
    ------
    ValueError
        If input is invalid

    See Also
    --------
    norm_matrix : Enforces orthonormality of a rotation matrix.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 3 or R.shape[1:] != R_SHAPE:
        raise ValueError("Expected rotation matrices with shape (batch_size, 3, 3), got "
                         "array-like object with shape %s" % (R.shape,))
    RRT = np.einsum('...ij,...jk->...ik', R, R.transpose(0, 2, 1)) # RRT = np.dot(R, R.T)
    if not np.allclose(RRT, np.eye(3), atol=tolerance):
        error_msg = ("Expected rotation matrix, but it failed the test "
                     "for inversion by transposition. np.dot(R, R.T) "
                     "gives %r" % RRT)
        if strict_check:
            raise ValueError(error_msg)
        warnings.warn(error_msg)
    R_det = np.linalg.det(R)
    if np.any(R_det < 0.0):
        error_msg = ("Expected rotation matrix, but it failed the test "
                     "for the determinant, which should be 1 but is %g; "
                     "that is, it probably represents a rotoreflection"
                     % R_det)
        if strict_check:
            raise ValueError(error_msg)
        warnings.warn(error_msg)
    return R

def batch_check_transform(A2B, strict_check=True):
    """Input validation of transform.

    Parameters
    ----------
    A2B : array-like, shape (batch_size, 4, 4)
        Transforms from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    A2B : array, shape (batch_size, 4, 4)
        Validated transforms from frame A to frame B

    Raises
    ------
    ValueError
        If input is invalid
    """
    A2B = np.asarray(A2B, dtype=np.float64)
    if A2B.ndim != 3 or A2B.shape[1:] != HOMOGENEOUS_SHAPE:
        raise ValueError("Expected homogeneous transformation matrices with "
                         "shape (batch_size, 4, 4), got array-like object with shape %s"
                         % (A2B.shape,))
    batch_check_matrix(A2B[..., :3, :3], strict_check=strict_check)
    if not np.allclose(A2B[..., 3], np.array([0.0, 0.0, 0.0, 1.0])):
        error_msg = ("Excpected homogeneous transformation matrix with "
                     "[0, 0, 0, 1] at the bottom, got %r" % A2B)
        if strict_check:
            raise ValueError(error_msg)
        warnings.warn(error_msg)
    return A2B

def batch_quaternion_from_matrix(R, strict_check=True):
    """Compute quaternion from rotation matrix for batched inputs.

    We usually assume active rotations.

    .. warning::

        When computing a quaternion from the rotation matrix there is a sign
        ambiguity: q and -q represent the same rotation.

    Parameters
    ----------
    R : array-like, shape (batch_size, 3, 3)
        Batched rotation matrices

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    q : array-like, shape (batch_size, 4)
        Unit quaternions to represent rotations: (w, x, y, z)
    """

    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError("Expected rotation matrices with shape (batch_size, 3, 3), "
                         f"got array-like object with shape {R.shape}")
    
    R = batch_check_matrix(R, strict_check=strict_check)
    q = np.empty((len(R), 4)) # Initialize an array to store quaternions

    # Source:
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions
    
    trace = np.trace(R, axis1=1, axis2=2) # Compute the trace of each rotation matrix in the batch
    mask_pos_trace = trace > 0.0
    
    # Compute quaternions for cases where trace > 0.0:
    sqrt_trace = np.sqrt(1.0 + trace[mask_pos_trace])
    q[mask_pos_trace, 0] = 0.5 * sqrt_trace
    q[mask_pos_trace, 1] = 0.5 / sqrt_trace * (R[mask_pos_trace, 2, 1] - R[mask_pos_trace, 1, 2])
    q[mask_pos_trace, 2] = 0.5 / sqrt_trace * (R[mask_pos_trace, 0, 2] - R[mask_pos_trace, 2, 0])
    q[mask_pos_trace, 3] = 0.5 / sqrt_trace * (R[mask_pos_trace, 1, 0] - R[mask_pos_trace, 0, 1])

    # Handle cases where trace <= 0
    mask_not_pos_trace = ~mask_pos_trace
    
    # if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
    mask_x_largest = (R[mask_not_pos_trace, 0, 0] > R[mask_not_pos_trace, 1, 1]) & (R[mask_not_pos_trace, 0, 0] > R[mask_not_pos_trace, 2, 2])
    mask_x = mask_not_pos_trace & mask_x_largest
    sqrt_trace_x = np.sqrt(1.0 + R[mask_x, 0, 0] - R[mask_x, 1, 1] - R[mask_x, 2, 2])
    q[mask_x, 0] = 0.5 / sqrt_trace_x * (R[mask_x, 2, 1] - R[mask_x, 1, 2])
    q[mask_x, 1] = 0.5 * sqrt_trace_x
    q[mask_x, 2] = 0.5 / sqrt_trace_x * (R[mask_x, 1, 0] + R[mask_x, 0, 1])
    q[mask_x, 3] = 0.5 / sqrt_trace_x * (R[mask_x, 0, 2] + R[mask_x, 2, 0])

    #     elif R[1, 1] > R[2, 2]:
    mask_y_largest = ~mask_x_largest & (R[mask_not_pos_trace, 1, 1] > R[mask_not_pos_trace, 2, 2])
    mask_y = mask_not_pos_trace & mask_y_largest
    sqrt_trace_y = np.sqrt(1.0 + R[mask_y, 1, 1] - R[mask_y, 0, 0] - R[mask_y, 2, 2])
    q[mask_y, 0] = 0.5 / sqrt_trace_y * (R[mask_y, 0, 2] - R[mask_y, 2, 0])
    q[mask_y, 1] = 0.5 / sqrt_trace_y * (R[mask_y, 1, 0] + R[mask_y, 0, 1])
    q[mask_y, 2] = 0.5 * sqrt_trace_y
    q[mask_y, 3] = 0.5 / sqrt_trace_y * (R[mask_y, 2, 1] + R[mask_y, 1, 2])
    
    #     else:
    mask_z = mask_not_pos_trace & ~mask_x_largest & ~mask_y_largest
    sqrt_trace_z = np.sqrt(1.0 + R[mask_z, 2, 2] - R[mask_z, 0, 0] - R[mask_z, 1, 1])
    q[mask_z, 0] = 0.5 / sqrt_trace_z * (R[mask_z, 1, 0] - R[mask_z, 0, 1])
    q[mask_z, 1] = 0.5 / sqrt_trace_z * (R[mask_z, 0, 2] + R[mask_z, 2, 0])
    q[mask_z, 2] = 0.5 / sqrt_trace_z * (R[mask_z, 2, 1] + R[mask_z, 1, 2])
    q[mask_z, 3] = 0.5 * sqrt_trace_z
    
    # if trace > 0.0:
    #     sqrt_trace = np.sqrt(1.0 + trace)
    #     q[0] = 0.5 * sqrt_trace
    #     q[1] = 0.5 / sqrt_trace * (R[2, 1] - R[1, 2])
    #     q[2] = 0.5 / sqrt_trace * (R[0, 2] - R[2, 0])
    #     q[3] = 0.5 / sqrt_trace * (R[1, 0] - R[0, 1])
    # else:
    #     if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
    #         sqrt_trace = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
    #         q[0] = 0.5 / sqrt_trace * (R[2, 1] - R[1, 2])
    #         q[1] = 0.5 * sqrt_trace
    #         q[2] = 0.5 / sqrt_trace * (R[1, 0] + R[0, 1])
    #         q[3] = 0.5 / sqrt_trace * (R[0, 2] + R[2, 0])
    #     elif R[1, 1] > R[2, 2]:
    #         sqrt_trace = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
    #         q[0] = 0.5 / sqrt_trace * (R[0, 2] - R[2, 0])
    #         q[1] = 0.5 / sqrt_trace * (R[1, 0] + R[0, 1])
    #         q[2] = 0.5 * sqrt_trace
    #         q[3] = 0.5 / sqrt_trace * (R[2, 1] + R[1, 2])
    #     else:
    #         sqrt_trace = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    #         q[0] = 0.5 / sqrt_trace * (R[1, 0] - R[0, 1])
    #         q[1] = 0.5 / sqrt_trace * (R[0, 2] + R[2, 0])
    #         q[2] = 0.5 / sqrt_trace * (R[2, 1] + R[1, 2])
    #         q[3] = 0.5 * sqrt_trace
    return q

def batch_concatenate_quaternions(Q1, Q2, out=None):
    """Concatenate two batches of quaternions.

    We use Hamilton's quaternion multiplication.

    Suppose we want to apply two extrinsic rotations given by quaternions
    q1 and q2 to a vector v. We can either apply q2 to v and then q1 to
    the result or we can concatenate q1 and q2 and apply the result to v.

    Parameters
    ----------
    Q1 : array-like, shape (..., 4)
        First batch of quaternions

    Q2 : array-like, shape (..., 4)
        Second batch of quaternions

    out : array, shape (..., 4), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Q12 : array, shape (..., 4)
        Batch of quaternions that represents the concatenated rotations q1 * q2
        
    See Also
    --------
    concatenate_rotors : Concatenate rotors, which is the same operation.

    Raises
    ------
    ValueError
        If the input dimensions are incorrect
    """
    Q1 = np.asarray(Q1)
    Q2 = np.asarray(Q2)

    if Q1.ndim != Q2.ndim:
        raise ValueError("Number of dimensions must be the same. "
                         "Got %d for Q1 and %d for Q2." % (Q1.ndim, Q2.ndim))
    for d in range(Q1.ndim - 1):
        if Q1.shape[d] != Q2.shape[d]:
            raise ValueError(
                "Size of dimension %d does not match: %d != %d"
                % (d + 1, Q1.shape[d], Q2.shape[d]))
    if Q1.shape[-1] != 4:
        raise ValueError(
            "Last dimension of first argument does not match. A quaternion "
            "must have 4 entries, got %d" % Q1.shape[-1])
    if Q2.shape[-1] != 4:
        raise ValueError(
            "Last dimension of second argument does not match. A quaternion "
            "must have 4 entries, got %d" % Q2.shape[-1])

    if out is None:
        out = np.empty_like(Q1)

    vector_inner_products = np.sum(Q1[..., 1:] * Q2[..., 1:], axis=-1)
    out[..., 0] = Q1[..., 0] * Q2[..., 0] - vector_inner_products
    out[..., 1:] = (Q1[..., 0, np.newaxis] * Q2[..., 1:] +
                    Q2[..., 0, np.newaxis] * Q1[..., 1:] +
                    np.cross(Q1[..., 1:], Q2[..., 1:]))
    
    # q1 = check_quaternion(q1, unit=False)
    # q2 = check_quaternion(q2, unit=False)
    # q12 = np.empty(4)
    # q12[0] = q1[0] * q2[0] - np.dot(q1[1:], q2[1:])
    # q12[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:])
    
    return out

def batch_check_dual_quaternion(dq, unit=True):
    """Input validation of dual quaternion representation.

    See http://web.cs.iastate.edu/~cs577/handouts/dual-quaternion.pdf

    A dual quaternion is defined as

    .. math::

        \\boldsymbol{\\sigma} = \\boldsymbol{p} + \\epsilon \\boldsymbol{q},

    where :math:`\\boldsymbol{p}` and :math:`\\boldsymbol{q}` are both
    quaternions and :math:`\\epsilon` is the dual unit with
    :math:`\\epsilon^2 = 0`. The first quaternion is also called the real part
    and the second quaternion is called the dual part.

    Parameters
    ----------
    dq : array-like, shape (N, 8)
        Dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    unit : bool, optional (default: True)
        Normalize the dual quaternion so that it is a unit dual quaternion.
        A unit dual quaternion has the properties
        :math:`p_w^2 + p_x^2 + p_y^2 + p_z^2 = 1` and
        :math:`p_w q_w + p_x q_x + p_y q_y + p_z q_z = 0`.

    Returns
    -------
    dq : array, shape (N, 8)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Raises
    ------
    ValueError
        If input is invalid
    """
    dq = np.asarray(dq, dtype=np.float64)
    if dq.ndim != 2 or dq.shape[1] != 8:
        raise ValueError("Expected dual quaternion with shape (N, 8), got "
                         "array-like object with shape %s" % (dq.shape,))
    if unit:
        # Norm of a dual quaternion only depends on the real part because
        # the dual part vanishes with epsilon ** 2 = 0.
        real_norm = np.linalg.norm(dq[:, :4], axis=1)
        mask_non_zero_norm = real_norm != 0.0  # if real_norm == 0.0:
        dq[mask_non_zero_norm] /= real_norm[mask_non_zero_norm, np.newaxis] # return dq / real_norm
        dq[~mask_non_zero_norm] = np.r_[1, 0, 0, 0, dq[~mask_non_zero_norm, 4:]] # return np.r_[1, 0, 0, 0, dq[4:]]  
    return dq

def batch_norm_angle(a):
    """Normalize angle to (-pi, pi].

    It is worth noting that using `numpy.ceil` to normalize angles will lose
    more digits of precision as angles going larger but can keep more digits
    of precision when angles are around zero. In common use cases, for example,
    -10.0*pi to 10.0*pi, it performs well.

    For more discussions on numerical precision:
    https://github.com/dfki-ric/pytransform3d/pull/263

    Parameters
    ----------
    a : float or array-like, shape (n,)
        Angle(s) in radians

    Returns
    -------
    a_norm : float or array, shape (n,)
        Normalized angle(s) in radians
    """
    a = np.asarray(a, dtype=np.float64)
    return a - (np.ceil((a + np.pi) / TWO_PI) - 1.0) * TWO_PI

def batch_norm_vector(v):
    """Normalize vector.

    Parameters
    ----------
    v : array-like, shape (n,)
        nd vector

    Returns
    -------
    u : array, shape (n,)
        nd unit vector with norm 1 or the zero vector
    """
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    mask_non_zero_norm = norm != 0.0 # if norm == 0.0:
    v[mask_non_zero_norm] /= norm[mask_non_zero_norm]
    return v # np.asarray(v) / norm

def batch_check_quaternion(q, unit=True):
    """Input validation of quaternion representation.

    Parameters
    ----------
    q : array-like, shape (4,)
        Quaternion to represent rotation: (w, x, y, z)

    unit : bool, optional (default: True)
        Normalize the quaternion so that it is a unit quaternion

    Returns
    -------
    q : array-like, shape (4,)
        Validated quaternion to represent rotation: (w, x, y, z)

    Raises
    ------
    ValueError
        If input is invalid
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("Expected quaternions with shape (N, 4), got "
                         "array-like object with shape %s" % (q.shape,))
    if unit:
        return batch_norm_vector(q)
    return q

def batch_norm_axis_angle(a):
    """Normalize axis-angle representation.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    a : array, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The length
        of the axis vector is 1 and the angle is in [0, pi). No rotation
        is represented by [1, 0, 0, 0].
    """
    
    a = np.asarray(a, dtype=np.float64)
    angle = a[:, 3]
    norm = np.linalg.norm(a[:, :3], axis=1)
    
    mask_non_zero_norm = norm != 0.0
    mask_non_zero_angle = angle != 0.0

    res = np.empty_like(a)
    
    # if angle == 0.0 or norm == 0.0:
    mask_x = ~mask_non_zero_angle | ~mask_non_zero_norm
    res[mask_x] = np.array([1.0, 0.0, 0.0, 0.0])  # return np.array([1.0, 0.0, 0.0, 0.0])
    
    
    res[mask_non_zero_norm, :3] = a[mask_non_zero_norm, :3] / norm[mask_non_zero_norm, None] # res[:3] = a[:3] / norm
    
    angle = batch_norm_angle(angle)
    
    mask_negative_angle = angle < 0.0  # if angle < 0.0:
    
    angle[mask_negative_angle] *= -1.0     #     angle *= -1.0

    res[mask_non_zero_norm][:, 3] *= -1.0     #     res[:3] *= -1.0
    
    res[:, 3] = angle     # res[3] = angle
    
    # if angle == 0.0 or norm == 0.0:
    #     return np.array([1.0, 0.0, 0.0, 0.0])

    # res = np.empty(4)
    # res[:3] = a[:3] / norm

    # angle = batch_norm_angle(angle)
    # if angle < 0.0:
    #     angle *= -1.0
    #     res[:3] *= -1.0

    # res[3] = angle

    return res

def batch_check_screw_parameters(q, s_axis, h):
    r"""Input validation of screw parameters.

    The parameters :math:`(\boldsymbol{q}, \hat{\boldsymbol{s}}, h)`
    describe a screw.

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
    q : array, shape (3,)
        Vector to a point on the screw axis. Will be set to zero vector when
        pitch is infinite (pure translation).

    s_axis : array, shape (3,)
        Unit direction vector of the screw axis

    h : float
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    Raises
    ------
    ValueError
        If input is invalid
    """
    s_axis = np.asarray(s_axis, dtype=np.float64)
    if s_axis.ndim != 2 or s_axis.shape[1] != 3:
        raise ValueError("Expected 3D vector with shape (B, 3), got array-like "
                         "object with shape %s" % (s_axis.shape,))
    if np.any(np.linalg.norm(s_axis, axis=1) == 0.0):
        raise ValueError("s_axis must not have norm 0")

    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("Expected 3D vector with shape (B, 3), got array-like "
                         "object with shape %s" % (q.shape,))
    q[np.isinf(h)] = 0.0  # Set q to zero for pure translation screws

    return q, batch_norm_vector(s_axis), h