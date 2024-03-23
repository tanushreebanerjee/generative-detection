import warnings
import numpy as np

R_SHAPE = (3, 3)

def check_matrix_batch(R, tolerance=1e-6, strict_check=True):
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
        Validated rotation matrix

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

def check_transform_batch(A2B, strict_check=True):
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
    if A2B.ndim != 3 or A2B.shape[1:] != (4, 4):
        raise ValueError("Expected homogeneous transformation matrices with "
                         "shape (batch_size, 4, 4), got array-like object with shape %s"
                         % (A2B.shape,))
    check_matrix_batch(A2B[..., :3, :3], strict_check=strict_check)
    if not np.allclose(A2B[..., 3], np.array([0.0, 0.0, 0.0, 1.0])):
        error_msg = ("Excpected homogeneous transformation matrix with "
                     "[0, 0, 0, 1] at the bottom, got %r" % A2B)
        if strict_check:
            raise ValueError(error_msg)
        warnings.warn(error_msg)
    return A2B