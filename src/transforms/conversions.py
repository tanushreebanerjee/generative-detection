import numpy as np
from src.transforms.utils import check_transform_batch
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

def active_matrix_from_angle_batch(basis, angle):
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
    
def matrix_from_euler_batch(e, i, j, k, extrinsic):
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

    R = active_matrix_from_angle_batch(k, gamma).dot(
        active_matrix_from_angle_batch(j, beta)).dot(
        active_matrix_from_angle_batch(i, alpha))

    return R

def translate_transform_batch(A2B, p, strict_check=True, check=True):
    """Sets the translation of a transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    p : array-like, shape (3,)
        Translation

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B
    """
    if check:
        A2B = check_transform_batch(A2B, strict_check=strict_check)
    out = A2B.copy()
    out[:3, -1] = p
    return out

def transform_from_batch(R, p, strict_check=True):
    r"""Make transformation from rotation matrix and translation.

    .. math::

        \boldsymbol{T}_{BA} = \left(
        \begin{array}{cc}
        \boldsymbol{R} & \boldsymbol{p}\\
        \boldsymbol{0} & 1
        \end{array}
        \right) \in SE(3)

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    p : array-like, shape (3,)
        Translation

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B
    """
    A2B = rotate_transform_batch(
        np.eye(4), R, strict_check=strict_check, check=False)
    A2B = translate_transform_batch(
        A2B, p, strict_check=strict_check, check=False)
    return A2B


def rotate_transform_batch(A2B, R, strict_check=True, check=True):
    """Sets the rotation of a transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B
    """
    if check:
        A2B = check_transform_batch(A2B, strict_check=strict_check)
    out = A2B.copy()
    out[:3, :3] = R
    return out