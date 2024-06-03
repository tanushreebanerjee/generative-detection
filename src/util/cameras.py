import torch
from pytorch3d.renderer.cameras import _R, _T, PerspectiveCameras, _FocalLengthType, get_ndc_to_screen_transform
from pytorch3d.common.datatypes import Device
from pytorch3d.transforms import Transform3d
from typing import Optional, Union, Tuple, List
import logging
import numpy as np

def robust_inverse(transforms, eps=1e-6):
    try:
        ret = transforms.inverse()
    except Exception as e:
        # add a small value to the diagonal to make the matrix invertible
        matrix = transforms.get_matrix()
        delta = torch.eye(matrix.shape[-1], device=matrix.device) * eps
        matrix = matrix + delta
        transforms = Transform3d(matrix=matrix, device=matrix.device)
        ret = transforms.inverse()
    return ret
        
class PatchPerspectiveCameras(PerspectiveCameras):
    """
    A class representing patch perspective cameras.
    
    This class extends the PerspectiveCameras class and provides additional functionality for patch perspective cameras.
    """
    _FIELDS = (
        "K",
        "R",
        "T",
        "focal_length",
        "principal_point",
        "_in_ndc",  # arg is in_ndc but attribute set as _in_ndc
        "image_size",
        "znear",
        "zfar",
    )

    _SHARED_FIELDS = ("_in_ndc",)
    
    def __init__(self, 
                 znear: float = 0.0,
                 zfar: float = 80.0,
                 focal_length: _FocalLengthType = 1.0, 
                 principal_point=((0.0, 0.0),), 
                 R: torch.Tensor = _R, 
                 T: torch.Tensor = _T,
                 K: Optional[torch.Tensor] = None,
                 device: Device = "cpu",
                 in_ndc: bool = False, # We specify the camera parameters in screen space. We want to convert to and from patch NDC space.
                 image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
                 ) -> None:
        """
        Initialize the PatchPerspectiveCameras object.
        
        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            focal_length: Focal length of the camera in world units.
                A tensor of shape (N, 1) or (N, 2) for
                square and non-square pixels respectively.
            principal_point: xy coordinates of the center of
                the principal point of the camera in pixels.
                A tensor of shape (N, 2).
            in_ndc: True if camera parameters are specified in NDC.
                If camera parameters are in screen space, it must
                be set to False.
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            K: (optional) A calibration matrix of shape (N, 4, 4)
                If provided, don't need focal_length, principal_point
            image_size: (height, width) of image size.
                A tensor of shape (N, 2) or a list/tuple. Required for screen cameras.
            device: torch.device or string
        """
        assert not in_ndc, "Camera parameters must be specified in screen space."
        super().__init__(focal_length, principal_point, R, T, K, device, in_ndc, image_size)
        self.znear = znear
        self.zfar = zfar
     
    def get_patch_projection_transform(self, patch_size, patch_center, **kwargs):
        # world --> ndc
        world_to_ndc_transform = self.get_full_projection_transform(**kwargs)
        
        if not self.in_ndc():
            to_ndc_transform = self.get_ndc_camera_transform(**kwargs)
            world_to_ndc_transform = world_to_ndc_transform.compose(to_ndc_transform)
        
        # ndc --> patch ndc
        ndc_to_patch_ndc_transform = self.get_patch_ndc_camera_transform(patch_size, patch_center, **kwargs) 
        
        # world --> patch ndc
        transform = world_to_ndc_transform.compose(ndc_to_patch_ndc_transform)
        
        return transform
        
    def transform_points_world_from_patch_ndc(self, points_patch_ndc, # points in patch ndc
                                              depth,
                                            patch_size,
                                            patch_center,
                                            eps: Optional[float] = None, **kwargs) -> torch.Tensor:
        points_patch_ndc = points_patch_ndc.to(self.device)
        world_depth = torch.zeros([1, 1, 3])
        world_depth[..., -1] = depth
        screen_depth = self.transform_points_screen(world_depth)
        # patch --> ndc
        points_ndc = robust_inverse(self.get_patch_ndc_camera_transform(patch_size, patch_center, **kwargs)).transform_points(points_patch_ndc, eps=1e-7)
        # ndc --> screen
        points_screen = robust_inverse(self.get_ndc_camera_transform()).transform_points(points_ndc, eps=1e-7)
        # points_screen[..., -1] = depth
        # screen --> camera
        points_screen[..., -1] = screen_depth[..., -1] 
        points_camera = -self.get_projection_transform().inverse().transform_points(points_screen, eps=1e-7)
        points_camera[..., -1] = -points_camera[..., -1]
        return points_camera
        
    def transform_points_patch_ndc(self, points, 
                                   patch_size, 
                                   patch_center,
                                   eps: Optional[float] = None, **kwargs) -> torch.Tensor:
        # camera --> ndc points
        points = points.to(self.device)
        # points_ndc = self.transform_points_ndc(points, eps=None)
        points_screen = self.transform_points_screen(points)
        points_ndc = self.get_ndc_camera_transform().transform_points(points_screen)
        
        # ndc --> patch ndc transform
        ndc_to_patch_ndc_transform = self.get_patch_ndc_camera_transform(patch_size, patch_center, **kwargs) 
        
        # ndc --> patch ndc points
        points_patch_ndc = ndc_to_patch_ndc_transform.transform_points(points_ndc, eps=1e-7)
        ##### DEBUGGING #####
        # world_depth = points[..., -1]
        # abc = self.transform_points_world_from_patch_ndc(points_patch_ndc, world_depth, patch_size, patch_center, eps=1e-7)
        ##### DEBUGGING #####
        # if first 2 values are > 0.5 print
        if points_patch_ndc.dim() == 3:
            points_patch_ndc = points_patch_ndc.view(-1)
        
        return points_patch_ndc
    
    def get_patch_ndc_camera_transform(self,
                                        patch_size,
                                        patch_center,
                                        **kwargs):
        """
        Returns the transform from camera projection space (screen or NDC) to Patch NDC space.

        This transform leaves the depth unchanged.

        Important: This transforms assumes PyTorch3D conventions for the input points,
        i.e. +X left, +Y up.

        Args:
            patch_size (tuple): The size of the patch in pixels.
            patch_center (tuple): The center of the patch in pixels.
            **kwargs: Additional keyword arguments.

        Returns:
            patch_ndc_transform (Transform): The camera transform from screen coordinates to patch NDC coordinates.
        """
        image_size = kwargs.get("image_size", self.get_image_size())
        ndc_to_patch_ndc_transform = get_ndc_to_patch_ndc_transform( # ndc --> patch ndc
            self, with_xyflip=False, image_size=image_size, patch_size=patch_size, patch_center=patch_center
        )
        return ndc_to_patch_ndc_transform # ndc --> patch ndc

################################################
# Helper functions for patch perspective cameras
################################################

def get_patch_ndc_to_ndc_transform(
    cameras,
    with_xyflip: bool = False,
    image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    patch_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    patch_center: Optional[Union[List, Tuple, torch.Tensor]] = None,
    ) -> Transform3d:
    """
    Patch NDC --> NDC 
    PyTorch3D patch NDC to screen conversion.
    Conversion from PyTorch3D's patch NDC space (+X left, +Y up) to screen/image space
    (+X right, +Y down, origin top left).

    Args:
        cameras
        with_xyflip: flips x- and y-axis if set to True.
    Optional kwargs:
        image_size: ((height, width),) specifying the height, width
        of the image. If not provided, it reads it from cameras.

    We represent the NDC to screen conversion as a Transform3d
    with projection matrix

    K = [
            [s,   0,    0,  cx],
            [0,   s,    0,  cy],
            [0,   0,    1,   0],
            [0,   0,    0,   1],
    ]

    """ 
    ndc_to_patch_ndc_transform = get_ndc_to_patch_ndc_transform(
        cameras,
        with_xyflip=with_xyflip,
        image_size=image_size,
        patch_size=patch_size,
        patch_center=patch_center,
    )
    patch_ndc_to_ndc_transform = robust_inverse(ndc_to_patch_ndc_transform)
    return patch_ndc_to_ndc_transform

def get_ndc_to_patch_ndc_transform(
    cameras,
    with_xyflip: bool = False,
    image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    patch_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    patch_center: Optional[Union[List, Tuple, torch.Tensor]] = None,
) -> Transform3d:
    """
    NDC --> patch NDC
    Screen to PyTorch3D Patch NDC conversion (inverse of NDC to screen conversion).
    Conversion from screen/image space (+X right, +Y down, origin top left)
    to PyTorch3D's NDC space (+X left, +Y up) with respect to the patch.

    Args:
        cameras
        with_xyflip: flips x- and y-axis if set to True.
    Optional kwargs:
        image_size: ((height, width),) specifying the height, width
        of the image. If not provided, it reads it from cameras.

    We represent the screen to NDC conversion as a Transform3d
    with projection matrix

    K = [
            [1/s,    0,    0,  cx/s],
            [  0,  1/s,    0,  cy/s],
            [  0,    0,    1,     0],
            [  0,    0,    0,     1],
    ]

    """
    device = cameras.device
    # We require the image size, which is necessary for the transform
    if image_size is None:
        raise ValueError("For NDC to patch conversion, image_size=(height, width) needs to be specified.")
    if not torch.is_tensor(image_size):
        image_size = torch.tensor(image_size, device=cameras.device)
    
    batch_size = image_size.shape[0]
    image_sizes = image_size.view(batch_size, 1, 2)
    # We require the patch size, which is necessary for the transform
    if patch_size is None:
        raise ValueError("For NDC to patch conversion, patch_size=(patch_height, patch_width) needs to be specified.")
    if not torch.is_tensor(patch_size):
        patch_size = torch.tensor(patch_size, device=cameras.device)
    patch_size = patch_size.view(batch_size, 1, 2)
    # We require the patch center, which is necessary for the transform
    if patch_center is None:
        raise ValueError("For NDC to patch conversion, patch_center=(patch_center_x, patch_center_y) needs to be specified.")
    if not torch.is_tensor(patch_center):
        patch_center = torch.tensor(patch_center, device=cameras.device)
    # Patch center given in screen coordinates
    patch_center = patch_center.view(batch_size, 1, 2).to(cameras.device)
    patch_center = torch.cat([patch_center, torch.zeros_like(patch_center[..., :1])], dim=-1)
    
    K = torch.zeros((cameras._N, 4, 4), device=cameras.device, dtype=torch.float32)
    cx_screen, cy_screen = patch_center[..., 0], patch_center[..., 1]
    
    screen_to_ndc_transform = cameras.get_ndc_camera_transform()
    patch_center_ndc = screen_to_ndc_transform.transform_points(patch_center)
    cx_ndc = patch_center_ndc[..., 0]
    cy_ndc = patch_center_ndc[..., 1]
    
    
    patch_size = patch_size.squeeze(0)

    scale = (image_size.min(dim=1).values - 0.0).to(device)
    patch_scale = (patch_size.min(dim=-1).values - 0.0).to(device)
    principal_point = cameras.get_principal_point()
    
    K[:, 0, 0] = (patch_scale / scale)
    K[:, 1, 1] = (patch_scale / scale)
    
    # tx = cx_ndc
    # ty = cy_ndc
    
    K[:, 3, 0] = -(patch_scale / scale) * cx_ndc
    K[:, 3, 1] = -(patch_scale / scale) * cy_ndc
    K[:, 2, 2] = 1.0
    K[:, 3, 3] = 1.0
    
    transform = Transform3d(
        matrix=K.contiguous(), device=cameras.device
    )

    if with_xyflip:
        # flip x, y axis
        xyflip = torch.eye(4, device=cameras.device, dtype=torch.float32)
        xyflip[0, 0] = -1.0
        xyflip[1, 1] = -1.0
        xyflip = xyflip.view(1, 4, 4).expand(cameras._N, -1, -1)
        xyflip_transform = Transform3d(
            matrix=xyflip.transpose(1, 2).contiguous(), device=cameras.device
        )
        transform = transform.compose(xyflip_transform)
    return transform

def z_patch_to_world(z_patch, patch_resampling_factor):
    z_world = z_patch * patch_resampling_factor
    return z_world

def z_world_to_patch(z_world, patch_resampling_factor):
    z_patch = z_world / patch_resampling_factor
    return z_patch

def z_patch_to_learned(z_patch, zmin, zmax):
    z_learned = 2 * ((z_patch - zmin) / (zmax - zmin)) - 1
    return z_learned

def z_learned_to_patch(z_learned, zmin, zmax):
    z_patch = 0.5 * (z_learned + 1) * (zmax - zmin) + zmin
    return z_patch

def z_world_to_learned(z_world, zmin, zmax, patch_resampling_factor):
    z_patch = z_world_to_patch(z_world, patch_resampling_factor)
    z_learned = z_patch_to_learned(z_patch, zmin, zmax)
    return z_learned

def z_learned_to_world(z_learned, zmin, zmax, patch_resampling_factor):
    z_patch = z_learned_to_patch(z_learned, zmin, zmax)
    z_world = z_patch_to_world(z_patch, patch_resampling_factor)
    return z_world
