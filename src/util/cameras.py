import torch
from pytorch3d.renderer.cameras import _R, _T, PerspectiveCameras, _FocalLengthType, get_screen_to_ndc_transform, get_ndc_to_screen_transform
from pytorch3d.common.datatypes import Device
from pytorch3d.transforms import Transform3d
from typing import Optional, Union, Tuple, List
import logging
import numpy as np

EPS = 1e-10

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
        world_to_proj_transform = self.get_full_projection_transform(**kwargs) # camera --> screen
        screen_to_patch_ndc_transform = self.get_patch_ndc_camera_transform(patch_size, patch_center, **kwargs) # screen --> patch ndc
        world_to_patch_ndc_transform = world_to_proj_transform.compose(screen_to_patch_ndc_transform) # camera --> patch ndc
        return world_to_patch_ndc_transform
    
    def transform_points_world_from_patch_ndc(self, points,
                                            patch_size,
                                            patch_center,
                                            eps: Optional[float] = None, **kwargs) -> torch.Tensor:
        world_to_ndc_transform = self.get_full_projection_transform(**kwargs) # camera --> screen/ndc
        if not self.in_ndc():
            to_ndc_transform = self.get_ndc_camera_transform(**kwargs) # screen/ndc --> ndc
            world_to_ndc_transform = world_to_ndc_transform.compose(to_ndc_transform) # camera --> ndc
        to_patch_ndc_transform = self.get_patch_ndc_camera_transform(patch_size, patch_center, **kwargs) # screen/ndc --> patch ndc
        points_patch_ndc = points.to(self.device)
        points_ndc = to_patch_ndc_transform.inverse().transform_points(points_patch_ndc, eps=None)
        points_world = world_to_ndc_transform.inverse().transform_points(points_ndc, eps=None)
        return points_world
    
    def transform_points_patch_ndc(self, points, 
                                   patch_size, 
                                   patch_center,
                                   eps: Optional[float] = None, **kwargs) -> torch.Tensor:
        world_to_ndc_transform = self.get_full_projection_transform(**kwargs) # camera --> screen/ndc
        world_to_ndc_transform = world_to_ndc_transform.to(self.device)
        points = points.to(self.device)
        points_screen = world_to_ndc_transform.transform_points(points, eps=None)
        # print("points_screen world_to_ndc_transform.transform_points", points_screen)
        # print("world_to_ndc_transform 1", world_to_ndc_transform.get_matrix())
        if not self.in_ndc():
            to_ndc_transform = self.get_ndc_camera_transform(**kwargs) # screen/ndc --> ndc
            world_to_ndc_transform = world_to_ndc_transform.compose(to_ndc_transform) # camera --> ndc
        # print("world_to_ndc_transform 2", world_to_ndc_transform.get_matrix())    
        to_patch_ndc_transform = self.get_patch_ndc_camera_transform(patch_size, patch_center, **kwargs) # screen/ndc --> patch ndc
        # print("to_patch_ndc_transform", to_patch_ndc_transform.get_matrix())
        # world_to_patch_ndc_transform = world_to_ndc_transform.compose(to_patch_ndc_transform) # camera --> patch ndc
        points = points.to(self.device)
        points_ndc = world_to_ndc_transform.transform_points(points, eps=None)
        
        # print("points_screen: ", points_screen)
        points_patch_ndc = to_patch_ndc_transform.transform_points(points_ndc, eps=None)
        # print("inside transform_points_patch_ndc")
        # print("points_world: ", points)
        # print("points_ndc: ", points_ndc)
        # points_patch_ndc = world_to_patch_ndc_transform.transform_points(points, eps=eps)
        # print("points_patch_ndc: ", points_patch_ndc)
        
        recovered_points_world = self.transform_points_world_from_patch_ndc(points_patch_ndc, patch_size, patch_center, eps=None, **kwargs)
        # print("recovered_points_world: ", recovered_points_world)
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
    
    def z_camera_to_ndc(self, z: torch.Tensor) -> torch.Tensor:
        """
        Scale z values from camera space to NDC space.
        
        Args:
            z: z values in camera space. A tensor of shape (N, H, W).
        
        Returns:
            z: z values in NDC space. A tensor of shape (N, H, W).
        """
        # TODO: Check if this is correct
        return 2 * z / (self.zfar - self.znear) - (self.zfar + self.znear) / (self.zfar - self.znear)
    
    def z_ndc_to_camera(self, z: torch.Tensor) -> torch.Tensor:
        """
        Scale z values from NDC space to camera space.
        
        Args:
            z: z values in NDC space. A tensor of shape (N, H, W).
        
        Returns:
            z: z values in camera space. A tensor of shape (N, H, W).
        """
        # TODO: Check if this is correct
        return 0.5 * (z + 1) * (self.zfar - self.znear) + self.znear

    def z_patch_to_ndc(self, z_ndc: torch.Tensor, patch_size: Tuple[int, int], image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Scale z values from NDC space to patch space.
        """
        # TODO: Check if this is correct
        im_size = torch.min(image_size, dim=1).values
        patch_size = torch.min(patch_size, dim=1).values
        # send to cpu device
        device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
        im_size = im_size.to(device)
        patch_size = patch_size.to(device)
        z_ndc = z_ndc.to(device)
        scale = patch_size / im_size 
        z_patch = z_ndc * scale
        return z_patch
    
    def z_ndc_to_patch(self, z_patch: torch.Tensor, patch_size: Tuple[int, int], image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Scale z values from patch space to NDC space to make the network aware of the patch size.
        """
        # TODO: Check if this is correct
        im_size = torch.min(image_size, dim=1).values
        patch_size = torch.min(patch_size, dim=1).values
        # send to cpu device
        device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
        im_size = im_size.to(device)
        patch_size = patch_size.to(device)
        z_patch = z_patch.to(device)
        scale = im_size / patch_size
        z_ndc = z_patch * scale
        return z_ndc

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
    patch_ndc_to_ndc_transform = ndc_to_patch_ndc_transform.inverse()
    return patch_ndc_to_ndc_transform

def get_ndc_to_patch_ndc_transform(
    cameras,
    with_xyflip: bool = False,
    image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    patch_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    patch_center: Optional[Union[List, Tuple, torch.Tensor]] = None,
) -> Transform3d:
    """
    Patch --> NDC
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
    
    # We require the image size, which is necessary for the transform
    if image_size is None:
        msg = "For NDC to screen conversion, image_size=(height, width) needs to be specified."
        raise ValueError(msg)
    # We require the patch size, which is necessary for the transform
    if patch_size is None:
        msg = "For NDC to screen conversion, patch_size=(patch_height, patch_width) needs to be specified."
        raise ValueError(msg)
    # We require the patch center, which is necessary for the transform
    if patch_center is None:
        msg = "For NDC to screen conversion, patch_center=(patch_center_x, patch_center_y) needs to be specified."
        raise ValueError(msg)
    
    K = torch.zeros((cameras._N, 4, 4), device=cameras.device, dtype=torch.float32)
    if not torch.is_tensor(image_size):
        image_size = torch.tensor(image_size, device=cameras.device)
    
    if not torch.is_tensor(patch_size):
        patch_size = torch.tensor(patch_size, device=cameras.device)
        
    if not torch.is_tensor(patch_center):
        patch_center = torch.tensor(patch_center, device=cameras.device)
        
    # pyre-fixme[16]: Item `List` of `Union[List[typing.Any], Tensor, Tuple[Any,
    #  ...]]` has no attribute `view`.
    # b, h, w - shape of the imagesize patchsize and patchcenter
    batch_size = image_size.shape[0]
    image_sizes = image_size.view(batch_size, 1, 2)
    image_heights, image_widths = image_sizes[..., 0], image_sizes[..., 1]
    
    patch_size = patch_size.view(batch_size, 1, 2)
    patch_height, patch_width = patch_size[..., 0], patch_size[..., 1]
    
    patch_center = patch_center.view(batch_size, 1, 2)
    cx_patch, cy_patch = patch_center[..., 0], patch_center[..., 1]

    # For non square images, we scale the points such that the aspect ratio is preserved.
    # We assume that the image is centered at the origin
    # patch ndc --> ndc
    # send to available device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # send to device
    patch_width = patch_width.to(device)
    patch_height = patch_height.to(device)
    image_widths = image_widths.to(device)
    image_heights = image_heights.to(device)
    cx_patch = cx_patch.to(device)
    cy_patch = cy_patch.to(device)
    # print("patch_size", patch_size)
    patch_size = patch_size.squeeze(0)
    # print("patch_size", patch_size)
    scale = (image_size.min(dim=1).values - 0.0).to(device)
    patch_scale = (patch_size.min(dim=-1).values - 0.0).to(device)
    # print("scale", scale, np.shape(scale))
    # print("patch_scale", patch_scale, np.shape(patch_scale))
    K[:, 0, 0] = 2 * (patch_scale / scale)
    K[:, 1, 1] = 2 * (patch_scale / scale)
    # print("cx_patch, cy_patch", cx_patch, cy_patch)
    
    tx = -(2* (cx_patch.view(-1) / scale) - image_widths.view(-1) / scale)
    ty = -(2* (cy_patch.view(-1) / scale) - image_heights.view(-1) / scale)
    # print("tx, ty", tx, ty)
    K[:, 0, 3] = 2 * (patch_scale / scale) * tx
    K[:, 1, 3] = 2 * (patch_scale / scale) * ty
    K[:, 2, 2] = 1.0
    K[:, 3, 3] = 1.0
    # print("K in get_ndc_to_patch_ndc_transform", K)
    # Transpose the projection matrix as PyTorch3D transforms use row vectors.
    # transform = Transform3d(
    #     matrix=K.transpose(1, 2).contiguous(), device=cameras.device
    # )
    
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
