import torch
from pytorch3d.renderer.cameras import _R, _T, PerspectiveCameras, _FocalLengthType, get_screen_to_ndc_transform
from pytorch3d.common.datatypes import Device
from pytorch3d.transforms import Transform3d
from typing import Optional, Union, Tuple, List

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
        
    def get_patch_ndc_camera_transform(self,
                                        patch_size,
                                 patch_center,
                                 **kwargs):
        # get_ndc_camera_transform  - Screen --> NDC transform.
        # get_patch_ndc_to_ndc_transform - NDC --> patch
        # overall: screen --> patch ndc
        ndc_camera_transform = self.get_ndc_camera_transform(**kwargs)
        patch_ndc_to_ndc_transform = get_patch_ndc_to_ndc_transform(
            self, with_xyflip=False, image_size=self.get_image_size(),
            patch_size=patch_size, patch_center=patch_center
        )
        
        patch_ndc_camera_transform = ndc_camera_transform.compose(patch_ndc_to_ndc_transform)
        return patch_ndc_camera_transform
        
    def transform_points_ndc( self, points, eps: Optional[float] = None, **kwargs) -> torch.Tensor:
        """
        Camera --> NDC Patch
        Transforms points from PyTorch3D world/camera space to NDC patch space.
        Scales z values from camera space to NDC space.
        """
        world_to_ndc_transform = self.get_full_projection_transform(**kwargs)
        if not self.in_ndc():
            to_ndc_transform = self.get_ndc_camera_transform(**kwargs)
            world_to_ndc_transform = world_to_ndc_transform.compose(to_ndc_transform)
            
        new_points = world_to_ndc_transform.transform_points(points, eps=eps)
        
        # scale z values from camera space to NDC space
        z = new_points[..., 2]
        new_points[..., 2] = self.z_camera_to_ndc(z)
        
        return new_points
    
    def z_camera_to_ndc(self, z: torch.Tensor) -> torch.Tensor:
        """
        Scale z values from camera space to NDC space.
        
        Args:
            z: z values in camera space. A tensor of shape (N, H, W).
        
        Returns:
            z: z values in NDC space. A tensor of shape (N, H, W).
        """
        
        return 2 * z / (self.zfar - self.znear) - (self.zfar + self.znear) / (self.zfar - self.znear)
    
    def z_ndc_to_camera(self, z: torch.Tensor) -> torch.Tensor:
        """
        Scale z values from NDC space to camera space.
        
        Args:
            z: z values in NDC space. A tensor of shape (N, H, W).
        
        Returns:
            z: z values in camera space. A tensor of shape (N, H, W).
        """
        
        return 0.5 * (z + 1) * (self.zfar - self.znear) + self.znear
    
def get_patch_ndc_to_ndc_transform(
    cameras,
    with_xyflip: bool = False,
    image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    patch_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    patch_center: Optional[Union[List, Tuple, torch.Tensor]] = None,
    ) -> Transform3d:
    """
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
    image_size = image_size.view(-1, 2)  # of shape (1 or B)x2
    image_height, image_width = image_size.unbind(1)
    
    patch_size = patch_size.view(-1, 2)  # of shape (1 or B)x2
    patch_height, patch_width = patch_size.unbind(1)
    
    patch_center = patch_center.view(-1, 2)  # of shape (1 or B)x2
    cx_patch, cy_patch = patch_center.unbind(1)

    # For non square images, we scale the points such that the aspect ratio is preserved.
    # We assume that the image is centered at the origin
    # patch ndc --> ndc
    K[:, 0, 0] = patch_width / image_width
    K[:, 1, 1] = patch_height / image_height
    K[:, 0, 3] = (2.0 * cx_patch / image_width) - 1.0
    K[:, 1, 3] = (2.0 * cy_patch / image_height) - 1.0
    K[:, 2, 2] = 1.0
    K[:, 3, 3] = 1.0

    # Transpose the projection matrix as PyTorch3D transforms use row vectors.
    transform = Transform3d(
        matrix=K.transpose(1, 2).contiguous(), device=cameras.device
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

def get_ndc_to_patch_ndc_transform(
    cameras,
    with_xyflip: bool = False,
    image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    patch_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    patch_center: Optional[Union[List, Tuple, torch.Tensor]] = None,
) -> Transform3d:
    """
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
    transform = get_patch_ndc_to_ndc_transform(
        cameras,
        with_xyflip=with_xyflip,
        image_size=image_size,
        patch_size=patch_size,
        patch_center=patch_center,
    ).inverse()
    return transform