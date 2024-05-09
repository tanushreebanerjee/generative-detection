# src/data/preprocessing/nusc_utils.py
from PIL import Image as Image
from PIL.Image import Resampling
from src.util.cameras import PatchPerspectiveCameras as PatchCameras
import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, se3_log_map, se3_exp_map


def generate_patch(self, img_path, cam_instance):
    # load image from path using PIL
    img_pil = Image.open(img_path)
    if img_pil is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
        
    # return croped list of images as defined by 2d bbox for each instance
    bbox = cam_instance.bbox # bounding box annotation (exterior rectangle of the projected 3D box), a list arrange as [x1, y1, x2, y2].
    center_2d = cam_instance.center_2d 
    # use interpolation torch to get crop of image from bbox of size patch_size
    # need to use torch for interpolation, not cv2
    x1, y1, x2, y2 = bbox
    try:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # make patch a square by taking max of width and height, centered at center_2d
        width = x2 - x1
        height = y2 - y1
        
        if width > height:
            y1 = center_2d[1] - width // 2
            y2 = center_2d[1] + width // 2
        else:
            x1 = center_2d[0] - height // 2
            x2 = center_2d[0] + height // 2
            
        # check if x1, x2, y1, y2 are within bounds of image. if not, return None, None
        if x1 < 0 or x2 > img_pil.size[0] or y1 < 0 or y2 > img_pil.size[1]:
            return None, None
            
        patch = img_pil.crop((x1, y1, x2, y2))
        patch_size_sq = torch.tensor(patch.size, dtype=torch.float32)
    except Exception as e:
        logging.info(f"Error in cropping image: {e}")
        # return full image if error occurs
        patch = img
    
    resized_width, resized_height = self.patch_size
    patch_resized = patch.resize((resized_width, resized_height), resample=Resampling.BILINEAR, reducing_gap=1.0)
    patch_resized_tensor = torch.tensor(np.array(patch_resized), dtype=torch.float32)
    return patch_resized_tensor, patch_size_sq
    
def get_pose_6d_lhw(self, camera, cam_instance, patch_size):
    bbox_3d = cam_instance.bbox_3d
    
    # How to convert yaw from cam coords to patch ndc?
    x, y, z, l, h, w, yaw = bbox_3d # in camera coordinate system? need to convert to patch NDC
    
    point_camera = (x, y, z)
    points = torch.tensor([point_camera], dtype=torch.float32)
    patch_center = cam_instance.center_2d
    if len(patch_center) == 2:
        # add batch dimension
        patch_center = torch.tensor(patch_center, dtype=torch.float32).unsqueeze(0)
    
    point_camera = torch.tensor(point_camera, dtype=torch.float32)
    
    if point_camera.dim() == 1:
        point_camera = point_camera.view(1, 1, 3)
    
    assert point_camera.dim() == 3 or point_camera.dim() == 2, f"point_camera dim is {point_camera.dim()}"
    assert isinstance(point_camera, torch.Tensor), f"point_camera is not a torch tensor"
    
    point_patch_ndc = camera.transform_points_patch_ndc(points=point_camera,
                                                                patch_size=patch_size, 
                                                                patch_center=patch_center)

    if point_patch_ndc.dim() == 3:
        point_patch_ndc = point_patch_ndc.view(-1)
    x_patch, y_patch, z_patch = point_patch_ndc
    
    roll, pitch = 0.0, 0.0 # roll and pitch are 0 for all instances in nuscenes dataset
    
    euler_angles = torch.tensor([yaw, pitch, roll], dtype=torch.float32)
    convention = "XYZ"
    
    R = euler_angles_to_matrix(euler_angles, convention)
    
    translation = torch.tensor([x_patch, y_patch, z_patch], dtype=torch.float32)
    
    # A SE(3) matrix has the following form: ` [ R 0 ] [ T 1 ] , `
    se3_exp_map_matrix = torch.eye(4)
    se3_exp_map_matrix[:3, :3] = R
    se3_exp_map_matrix[:3, 3] = translation

    # form must be ` [ R 0 ] [ T 1 ] , ` so need to transpose
    se3_exp_map_matrix = se3_exp_map_matrix.T
    
    # addd batch dim if not present
    if se3_exp_map_matrix.dim() == 2:
        se3_exp_map_matrix = se3_exp_map_matrix.unsqueeze(0)
    pose_6d = se3_log_map(se3_exp_map_matrix) # 6d vector
    
    h_aspect_ratio = h / l
    w_aspect_ratio = w / l
    
    bbox_sizes = torch.tensor([l, h_aspect_ratio, w_aspect_ratio], dtype=torch.float32)
    return pose_6d, bbox_sizes

def _get_cam_instance(self, cam_instance, img_path, patch_size, cam2img):
        # get a easy dict / edict with the same keys as the CamInstance class
        cam_instance = edict(cam_instance)
        
        ### must be original patch size!!
        patch, patch_size_original = self._generate_patch(img_path, cam_instance)
        # patch_size = torch.tensor(patch_size, dtype=torch.float32)
        if patch is None or patch_size_original is None:
            return None
        
        
        # if no batch dimension, add it
        if patch_size_original.dim() == 1:
            patch_size_original = patch_size_original.unsqueeze(0)
            
        cam2img = torch.tensor(cam2img, dtype=torch.float32)
        
        # make 4x4 matrix from 3x3 camera matrix
        K = torch.eye(4, dtype=torch.float32)
        K[:3, :3] = cam2img
        K = K.clone().detach().requires_grad_(True).unsqueeze(0) # add batch dimension
        # torch.tensor(K, dtype=torch.float32).unsqueeze(0) # add batch dimension
    
        image_size = [(NUSC_IMG_HEIGHT, NUSC_IMG_WIDTH)]
        
        camera = PatchCameras(
            znear=Z_NEAR,
            zfar=Z_FAR,
            K=K,
            device="cuda" if torch.cuda.is_available() else "cpu",
            image_size=image_size)
        
        # normalize patch to [0, 1]
        patch = patch / 255.0
        # scale to [-1, 1]
        patch = patch * 2 - 1
        
        
        cam_instance.patch = patch
        # if no instances add 6d vec of zeroes for pose, 3 d vec of zeroes for bbox sizes and -1 for class id
        cam_instance.pose_6d, cam_instance.bbox_sizes = self._get_pose_6d_lhw(camera, cam_instance, patch_size_original)
        cam_instance.class_id = cam_instance.bbox_label
        return cam_instance