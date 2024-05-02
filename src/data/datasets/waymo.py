# src/data/waymo.py
import torch
from mmdet3d.registry import DATASETS
from mmdet3d.datasets.waymo_dataset import WaymoDataset as MMDetWaymoDataset
from torch.utils.data import Dataset
from src.util.misc import EasyDict as edict
import cv2
import os
from src.util.cameras import PatchPerspectiveCameras as PatchCameras
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, se3_log_map, se3_exp_map
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms
from PIL import Image as Image
from PIL.Image import Resampling
import numpy as np
import logging

LABEL_NAME2ID = {
    'Car': 0, 
    'Pedestrian': 1,
    'Cyclist': 2
}

LABEL_ID2NAME = {v: k for k, v in LABEL_NAME2ID.items()}


CAM_NAMESPACE = 'CAM'
CAMERAS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
CAMERA_NAMES = [f"{CAM_NAMESPACE}_{camera}" for camera in CAMERAS]
CAM_NAME2CAM_ID = {cam_name: i for i, cam_name in enumerate(CAMERA_NAMES)}
CAM_ID2CAM_NAME = {i: cam_name for i, cam_name in enumerate(CAMERA_NAMES)}

Z_NEAR = 0.01
Z_FAR = 80.0

WAYMO_IMG_WIDTH = 1600
WAYMO_IMG_HEIGHT = 900

class WaymoBase(MMDetWaymoDataset):
    def __init__(self, data_root, label_names, patch_height=256, patch_aspect_ratio=1.,
                 is_sweep=False, **kwargs):
        self.data_root = data_root
        self.img_root = os.path.join(data_root, "samples" if not is_sweep else "sweeps")
        super().__init__(data_root=data_root, **kwargs)
        self.label_names = label_names
        self.label_ids = [LABEL_NAME2ID[label_name] for label_name in label_names]
        logging.info(f"Using label names: {self.label_names}, label ids: {self.label_ids}")
        self.patch_size = (patch_height, int(patch_height * patch_aspect_ratio)) # aspect ratio is width/height
        # define mapping from waymo label ids to our label ids depending on num of classes we predict
        self.label_id2class_id = {label : i for i, label in enumerate(self.label_ids)}  
        self.class_id2label_id = {v: k for k, v in self.label_id2class_id.items()}
        
    def __len__(self):
        self.num_samples = super().__len__()
        self.num_cameras = len(CAMERA_NAMES)
        return self.num_samples * self.num_cameras
    
    def _generate_patch(self, img_path, cam_instance):
        # load image from path using PIL
        img_pil = Image.open(img_path)
        if img_pil is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        # return croped list of images as defined by 2d bbox for each instance
        bbox = cam_instance.bbox # bounding box annotation (exterior rectangle of the projected 3D box), a list arrange as [x1, y1, x2, y2].
        center_2d = cam_instance.center_2d # Projected center location on the image, a list has shape (2,)
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
                print("bbox out of bounds of image")
                return None, None
                
            patch = img_pil.crop((x1, y1, x2, y2)) # left, upper, right, lowe
            patch_size_sq = torch.tensor(patch.size, dtype=torch.float32)
        except Exception as e:
            logging.info(f"Error in cropping image: {e}")
            # return full image if error occurs
            patch = img
        
        resized_width, resized_height = self.patch_size
        patch_resized = patch.resize((resized_width, resized_height), resample=Resampling.BILINEAR, reducing_gap=1.0)
        transform = transforms.Compose([transforms.ToTensor()])
        patch_resized_tensor = transform(patch_resized)
        return patch_resized_tensor, patch_size_sq
     
    def _get_pose_6d_lhw(self, camera, cam_instance, patch_size):
        # logging.info("_get_pose_6d_lhw")
        bbox_3d = cam_instance.bbox_3d
        
        # How to convert yaw from cam coords to patch ndc?
        x, y, z, l, h, w, yaw = bbox_3d # in camera coordinate system? need to convert to patch NDC
        roll, pitch = 0.0, 0.0 # roll and pitch are 0 for all instances in waymo dataset
        
        point_camera = (x, y, z)
        print("point_camera", point_camera)
        patch_center = cam_instance.center_2d
        
        # TODO: patch center is currently relative to upper left corner of full image. needs to be relative to the center of the full image
        # center of patch relative to center of full image so 0, 0 is center of full image and not top left corner
        full_im_center_wrt_upper_left_corner = (WAYMO_IMG_WIDTH // 2, WAYMO_IMG_HEIGHT // 2)
        patch_center_wrt_im_center = (patch_center[0] - full_im_center_wrt_upper_left_corner[0], patch_center[1] - full_im_center_wrt_upper_left_corner[1])
        print("patch_center", patch_center)
        patch_center = patch_center_wrt_im_center
        print("patch_center_wrt_im_center", patch_center_wrt_im_center)
        print("patch_size", patch_size)
        
        # logging.info("center2d", patch_center)
        if len(patch_center) == 2:
            # add batch dimension
            patch_center = torch.tensor(patch_center, dtype=torch.float32).unsqueeze(0)
       
        point_camera = torch.tensor(point_camera, dtype=torch.float32)
        # logging.info("point_camera", point_camera, point_camera.shape)
        if point_camera.dim() == 1:
            point_camera = point_camera.view(1, 1, 3)
        
        assert point_camera.dim() == 3 or point_camera.dim() == 2, f"point_camera dim is {point_camera.dim()}"
        assert isinstance(point_camera, torch.Tensor), f"point_camera is not a torch tensor"

        point_patch_ndc = camera.transform_points_patch_ndc(points=point_camera,
                                                                  patch_size=patch_size, 
                                                                    patch_center=patch_center)
        print("point_patch_ndc", point_patch_ndc, point_patch_ndc.shape)  
        z_camera = point_patch_ndc[..., 2]
        # logging.info("z_camera", z_camera, z_camera.shape) # torch.Size([1, 1, 2]
        # scale z values from patch space to NDC space
        z_ndc = camera.z_camera_to_ndc(z_camera)
        # logging.info("z_ndc", z_ndc, z_ndc.shape) # torch.Size([1, 1, 2])
        z_patch = camera.z_ndc_to_patch(z_ndc, patch_size, camera.get_image_size()) # torch.Size([1, 2])
        # logging.info("z_patch", z_patch, z_patch.shape) # torch.Size([1, 1, 2])
        
        # logging.info("point_patch_ndc[..., 2]", point_patch_ndc[..., 2], point_patch_ndc[..., 2].shape) # torch.Size([1, 1, 2]) 
        
        point_patch_ndc[..., 2] = z_patch
        
        
        if point_patch_ndc.dim() == 3:
            point_patch_ndc = point_patch_ndc.view(-1)
        x_patch, y_patch, z_patch = point_patch_ndc
        # logging.info("x_patch, y_patch, z_patch", x_patch, y_patch, z_patch)
        
        
        # TODO: Roll Pitch Yaw must be in Patch NDC, not in camera coords
        euler_angles = torch.tensor([yaw, pitch, roll], dtype=torch.float32)
        convention = "XYZ"
        
        R = euler_angles_to_matrix(euler_angles, convention) 
        
        # R_homogeneous = torch.eye(4)
        # R_homogeneous[:3, :3] = R
        # print("R", R, R.shape)
        # print("R_homogeneous", R_homogeneous, R_homogeneous.shape)
        R_patch = camera.transform_points_patch_ndc(points=R,
                                                    patch_size=patch_size,
                                                    patch_center=patch_center)
        # print("R_patch_homogenous", R_patch_homogenous, R_patch_homogenous.shape)
        # R_patch = R_patch_homogenous[:3, :3]
        # logging.info("R_patch", R_patch, R_patch.shape)
        translation = torch.tensor([x_patch, y_patch, z_patch], dtype=torch.float32)
        # logging.info("translation", translation, translation.shape)
        # A SE(3) matrix has the following form: ` [ R 0 ] [ T 1 ] , `
        se3_exp_map_matrix = torch.eye(4)
        se3_exp_map_matrix[:3, :3] = R_patch
        se3_exp_map_matrix[:3, 3] = translation
        

        # form must be ` [ R 0 ] [ T 1 ] , ` so need to transpose
        se3_exp_map_matrix = se3_exp_map_matrix.T
        
        # logging.info("se3_exp_map_matrix", se3_exp_map_matrix, se3_exp_map_matrix.shape)
        
        # add batch dim if not present
        if se3_exp_map_matrix.dim() == 2:
            se3_exp_map_matrix = se3_exp_map_matrix.unsqueeze(0)
        # logging.info("se3_exp_map_matrix", se3_exp_map_matrix, se3_exp_map_matrix.shape)
        # print("se3_exp_map_matrix", se3_exp_map_matrix, se3_exp_map_matrix.shape)
        # print trace of rotation matrix
        rotation_matrix = se3_exp_map_matrix[:, :3, :3]
        # logging.info("rotation_matrix", rotation_matrix, rotation_matrix.shape) # torch.Size([1, 3, 3])
        rot_trace = rotation_matrix.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        logging.info("rot_trace", rot_trace, rot_trace.shape)
        
        # trace must be in 
        pose_6d = se3_log_map(se3_exp_map_matrix) # 6d vector
        try: 
            pose_6d = se3_log_map(se3_exp_map_matrix) # 6d vector
        except Exception as e:
            print("Error in se3_log_map", e)
            return None, None
        # logging.info("pose_6d", pose_6d, pose_6d.shape)
        h_aspect_ratio = h / l
        w_aspect_ratio = w / l
        
        bbox_sizes = torch.tensor([l, h_aspect_ratio, w_aspect_ratio], dtype=torch.float32)
        return pose_6d, bbox_sizes
    
    
    def _get_cam_instance(self, cam_instance, img_path, patch_size, cam2img, cam2ego):
        # get a easy dict / edict with the same keys as the CamInstance class
        cam_instance = edict(cam_instance)
        
        ### must be original patch size!!
        patch, patch_size_original = self._generate_patch(img_path, cam_instance)
        # patch_size = torch.tensor(patch_size, dtype=torch.float32)
        if patch is None or patch_size_original is None:
            print("patch", patch)
            print("patch_size_original", patch_size_original)
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
    
        image_size = [(WAYMO_IMG_HEIGHT, WAYMO_IMG_WIDTH)]
        
        # cam2ego: The transformation matrix from this camera sensor to ego vehicle. (4x4 list)
        # cam2ego = torch.tensor(cam2ego, dtype=torch.float32)
        # R = torch.tensor(cam2ego[:3, :3], dtype=torch.float32)
        # T = torch.tensor(cam2ego[:3, 3], dtype=torch.float32)
        # print("R", R, R.shape)
        # print("T", T, T.shape)
        # add batch dimension if not present
        # if R.dim() == 2:
        #     R = R.unsqueeze(0)
        # if T.dim() == 1:
        #     T = T.unsqueeze(0)
        
        camera = PatchCameras(
            znear=Z_NEAR,
            zfar=Z_FAR,
            K=K,
            device="cuda" if torch.cuda.is_available() else "cpu",
            image_size=image_size)
         
        cam_instance.patch = patch
        # if no instances add 6d vec of zeroes for pose, 3 d vec of zeroes for bbox sizes and -1 for class id
        cam_instance.pose_6d, cam_instance.bbox_sizes = self._get_pose_6d_lhw(camera, cam_instance, patch_size_original)
        
        if cam_instance.pose_6d is None or cam_instance.bbox_sizes is None:
            print("pose_6d", cam_instance.pose_6d)
            print("bbox_sizes", cam_instance.bbox_sizes)
            return None
        
        cam_instance.class_id = cam_instance.bbox_label
        cam_instance.patch_size = patch_size_original
        return cam_instance
    
    def __getitem__(self, idx):
        ret = edict()
        sample_idx = idx // self.num_cameras
        cam_idx = idx % self.num_cameras
        sample_info = edict(super().__getitem__(sample_idx))
        cam_name = CAM_ID2CAM_NAME[cam_idx]
        ret.sample_idx = sample_idx
        ret.cam_idx = cam_idx
        ret.cam_name = cam_name
        sample_img_info = edict(sample_info.images[cam_name])
        ret.update(sample_img_info)
        cam_instances = sample_info.cam_instances[cam_name] # list of dicts for each instance in the current camera image
        # filter out instances that are not in the label_names
        cam_instances = [cam_instance for cam_instance in cam_instances if cam_instance['bbox_label'] in self.label_ids]
        
        if len(cam_instances) == 0:
            # iter next sample if no instances present
            # prevent querying idx outside of bounds
            if idx + 1 >= self.__len__():
                return self.__getitem__(0)
            else:
                return self.__getitem__(idx + 1)
         
        random_idx = np.random.randint(0, len(cam_instances))
        cam_instance = cam_instances[random_idx]
        
        img_file = sample_img_info.img_path.split("/")[-1]

        print(sample_img_info)
        
        ret_cam_instance = self._get_cam_instance(cam_instance=cam_instance, 
        img_path=os.path.join(self.img_root, cam_name, img_file), patch_size=self.patch_size,
         cam2img=sample_img_info.cam2img, cam2ego=sample_img_info.cam2ego)
        if ret_cam_instance is None:
            if idx + 1 >= self.__len__():
                return self.__getitem__(0)
            else:
                return self.__getitem__(idx + 1)
        
        full_im_path = os.path.join(self.img_root, cam_name, img_file)
        full_img = Image.open(full_im_path)
        transform = transforms.Compose([transforms.ToTensor()])
        full_img_tensor = transform(full_img)
        ret.full_img = full_img_tensor
        ret.patch = ret_cam_instance.patch
        ret.class_id = self.label_id2class_id[ret_cam_instance.class_id]
        pose_6d, bbox_sizes = ret_cam_instance.pose_6d, ret_cam_instance.bbox_sizes
        # set requires grad = False
        pose_6d = pose_6d.unsqueeze(0) if pose_6d.dim() == 1 else pose_6d
        pose_6d = pose_6d.detach().requires_grad_(False)
        bbox_sizes = bbox_sizes.detach().requires_grad_(False)
        
        ret.pose_6d, ret.bbox_sizes = pose_6d, bbox_sizes
        patch_size_original = ret_cam_instance.patch_size
        patch_size_original = torch.tensor(self.patch_size, dtype=torch.float32, requires_grad=False)
        patch_center_2d = torch.tensor(ret_cam_instance.center_2d, dtype=torch.float32, requires_grad=False)
        ret.patch_size = patch_size_original
        ret.patch_center_2d = patch_center_2d
        assert ret.pose_6d.dim() == 2, f"pose_6d dim is {ret.pose_6d.dim()}"
        return ret

@DATASETS.register_module()
class WaymoTrain(WaymoBase):
    def __init__(self, **kwargs):
        self.split = "train"
        self.ann_file = "waymo_infos_train.pkl"
        kwargs.update({"ann_file": self.ann_file})
        super().__init__(**kwargs)
        
@DATASETS.register_module()     
class WaymoValidation(WaymoBase):
    def __init__(self, **kwargs):
        self.split = "validation"
        self.ann_file = "waymo_infos_val.pkl"
        kwargs.update({"ann_file": self.ann_file})
        super().__init__(**kwargs)

@DATASETS.register_module()   
class WaymoTest(WaymoBase):
    def __init__(self, **kwargs):
        self.split = "test"
        ann_file = "waymo_infos_test.pkl"
        kwargs.update({"ann_file": ann_file})
        super().__init__(**kwargs)

@DATASETS.register_module()
class WaymoTrainMini(WaymoBase):
    def __init__(self, data_root=None, **kwargs):
        self.split = "train-mini"
        ann_file = "waymo_mini_infos_train.pkl"
        kwargs.update({"data_root": data_root, "ann_file": ann_file})
        super().__init__(**kwargs)
        
@DATASETS.register_module()
class WaymoValidationMini(WaymoBase):
    def __init__(self, data_root=None, **kwargs):
        self.split = "val-mini"
        ann_file = "waymo_mini_infos_val.pkl"
        kwargs.update({"data_root": data_root, "ann_file": ann_file})
        super().__init__(**kwargs)