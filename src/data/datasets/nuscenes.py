# src/data/nuscenes.py
import torch
from mmdet3d.registry import DATASETS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset as MMDetNuScenesDataset
from torch.utils.data import Dataset
from src.util.misc import EasyDict as edict
import cv2
import os
from src.util.cameras import PatchPerspectiveCameras as PatchCameras
from src.util.cameras import get_ndc_to_patch_ndc_transform, get_patch_ndc_to_ndc_transform
from src.util.cameras import z_world_to_learned
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, se3_log_map, se3_exp_map
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms
from PIL import Image as Image
from PIL.Image import Resampling
import numpy as np
import logging

L_MIN = 0.5 
L_MAX = 3.0

LABEL_NAME2ID = {
    'car': 0, 
    'truck': 1,
    'trailer': 2,
    'bus': 3,
    'construction_vehicle': 4,
    'bicycle': 5,
    'motorcycle': 6,
    'pedestrian': 7,
    'traffic_cone': 8,
    'barrier': 9
}

LABEL_ID2NAME = {v: k for k, v in LABEL_NAME2ID.items()}

CAM_NAMESPACE = 'CAM'
CAMERAS = ["FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT"]
CAMERA_NAMES = [f"{CAM_NAMESPACE}_{camera}" for camera in CAMERAS]
CAM_NAME2CAM_ID = {cam_name: i for i, cam_name in enumerate(CAMERA_NAMES)}
CAM_ID2CAM_NAME = {i: cam_name for i, cam_name in enumerate(CAMERA_NAMES)}

Z_NEAR = 0.01
Z_FAR = 55.0

NUSC_IMG_WIDTH = 1600
NUSC_IMG_HEIGHT = 900

POSE_DIM = 4

class NuScenesBase(MMDetNuScenesDataset):
    def __init__(self, data_root, label_names, patch_height=256, patch_aspect_ratio=1.,
                 is_sweep=False, **kwargs):
        self.data_root = data_root
        self.img_root = os.path.join(data_root, "samples" if not is_sweep else "sweeps")
        super().__init__(data_root=data_root, **kwargs)
        self.label_names = label_names
        self.label_ids = [LABEL_NAME2ID[label_name] for label_name in label_names]
        logging.info(f"Using label names: {self.label_names}, label ids: {self.label_ids}")
        self.patch_size = (patch_height, int(patch_height * patch_aspect_ratio)) # aspect ratio is width/height
        # define mapping from nuscenes label ids to our label ids depending on num of classes we predict
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
            floored_center = np.floor(center_2d).astype(np.int32)
            box_size = max(int(width), int(height))
            fill_factor = max(int(width), int(height)) - min(int(width), int(height)) # TODO: account for fill factor
            box_size = box_size+1 if box_size%2 == 0 else box_size 
            y1 = floored_center[1] - box_size // 2 - 1
            y2 = floored_center[1] + box_size // 2 + 1
            x1 = floored_center[0] - box_size // 2 - 1
            x2 = floored_center[0] + box_size // 2 + 1

            # TODO: Pad instead
            # TODO: check dims
            # check if x1, x2, y1, y2 are within bounds of image. if not, return None, None
            if x1 < 0 or x2 > img_pil.size[0] or y1 < 0 or y2 > img_pil.size[1]:
                return None, None, None, None
            patch = img_pil.crop((x1, y1, x2, y2)) # left, upper, right, lowe
            patch_size_sq = torch.tensor(patch.size, dtype=torch.float32)
        except Exception as e:
            logging.info(f"Error in cropping image: {e}")
            # return full image if error occurs
            patch = img
        
        resized_width, resized_height = self.patch_size
        # ratio of original image to resized image
        try:
            resampling_factor = (resized_width / patch.size[0], resized_height / patch.size[1])
            assert resampling_factor[0] == resampling_factor[1], "resampling factor of width and height must be the same but they are not."
        except ZeroDivisionError:
            logging.info("patch size is 0", patch.size)
            return None, None, None, None
        patch_resized = patch.resize((resized_width, resized_height), resample=Resampling.BILINEAR, reducing_gap=1.0)
        transform = transforms.Compose([transforms.ToTensor()])
        patch_resized_tensor = transform(patch_resized)
        fill_factor_resampled = fill_factor * resampling_factor[0]
        return patch_resized_tensor, patch_size_sq, resampling_factor, fill_factor_resampled
    
    def _get_yaw_perturbed(self, yaw, perturb_degrees=20):
        # perturb yaw by a random value between -perturb_degrees and perturb_degrees
        perturb_radians = np.radians(perturb_degrees)
        yaw_perturbed = yaw + np.random.uniform(-perturb_radians, perturb_radians)
        return yaw_perturbed

    def _get_pose_6d_perturbed(self, cam_instance):
        bbox_3d = cam_instance.bbox_3d
        x, y, z, l, h, w, yaw = bbox_3d # in camera coordinate system? need to convert to patch NDC
        yaw = self._get_yaw_perturbed(yaw)
        roll, pitch = 0.0, 0.0 # roll and pitch are 0 for all instances in nuscenes dataset
    
        euler_angles = torch.tensor([pitch, roll, yaw], dtype=torch.float32)
        convention = "XYZ"
        R = euler_angles_to_matrix(euler_angles, convention)
        # A SE(3) matrix has the following form: ` [ R 0 ] [ T 1 ] , `
        se3_exp_map_matrix = torch.eye(4)
        se3_exp_map_matrix[:3, :3] = R 

        # form must be ` [ R 0 ] [ T 1 ] , ` so need to transpose
        se3_exp_map_matrix = se3_exp_map_matrix.T
        
        # add batch dim if not present
        if se3_exp_map_matrix.dim() == 2:
            se3_exp_map_matrix = se3_exp_map_matrix.unsqueeze(0)
        
        try: 
            pose_6d = se3_log_map(se3_exp_map_matrix) # 6d vector
        except Exception as e:
            logging.info("Error in se3_log_map", e)
            return None, None
          
       
        yaw_perturbed = yaw
        return pose_6d, yaw_perturbed
        
    def _get_pose_6d_lhw(self, camera, cam_instance, patch_size, patch_resampling_factor, fill_factor_resampled):
        bbox_3d = cam_instance.bbox_3d
        x, y, z, l, h, w, yaw = bbox_3d # in camera coordinate system? need to convert to patch NDC
        roll, pitch = 0.0, 0.0 # roll and pitch are 0 for all instances in nuscenes dataset
        
        object_centroid_3D = (x, y, z)
        patch_center = cam_instance.center_2d
        bbox_2d = cam_instance.bbox
        width = cam_instance.bbox[2] - cam_instance.bbox[0]
        height = cam_instance.bbox[3] - cam_instance.bbox[1]
        box_size = max(int(width), int(height))
        box_size = box_size+1 if box_size%2 == 0 else box_size
        if len(patch_center) == 2:
            # add batch dimension
            patch_center = torch.tensor(patch_center, dtype=torch.float32).unsqueeze(0)
       
        object_centroid_3D = torch.tensor(object_centroid_3D, dtype=torch.float32)
        if object_centroid_3D.dim() == 1:
            object_centroid_3D = object_centroid_3D.view(1, 1, 3)
        
        assert object_centroid_3D.dim() == 3 or object_centroid_3D.dim() == 2, f"object_centroid_3D dim is {object_centroid_3D.dim()}"
        assert isinstance(object_centroid_3D, torch.Tensor), f"object_centroid_3D is not a torch tensor"

        point_patch_ndc = camera.transform_points_patch_ndc(points=object_centroid_3D,
                                                            patch_size=patch_size,
                                                            patch_center=patch_center)
        # TODO: scale z values from camera to learned
        z_world = z
        
        def get_zminmax(min_val, max_val, focal_length, patch_height):
            # TODO: need to account for fill factor
            zmin = -(min_val * focal_length.squeeze()[0]) / (patch_height - fill_factor_resampled)
            zmax = -(max_val * focal_length.squeeze()[0]) / (patch_height - fill_factor_resampled)
            return zmin, zmax
        
        zmin, zmax = get_zminmax(min_val=L_MIN, max_val=L_MAX, 
                                 focal_length=camera.focal_length, 
                                 patch_height=self.patch_size[0])
        
        z_learned_unscaled = z_world_to_learned(z_world=z_world, zmin=zmin, zmax=zmax, 
                                       patch_resampling_factor=patch_resampling_factor[0])
        
        # z_learned_far = z_world_to_learned(z_world=Z_FAR, zmin=zmin, zmax=zmax, 
        #                                patch_resampling_factor=patch_resampling_factor[0])
        # z_learned_near = z_world_to_learned(z_world=Z_NEAR, zmin=zmin, zmax=zmax, 
        #                                patch_resampling_factor=patch_resampling_factor[0])
        
        # def rescale_z_learned(val, min_val, max_val):
        #     rescaled_val = 2*((val - min_val) / (max_val - min_val)) - 1
        #     return rescaled_val
        
        # z_learned = rescale_z_learned(z_learned_unscaled, z_learned_near, z_learned_far)
        
        z_learned = z_learned_unscaled
        
        if point_patch_ndc.dim() == 3:
            point_patch_ndc = point_patch_ndc.view(-1)
        x_patch, y_patch, _ = point_patch_ndc
        
        euler_angles = torch.tensor([pitch, roll, yaw], dtype=torch.float32)
        convention = "XYZ"
        R = euler_angles_to_matrix(euler_angles, convention)
        
        translation = torch.tensor([x_patch, y_patch, z_learned], dtype=torch.float32)
        # A SE(3) matrix has the following form: ` [ R 0 ] [ T 1 ] , `
        se3_exp_map_matrix = torch.eye(4)
        se3_exp_map_matrix[:3, :3] = R 
        se3_exp_map_matrix[:3, 3] = translation

        # form must be ` [ R 0 ] [ T 1 ] , ` so need to transpose
        se3_exp_map_matrix = se3_exp_map_matrix.T
        
        # add batch dim if not present
        if se3_exp_map_matrix.dim() == 2:
            se3_exp_map_matrix = se3_exp_map_matrix.unsqueeze(0)
        
        try: 
            pose_6d = se3_log_map(se3_exp_map_matrix) # 6d vector
        except Exception as e:
            logging.info("Error in se3_log_map", e)
            return None, None
        
        # l and w pred as aspect ratio wrt h
        l_pred = l / h
        h_pred = h
        w_pred = w / h
        
        bbox_sizes = torch.tensor([l_pred, h_pred, w_pred], dtype=torch.float32)
        
        # remove v1, v2 from pose_6d = t1, t2, t3, v1, v2, v3
        pose_6d_no_v1v2 = torch.zeros_like(pose_6d[:, :POSE_DIM])
        pose_6d_no_v1v2[:, :3] = pose_6d[:, :POSE_DIM-1]
        # set v3 as last val
        pose_6d_no_v1v2[:, -1] = pose_6d[:, -1]
        return pose_6d_no_v1v2, bbox_sizes, yaw
    
    
    def _get_cam_instance(self, cam_instance, img_path, patch_size, cam2img):
        # get a easy dict / edict with the same keys as the CamInstance class
        cam_instance = edict(cam_instance)
        
        ### must be original patch size!!
        patch, patch_size_original, resampling_factor, fill_factor = self._generate_patch(img_path, cam_instance)
        # patch_size = torch.tensor(patch_size, dtype=torch.float32)
        if patch is None or patch_size_original is None:
            return None
        
        
        # if no batch dimension, add it
        if patch_size_original.dim() == 1:
            patch_size_original = patch_size_original.unsqueeze(0)
            
        cam2img = torch.tensor(cam2img, dtype=torch.float32)
        # make 4x4 matrix from 3x3 camera matrix
        K = torch.zeros(4, 4, dtype=torch.float32)
        
        K[:3, :3] = cam2img
        K_orig = K.clone().detach().requires_grad_(False)
        K[2, 2] = 0.0
        K[2, 3] = 1.0
        K[3, 2] = 1.0
        K = K.clone().detach().requires_grad_(False).unsqueeze(0) # add batch dimension
        
        image_size = [(NUSC_IMG_HEIGHT, NUSC_IMG_WIDTH)]
            
        focal_length = K[..., 0, 0] # shape (1,)
        principal_point = K[..., :2, 2] # shape (1, 2)
        
        # negate focal length
        focal_length = -focal_length
        
        camera = PatchCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            znear=Z_NEAR,
            zfar=Z_FAR,
            device="cuda" if torch.cuda.is_available() else "cpu",
            image_size=image_size)
        
        # projection_transform = camera.get_projection_transform()
        
        bbox_3d = cam_instance.bbox_3d
        x, y, z, l, h, w, yaw = bbox_3d
        
        center_2d = torch.ones(3, dtype=torch.float32)
        center_2d[:2] = torch.tensor(cam_instance.center_2d, dtype=torch.float32)
        center_2d = center_2d.unsqueeze(0)        
        point_screen = center_2d
        
        screen2ndc_transform = camera.get_ndc_camera_transform()
        point_screen = point_screen.to(device="cpu")
        screen2ndc_transform = screen2ndc_transform.to(device="cpu")
        
        point_ndc = screen2ndc_transform.transform_points(point_screen)
        
        ndc2patch_transform = get_ndc_to_patch_ndc_transform(camera, 
                                                             with_xyflip=False, 
                                                             image_size=image_size,
                                                             patch_size=patch_size_original, 
                                                             patch_center=center_2d[..., :2])
        
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ndc2patch_transform.to(device=device)
        
        point_ndc = point_ndc.to(device=device)
        point_patch_ndc = ndc2patch_transform.transform_points(point_ndc)
        
        cam_instance.patch = patch
        # if no instances add 6d vec of zeroes for pose, 3 d vec of zeroes for bbox sizes and -1 for class id
        
        cam_instance.pose_6d, cam_instance.bbox_sizes, cam_instance.yaw = self._get_pose_6d_lhw(camera, cam_instance, patch_size_original, resampling_factor, fill_factor)
        cam_instance.pose_6d_perturbed, cam_instance.yaw_perturbed = self._get_pose_6d_perturbed(cam_instance)
        if cam_instance.pose_6d is None or cam_instance.bbox_sizes is None:
            return None
        
        cam_instance.class_id = cam_instance.bbox_label
        cam_instance.patch_size = patch_size_original
        cam_instance.resampling_factor = resampling_factor
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
        
        ret_cam_instance = self._get_cam_instance(cam_instance, img_path=os.path.join(self.img_root, cam_name, img_file), patch_size=self.patch_size, cam2img=sample_img_info.cam2img)
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        patch_size_original = patch_size_original.to(device=device)
        patch_size_original = torch.tensor(patch_size_original, dtype=torch.float32, requires_grad=False)
        patch_center_2d = torch.tensor(ret_cam_instance.center_2d, dtype=torch.float32, requires_grad=False)
        ret.patch_size = patch_size_original
        ret.patch_center_2d = patch_center_2d
        ret.bbox_3d_gt = ret_cam_instance.bbox_3d
        ret.resampling_factor = ret_cam_instance.resampling_factor # ratio of resized image to original patch size
        ret.pose_6d_perturbed = ret_cam_instance.pose_6d_perturbed
        if ret.pose_6d.dim() == 3:
            ret.pose_6d = ret.pose_6d.squeeze(0)
        assert ret.pose_6d.dim() == 2, f"pose_6d dim is {ret.pose_6d.dim()}"
        ret.pose_6d = ret.pose_6d.squeeze(0)
        ret.yaw = ret_cam_instance.yaw
        ret.yaw_perturbed = ret_cam_instance.yaw_perturbed
        return ret

@DATASETS.register_module()
class NuScenesTrain(NuScenesBase):
    def __init__(self, **kwargs):
        self.split = "train"
        self.ann_file = "nuscenes_infos_train.pkl"
        kwargs.update({"ann_file": self.ann_file})
        super().__init__(**kwargs)
        
@DATASETS.register_module()     
class NuScenesValidation(NuScenesBase):
    def __init__(self, **kwargs):
        self.split = "validation"
        self.ann_file = "nuscenes_infos_val.pkl"
        kwargs.update({"ann_file": self.ann_file})
        super().__init__(**kwargs)

@DATASETS.register_module()   
class NuScenesTest(NuScenesBase):
    def __init__(self, **kwargs):
        self.split = "test"
        ann_file = "nuscenes_infos_test.pkl"
        kwargs.update({"ann_file": ann_file})
        super().__init__(**kwargs)

@DATASETS.register_module()
class NuScenesTrainMini(NuScenesBase):
    def __init__(self, data_root=None, **kwargs):
        self.split = "train-mini"
        ann_file = "nuscenes_mini_infos_train.pkl"
        kwargs.update({"data_root": data_root, "ann_file": ann_file})
        super().__init__(**kwargs)
        
@DATASETS.register_module()
class NuScenesValidationMini(NuScenesBase):
    def __init__(self, data_root=None, **kwargs):
        self.split = "val-mini"
        ann_file = "nuscenes_mini_infos_val.pkl"
        kwargs.update({"data_root": data_root, "ann_file": ann_file})
        super().__init__(**kwargs)
