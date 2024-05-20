# src/data/nuscenes.py
import torch
from mmdet3d.registry import DATASETS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset as MMDetNuScenesDataset
from torch.utils.data import Dataset
from src.util.misc import EasyDict as edict
import os
from src.util.cameras import PatchPerspectiveCameras as PatchCameras
from src.util.cameras import get_ndc_to_patch_ndc_transform, get_patch_ndc_to_ndc_transform
from src.util.cameras import z_world_to_learned
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, se3_log_map, se3_exp_map
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms
import numpy as np
import logging
import math
import torchvision.ops as ops
import random
import pickle as pkl

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
    'barrier': 9,
    'background': 10
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
LHW_DIM = 3
BBOX_3D_DIM = 7

PATCH_SIZES = [50, 100, 200, 400]

class NuScenesBase(MMDetNuScenesDataset):
    def __init__(self, data_root, label_names, patch_height=256, patch_aspect_ratio=1.,
                 is_sweep=False, perturb_center=False, perturb_scale=False, 
                 negative_sample_prob=0.5, h_minmax_dir = "dataset_stats/combined", **kwargs):
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
        self.perturb_center = perturb_center
        self.perturb_scale = perturb_scale
        hmin_path = os.path.join(h_minmax_dir, "hmin.pkl")
        hmax_path = os.path.join(h_minmax_dir, "hmax.pkl")
        with open(hmin_path, "rb") as f:
            self.hmin_dict = pkl.load(f)
        with open(hmax_path, "rb") as f:
            self.hmax_dict = pkl.load(f)
        
        if "background" in self.label_names:
            self.negative_sample_prob = negative_sample_prob
        else:
            self.negative_sample_prob = 0.0
        
    def __len__(self):
        self.num_samples = super().__len__()
        self.num_cameras = len(CAMERA_NAMES)
        return self.num_samples * self.num_cameras
    
    def _generate_patch(self, img_path, cam_instance):
        # load image from path using PIL
        # img_pil = Image.open(img_path)
        # if img_pil is None:
        #     raise FileNotFoundError(f"Image not found at {img_path}")

        # return croped list of images as defined by 2d bbox for each instance
        bbox = cam_instance.bbox # bounding box annotation (exterior rectangle of the projected 3D box), a list arrange as [x1, y1, x2, y2].
        center_2d = cam_instance.center_2d # Projected center location on the image, a list has shape (2,)
        
        # if center_2d is out bounds, return None, None, None, None since < 50% of the object is visible
        img_pil_size = (NUSC_IMG_WIDTH, NUSC_IMG_HEIGHT)
        if center_2d[0] < 0 or center_2d[1] < 0 or center_2d[0] >= img_pil_size[0] or center_2d[1] >= img_pil_size[1]:
            return None, None, None, None, None
        
        x1, y1, x2, y2 = bbox # actual object bbox
        is_corner_case = False
        
        try:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # make patch a square by taking max of width and height, centered at center_2d
            width = x2 - x1
            height = y2 - y1
            floored_center = np.floor(center_2d).astype(np.int32)
            box_size = max(int(width), int(height))
            
            if x1 >= img_pil_size[0] or y1 >= img_pil_size[1] or x2 <= 0 or y2 <= 0:
                is_corner_case = True
                # get intersection of bbox with image
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_pil_size[0], x2)
                y2 = min(img_pil_size[1], y2)
                
                width = x2 - x1
                height = y2 - y1
                # get max
                max_dim = max(width, height)
                
                # choose center of patch as min diff from PATCH_SIZES
                dim_diffs = [abs(max_dim - patch_size) for patch_size in PATCH_SIZES]
                patch_size = PATCH_SIZES[dim_diffs.index(min(dim_diffs))]

                x1 = x1 + (width - patch_size) // 2 
                y1 = y1 + (height - patch_size) // 2
                x2 = x1 + patch_size
                y2 = y1 + patch_size
                center_2d = (x1 + patch_size // 2, y1 + patch_size // 2)
            
            if self.perturb_scale and not is_corner_case:
                # make box_size one of PATCH_SIZES centered at center_2d 
                bbox_size_diffs = [abs(box_size - patch_size) for patch_size in PATCH_SIZES]
                box_size = PATCH_SIZES[bbox_size_diffs.index(min(bbox_size_diffs))]
                # make sure scaled box is within image, and still a square
                if floored_center[0] - box_size // 2 < 0:
                    floored_center[0] = box_size // 2
                if floored_center[1] - box_size // 2 < 0:
                    floored_center[1] = box_size // 2
                if floored_center[0] + box_size // 2 > img_pil_size[0]:
                    floored_center[0] = img_pil_size[0] - box_size // 2
                if floored_center[1] + box_size // 2 > img_pil_size[1]:
                    floored_center[1] = img_pil_size[1] - box_size // 2
                    
            if int(width) > int(height):
                padding_pixels = int(width) - int(height) 
            else:
                padding_pixels = 0
                
            y1 = floored_center[1] - box_size // 2 
            y2 = floored_center[1] + box_size // 2 
            x1 = floored_center[0] - box_size // 2 
            x2 = floored_center[0] + box_size // 2 
              
            # patch = img_pil.crop((x1, y1, x2, y2)) # left, upper, right, lowe
            patch_size = (box_size, box_size)
            patch_size_sq = torch.tensor(patch_size, dtype=torch.float32)
        except Exception as e:
            logging.info(f"Error in cropping image: {e}")
            return None, None, None, None, None
        
        resized_width, resized_height = self.patch_size
        # ratio of original image to resized image
        try:
            resampling_factor = (resized_width / patch_size[0], resized_height / patch_size[1])
            assert resampling_factor[0] == resampling_factor[1], "resampling factor of width and height must be the same but they are not."
        except ZeroDivisionError:
            logging.info("patch size is 0", patch_size)
            return None, None, None, None, None
        # patch_resized = patch.resize((resized_width, resized_height), resample=Resampling.BILINEAR, reducing_gap=1.0)
        # create a boolean mask for patch with gt 2d bbox as coordinates (x1, y1, x2, y2)
        mask_bool = np.zeros((patch_size[1], patch_size[0]), dtype=bool)
        # Set the area inside the bounding box to True
        # get original 2d bbox coords
        x1_full, y1_full, x2_full, y2_full = cam_instance.bbox
        x1_patch = int(x1_full - x1)
        y1_patch = int(y1_full - y1)
        x2_patch = int(x2_full - x1)
        y2_patch = int(y2_full - y1)
        
        mask_bool[y1_patch:y2_patch, x1_patch:x2_patch] = True
        # mask_pil = Image.fromarray(mask_bool)
        # mask_resized = mask_pil.resize((resized_width, resized_height), resample=Resampling.NEAREST, reducing_gap=1.0)
        # transform = transforms.Compose([transforms.ToTensor()])
        # patch_resized_tensor = transform(patch_resized)
        # mask_resized = transform(mask_resized)
        # dummy tensor of size resized_width x resized_height
        num_channels = 3
        patch_resized_tensor = torch.zeros(num_channels, resized_height, resized_width, dtype=torch.float32)
        mask_resized = torch.zeros(1, resized_height, resized_width, dtype=torch.float32)
        padding_pixels_resampled = padding_pixels * resampling_factor[0]
        return patch_resized_tensor, patch_size_sq, resampling_factor, padding_pixels_resampled, mask_resized
    
    def _get_yaw_perturbed(self, yaw, perturb_degrees_min=30, perturb_degrees_max=90):
        # perturb yaw by a random value between -perturb_degrees and perturb_degrees
        
        perturb_degrees = np.random.uniform(perturb_degrees_min, perturb_degrees_max)
        perturb_radians = np.radians(perturb_degrees)
        
        # add or subtract with probability 0.5
        yaw_perturbed = yaw + perturb_radians if np.random.rand() > 0.5 else yaw - perturb_radians
        
        # make sure yaw_perturbed is in -pi, pi, and if not, find equivalent
        if yaw_perturbed < -math.pi:
            yaw_perturbed = yaw_perturbed + (2 * math.pi)
        elif yaw_perturbed > math.pi:
            yaw_perturbed = yaw_perturbed - (2 * math.pi)
        assert yaw_perturbed >= -math.pi and yaw_perturbed <= math.pi, f"yaw_perturbed is not in range -pi, pi: {yaw_perturbed}"
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
        v3_pert = pose_6d[:, -1]
        return v3_pert, yaw_perturbed
        
    def _get_pose_6d_lhw(self, camera, cam_instance, patch_size, patch_resampling_factor, fill_factor):
        
        # fill_factor = padding_pixels_resampled / self.patch_size[0]
        padding_pixels_resampled = fill_factor * self.patch_size[0]
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
                                                            patch_size=patch_size, # add delta
                                                            patch_center=patch_center) # add scale
        # TODO: scale z values from camera to learned
        z_world = z
        
        def get_zminmax(min_val, max_val, focal_length, patch_height):
            # TODO: need to account for fill factor
            zmin = -(min_val * focal_length.squeeze()[0]) / (patch_height - padding_pixels_resampled)
            zmax = -(max_val * focal_length.squeeze()[0]) / (patch_height - padding_pixels_resampled)
            return zmin, zmax
        bbox_label_idx = cam_instance.bbox_label
        bbox_label_name = LABEL_ID2NAME[bbox_label_idx]
        assert bbox_label_name != "background", "cannot get zminmax for background class"
        min_val = self.hmin_dict[bbox_label_name]
        max_val = self.hmax_dict[bbox_label_name]
        
        zmin, zmax = get_zminmax(min_val=min_val, max_val=max_val, 
                                 focal_length=camera.focal_length, 
                                 patch_height=self.patch_size[0])
        
        z_learned = z_world_to_learned(z_world=z_world, zmin=zmin, zmax=zmax, 
                                       patch_resampling_factor=patch_resampling_factor[0])
        
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
    
    def get_perturbed_patch(self, center_2d, bbox):
        # given: original center_2d and bbox
        x1, y1, x2, y2 = bbox
        center_x, center_y = center_2d

        # Calculate width and height of the bounding box
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Calculate the maximum perturbation distance
        max_perturb = 0.5 * min(bbox_width, bbox_height)
        
        # Random perturbation in x direction within the max perturbation limit
        x_perturb = np.random.uniform(-max_perturb, max_perturb)
        
        # Calculate corresponding y perturbation to maintain visibility condition
        max_y_perturb = np.sqrt(max_perturb**2 - x_perturb**2)
        y_perturb = np.random.uniform(-max_y_perturb, max_y_perturb)
        
        # Perturbed center coordinates
        perturbed_center_x = int(center_x + x_perturb)
        perturbed_center_y = int(center_y + y_perturb)
        
        return [perturbed_center_x, perturbed_center_y]
    
    def _get_cam_instance(self, cam_instance, img_path, patch_size, cam2img):
        cam_instance = edict(cam_instance)
        
        if self.perturb_center:
            # random perturb center
            center_o = cam_instance.center_2d
            bbox_o = cam_instance.bbox
            center_perturbed = self.get_perturbed_patch(center_o, bbox_o)
            cam_instance.center_2d = center_perturbed
        
        patch, patch_size_original, resampling_factor, padding_pixels_resampled, mask_resampled = self._generate_patch(img_path, cam_instance)
        cam_instance.mask_2d_bbox = mask_resampled
        # patch_size = torch.tensor(patch_size, dtype=torch.float32)
        if patch is None or patch_size_original is None:
            return None
        
        fill_factor = padding_pixels_resampled / self.patch_size[0]
        
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
            device="cpu",#"cuda" if torch.cuda.is_available() else "cpu",
            image_size=image_size)
        
        bbox_3d = cam_instance.bbox_3d
        x, y, z, l, h, w, yaw = bbox_3d
        
        center_2d = torch.ones(3, dtype=torch.float32)
        center_2d[:2] = torch.tensor(cam_instance.center_2d, dtype=torch.float32)
        center_2d = center_2d.unsqueeze(0)        
        point_screen = center_2d
        
        screen2ndc_transform = camera.get_ndc_camera_transform()
        point_screen = point_screen#.to(device="cpu")
        screen2ndc_transform = screen2ndc_transform#.to(device="cpu")
        
        point_ndc = screen2ndc_transform.transform_points(point_screen)
        
        ndc2patch_transform = get_ndc_to_patch_ndc_transform(camera, 
                                                             with_xyflip=False, 
                                                             image_size=image_size,
                                                             patch_size=patch_size_original, 
                                                             patch_center=center_2d[..., :2])
        
        
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        ndc2patch_transform#.to(device=device)
        
        point_ndc = point_ndc#.to(device=device)
        point_patch_ndc = ndc2patch_transform.transform_points(point_ndc)
        
        cam_instance.patch = patch
        # if no instances add 6d vec of zeroes for pose, 3 d vec of zeroes for bbox sizes and -1 for class id
        
        cam_instance.pose_6d, cam_instance.bbox_sizes, cam_instance.yaw = self._get_pose_6d_lhw(camera, cam_instance, patch_size_original, resampling_factor, fill_factor)
        
        cam_instance.v3_pert, cam_instance.yaw_perturbed = self._get_pose_6d_perturbed(cam_instance)
        pose_pert = cam_instance.pose_6d.clone()
        pose_pert[:, -1] = cam_instance.v3_pert
        cam_instance.pose_6d_perturbed = pose_pert
        if cam_instance.pose_6d is None or cam_instance.bbox_sizes is None:
            return None
        
        cam_instance.class_id = cam_instance.bbox_label
        cam_instance.patch_size = patch_size_original
        cam_instance.resampling_factor = resampling_factor
        cam_instance.fill_factor = fill_factor
        return cam_instance
    
    def __getitem__(self, idx):
        ret = edict()
        
        sample_idx = idx // self.num_cameras
        cam_idx = idx % self.num_cameras
        sample_info = super().__getitem__(sample_idx)
        # sample_info = edict(super().__getitem__(sample_idx))
        cam_name = CAM_ID2CAM_NAME[cam_idx]
        ret.sample_idx = sample_idx
        ret.cam_idx = cam_idx
        ret.cam_name = cam_name
        sample_img_info = edict(sample_info['images'][cam_name])
        ret.update(sample_img_info)
        # cam_instances = sample_info.cam_instances[cam_name] # list of dicts for each instance in the current camera image
        cam_instances = sample_info['cam_instances'][cam_name] # list of dicts for each instance in the current camera image
        # filter out instances that are not in the label_names
        cam_instances = [cam_instance for cam_instance in cam_instances if cam_instance['bbox_label'] in self.label_ids]
        
        # if np.random.rand() <= (1. - self.negative_sample_prob): # get random crop of an instance with 50% overlap
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
        
        ret.patch = ret_cam_instance.patch
        ret.class_id = self.label_id2class_id[ret_cam_instance.class_id]
        ret.original_class_id = ret_cam_instance.class_id
        ret.class_name = LABEL_ID2NAME[ret.original_class_id]
        pose_6d, bbox_sizes = ret_cam_instance.pose_6d, ret_cam_instance.bbox_sizes
        # set requires grad = False
        pose_6d = pose_6d.unsqueeze(0) if pose_6d.dim() == 1 else pose_6d
        pose_6d = pose_6d.detach().requires_grad_(False)
        bbox_sizes = bbox_sizes.detach().requires_grad_(False)
        
        ret.pose_6d, ret.bbox_sizes = pose_6d, bbox_sizes
        patch_size_original = ret_cam_instance.patch_size
        patch_center_2d = torch.tensor(ret_cam_instance.center_2d).float()
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
        ret.fill_factor = ret_cam_instance.fill_factor
        mask_2d_bbox = ret_cam_instance.mask_2d_bbox
        if mask_2d_bbox.dim() == 2:
            mask_2d_bbox = mask_2d_bbox.unsqueeze(0)
        ret.mask_2d_bbox = mask_2d_bbox
    
        # else:  # get random crop without overlap
        #     # bbox = [x1, y1, x2, y2]
        #     bbox_2d_list = [cam_instance['bbox'] for cam_instance in cam_instances]
        #     img_file = sample_img_info.img_path.split("/")[-1]
        #     # get random crop from image, and make sure it does not overlap with any bbox
        #     img_path = os.path.join(self.img_root, cam_name, img_file)
        #     img_pil = Image.open(img_path)
        #     if img_pil is None:
        #         raise FileNotFoundError(f"Image not found at {img_path}")
            
        #     background_patch = self.get_random_crop_without_overlap(img_pil, bbox_2d_list, PATCH_SIZES)
        #     if background_patch is None:
        #         if idx + 1 >= self.__len__():
        #             return self.__getitem__(0)
        #         else:
        #             return self.__getitem__(idx + 1)
                
        #     background_patch_original_size = background_patch.size
        #     background_patch = background_patch.resize(self.patch_size, resample=Resampling.BILINEAR)
        #     transform = transforms.Compose([transforms.ToTensor()])
        #     background_patch_tensor = transform(background_patch)
        #     ret.patch = background_patch_tensor
        #     ret.class_id = self.label_id2class_id[LABEL_NAME2ID['background']]
        #     ret.original_class_id = LABEL_NAME2ID['background']
        #     ret.class_name = LABEL_ID2NAME[LABEL_NAME2ID['background']]
        #     ret.pose_6d = torch.zeros(POSE_DIM, dtype=torch.float32).squeeze(0)
        #     ret.bbox_sizes = torch.zeros(LHW_DIM, dtype=torch.float32)
        #     ret.patch_size = torch.tensor(self.patch_size, dtype=torch.float32).unsqueeze(0) # torch.Size([1, 2])
        #     ret.patch_center_2d = torch.tensor([self.patch_size[0] // 2, self.patch_size[1] // 2], dtype=torch.float32)
        #     ret.bbox_3d_gt = torch.zeros(BBOX_3D_DIM, dtype=torch.float32)
        #     ret.resampling_factor = (self.patch_size[0] / background_patch_original_size[0], self.patch_size[1] / background_patch_original_size[1])
        #     ret.pose_6d_perturbed = torch.zeros(POSE_DIM, dtype=torch.float32).unsqueeze(0)
        #     ret.yaw = 0.0
        #     ret.yaw_perturbed = 0.0
        #     ret.fill_factor = 0.0
        #     mask_2d_bbox = torch.zeros(self.patch_size[0], self.patch_size[1], dtype=torch.float32)
        #     if mask_2d_bbox.dim() == 2:
        #         mask_2d_bbox = mask_2d_bbox.unsqueeze(0)
        #     ret.mask_2d_bbox = mask_2d_bbox
        
        for key in ["cam2img", "cam2ego", "lidar2cam", "bbox_3d_gt"]:
            value = ret[key]
            if isinstance(value, list):
                ret[key] = torch.tensor(value, dtype=torch.float32)
         
        return ret
    
    # def get_random_crop_without_overlap(self, img_pil, bbox_2d_list, patch_sizes):
    #     width, height = img_pil.size
    #     bbox_tensor = torch.tensor(bbox_2d_list, dtype=torch.float)
    #     is_found = False
    #     timeout_iters = 10
    #     iters = 0
    #     while not is_found and iters < timeout_iters:
    #         # Randomly choose a patch size
    #         patch_size = random.choice(patch_sizes)
    #         crop_width = crop_height = patch_size
            
    #         # Randomly select the top-left corner of the crop
    #         crop_x = random.randint(0, width - crop_width)
    #         crop_y = random.randint(0, height - crop_height)
            
    #         # Define the crop as a bounding box
    #         crop_box = torch.tensor([[crop_x, crop_y, crop_x + crop_width, crop_y + crop_height]], dtype=torch.float)

    #         # Calculate IoU between the crop box and all bounding boxes
    #         if len(bbox_tensor) == 0: # no instances in the image, so any crop is valid
    #             is_found = True
    #             break
    #         iou = ops.box_iou(crop_box, bbox_tensor)
            
    #         # Check if there is a 50% overlap with any bounding box
    #         if torch.all(iou < 0.5):
    #             is_found = True
            
    #         iters += 1
    #         if iters >= timeout_iters:
    #             return None
                
    #     return img_pil.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

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
