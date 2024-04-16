# src/data/nuscenes.py
import torch
from mmdet3d.registry import DATASETS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset as MMDetNuScenesDataset
from torch.utils.data import Dataset
from src.util.misc import EasyDict as edict
import cv2
import os
from src.util.cameras import PatchPerspectiveCameras as PatchCameras
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, se3_log_map, se3_exp_map
from torchvision.transformsx import Resize
from torchvision.transforms.functional import InterpolationMode

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
Z_FAR = 80.0

NUSC_IMG_WIDTH = 1600
NUSC_IMG_HEIGHT = 900

@DATASETS.register_module()
class NuScenesCameraInstances(Dataset):
    def __init__(self, cam_instances, img_path, patch_size, cam2img):
        
        self.cam_instances = cam_instances
        self.img_path = img_path
        self.patches = self._generate_patches()
        self.patch_size = patch_size
        self.cam2img = cam2img
        self.camera = PatchCameras(
            znear=Z_NEAR,
            zfar=Z_FAR,
            K=cam2img,
            device=torch.device.is_available(),
            image_size=(NUSC_IMG_HEIGHT, NUSC_IMG_WIDTH))

        self.resize = Resize(size=patch_size, 
                             interpolation=InterpolationMode.BILINEAR, 
                             max_size=None, 
                             antialias=True)
        
    def __len__(self):
        return len(self.cam_instances)
    
    def _generate_patches(self):
        # load image from path
        patches = []
        img = cv2.imread(self.img_path)
        if img is None:
            # if instances present but file not found, raise error. else return empty list
            if self.__len__() > 0:
                raise FileNotFoundError(f"Image not found at {self.img_path} but {self.__len__()} instances are present")
            else:
                return []
 
        # return croped list of images as defined by 2d bbox for each instance
        for cam_instance in self.cam_instances:
            
            bbox = cam_instance.bbox # bounding box annotation (exterior rectangle of the projected 3D box), a list arrange as [x1, y1, x2, y2].
            
            # use interpolation torch to get crop of image from bbox of size patch_size
            # need to use torch for interpolation, not cv2
            x1, y1, x2, y2 = bbox
            try:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                patch = img[y1:y2, x1:x2]
            except Exception as e:
                print(f"Error in cropping image: {e}")
                # return full image if error occurs
                patch = img
            
            patch = self.resize(patch)
            patch = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1)
            
            patches.append(patch)
        return patches
     
    def _get_pose_6d_lhw(self, cam_instance):
        bbox_3d = cam_instance.bbox_3d
        
        # How to convert yaw from cam coords to patch ndc?
        x, y, z, l, h, w, yaw = bbox_3d # in camera coordinate system? need to convert to patch NDC
        
        point_camera = torch.tensor([x, y, z], dtype=torch.float32)
        point_patch_ndc = self.cameras.transform_points_patch_ndc(points=point_camera,
                                                                  patch_size=self.patch_size, 
                                                                    patch_center=cam_instance.center_2d)
        
        x_patch, y_patch, z_patch = point_patch_ndc
        
        roll, pitch = 0.0, 0.0 # roll and pitch are 0 for all instances in nuscenes dataset
        
        euler_angles = torch.tensor([yaw, pitch, roll], dtype=torch.float32)
        convention = "XYZ"
        
        R = euler_angles_to_matrix(euler_angles, convention)
        
        translation = torch.tensor([x_patch, y_patch, z_patch], dtype=torch.float32)
        
        # A SE(3) matrix has the following form: ` [ R 0 ] [ T 1 ] , `
        se3_exp_map = torch.eye(4)
        se3_exp_map[:3, :3] = R
        se3_exp_map[:3, 3] = translation

        # form must be ` [ R 0 ] [ T 1 ] , ` so need to transpose
        se3_exp_map = se3_exp_map.T
        
        se3_log_map = se3_log_map(se3_exp_map) # 6d vector
        
        h_aspect_ratio = h / l
        w_aspect_ratio = w / l
        
        bbox_sizes = torch.tensor([l, h_aspect_ratio, w_aspect_ratio], dtype=torch.float32)
        return se3_log_map, bbox_sizes
        
    def __getitem__(self, idx):
        cam_instance = edict(self.cam_instances[idx])   
        cam_instance.patch = self.patches[idx]
        cam_instance.pose_6d, cam_instance.bbox_sizes = self._get_pose_6d_lhw(cam_instance)
        return cam_instance

class NuScenesBase(MMDetNuScenesDataset):
    def __init__(self, data_root, label_names, patch_height=256, patch_aspect_ratio=1.0,
                 is_sweep=False, **kwargs):
        self.label_names = label_names
        self.label_ids = [LABEL_NAME2ID[label_name] for label_name in label_names]
        self.data_root = data_root
        self.img_root = os.path.join(data_root, "samples" if not is_sweep else "sweeps")
        self.patch_size = (patch_height, int(patch_height * patch_aspect_ratio)) # aspect ratio is width/height
        self._load_data()
        print(f"Using label names: {self.label_names}, label ids: {self.label_ids}")
        self.num_cam_instances = self._get_num_cam_instances()
        super().__init__(data_root=data_root, **kwargs)
        
    def _get_num_cam_instances(self):
        # sum of all instances in each camera for all samples
        num_cam_instances = 0
        for sample_idx in range(len(self)):
            sample_info = edict(super().__getitem__(sample_idx))
            for cam_name in CAMERA_NAMES:
                cam_instances = sample_info.cam_instances[cam_name]
                cam_instances = [cam_instance for cam_instance in cam_instances if cam_instance['bbox_label'] in self.label_ids]
                num_cam_instances += len(cam_instances)
        return num_cam_instances
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _load_data(self):
        num_samples = super().__len__()
        num_cameras = len(CAMERA_NAMES)
        # create empty list of length self._get_num_cam_instances() to store data
        data = []
        for sample_idx in range(num_samples):
            sample_info = edict(super().__getitem__(sample_idx))
            for cam_idx in range(num_cameras):
                cam_name = CAM_ID2CAM_NAME[cam_idx]
                sample_img_info = edict(sample_info.images[cam_name])
                cam_instances = sample_info.cam_instances[cam_name]
                cam_instances = [cam_instance for cam_instance in cam_instances if cam_instance['bbox_label'] in self.label_ids]
                for cam_instance_idx, cam_instance in enumerate(cam_instances):
                    ret = edict()
                    img_file = sample_img_info.img_path.split("/")[-1]
                    cam_instance.img_path = os.path.join(self.img_root, cam_name, img_file)
                    ret.sample_idx = sample_idx
                    ret.cam_idx = cam_idx
                    ret.cam_name = cam_name
                    ret.update(sample_img_info)
                    cam_instance = NuScenesCameraInstances(cam_instances=[cam_instance], img_path=cam_instance.img_path, patch_size=self.patch_size, cam2img=sample_info.cam2img)
                    ret.cam_instance = cam_instance
                    data.append(ret)
        self.data = data

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
