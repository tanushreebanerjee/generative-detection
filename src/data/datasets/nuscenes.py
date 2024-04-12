# src/data/nuscenes.py

from mmdet3d.registry import DATASETS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset as MMDetNuScenesDataset
from torch.utils.data import Dataset
from src.util.misc import EasyDict as edict
import cv2
import os

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

@DATASETS.register_module()
class NuScenesCameraInstances(Dataset):
    def __init__(self, cam_instances, img_path):
        
        self.cam_instances = cam_instances
        self.img_path = img_path
        self.patches = self._generate_patches()
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
            bbox = cam_instance.bbox # 2D bounding box annotation (exterior rectangle of the projected 3D box), a list arrange as [x1, y1, x2, y2].
            x1, y1, x2, y2 = bbox
            try:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                patch = img[y1:y2, x1:x2]
            except Exception as e:
                print(f"Error in cropping image: {e}")
                # return full image if error occurs
                patch = img
            
            patches.append(patch)
        return patches
  
    def __getitem__(self, idx):
        cam_instance = edict(self.cam_instances[idx])   
        cam_instance.patch = self.patches[idx]
        return cam_instance

class NuScenesBase(MMDetNuScenesDataset):
    def __init__(self, data_root, label_names, is_sweep=False, **kwargs):
        self.data_root = data_root
        self.img_root = os.path.join(data_root, "samples" if not is_sweep else "sweeps")
        super().__init__(data_root=data_root, **kwargs)
        self.label_names = label_names
        self.label_ids = [LABEL_NAME2ID[label_name] for label_name in label_names]
        print(f"Using label names: {self.label_names}, label ids: {self.label_ids}")
    
    def __len__(self):
        self.num_samples = super().__len__()
        self.num_cameras = len(CAMERA_NAMES)
        return self.num_samples * self.num_cameras
    
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
        img_file = sample_img_info.img_path.split("/")[-1]
        ret.cam_instances = NuScenesCameraInstances(cam_instances=cam_instances, img_path=os.path.join(self.img_root, cam_name, img_file))
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