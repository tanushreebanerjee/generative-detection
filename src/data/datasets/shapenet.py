# src/data/shapenet.py
import os
import json
import numpy as np
import albumentations
import torch
from PIL import Image
from torch.utils.data import Dataset
from omegaconf import OmegaConf
import cv2
from taming.data.imagenet import retrieve, ImagePaths
import logging
import cProfile, pstats, io
from pstats import SortKey
from math import radians
from src.util.pose_transforms import euler_angles_translation2se3_log_map

TRAIN_MINI_NUM_OBJECTS = 10
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
POSE_6D_DIM = 6

def create_splits(config):
    splits_dir = retrieve(config, "splits_dir", default="data/splits/shapenet")
    data_root = retrieve(config, "data_root", default="data/processed/shapenet/processed_get3d")
    split_prop = retrieve(config, "split", default={"train": TRAIN_SPLIT, "validation": VALIDATION_SPLIT, "test": TEST_SPLIT})
    shuffle = retrieve(config, "shuffle", default=True)
    
    synsets = os.listdir(os.path.join(data_root, "img"))
    objects = np.array([os.listdir(os.path.join(data_root, "img", synset)) for synset in synsets]).flatten()
    
    if shuffle:
        np.random.shuffle(objects)
    
    split_objects = {split: [None] * int(len(objects) * prop) for split, prop in split_prop.items()}
    split_counts = {split: 0 for split in split_prop.keys()}
    for obj in objects:
        for split, prop in split_prop.items():
            if split_counts[split] < len(objects) * prop:
                split_objects[split][split_counts[split]] = obj
                split_counts[split] += 1
                break

    os.makedirs(splits_dir, exist_ok=True)
    for split, objects in split_objects.items():
        with open(os.path.join(splits_dir, f"{split}.txt"), "w") as f:
            for obj in objects:
                f.write(obj + "\n")    

    return split_objects            
    
class ShapeNetBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self.keep_orig_class_label = self.config.get("keep_orig_class_label", False)
        self.process_images = True  # if False we skip loading & processing images and self.data contains filepaths
        pr = cProfile.Profile()
        pr.enable()
        self._prepare()
        self._load()
        pr.disable()
        self._output_profiler_logs(pr)
        
    def _output_profiler_logs(self, pr):
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        profiler_logs_dir = f"logs/profiler_logs/{self.__class__.__name__}"
        profiler_logs_path = os.path.join(profiler_logs_dir, self.split + ".txt")
        os.makedirs(profiler_logs_dir, exist_ok=True)
        ps.print_stats()
        with open(profiler_logs_path, "w") as f:
            f.write(s.getvalue())
          
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        raise NotImplementedError("Subclass must implement _prepare method")

    def _filter_relpaths(self, relpaths):
        ignore = set([])
        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
        return relpaths
    
    def _filter_split(self, relpaths):
        """Filters the given list of relative paths based on the specified split percentages."""
        if self.split is not None:
            assert self.split in ["train", "validation", "test", "train-mini"], f"Invalid split {self.split}."
            objects_in_split_path = os.path.join(self.config.get("splits_dir", "data/splits/shapenet"), f"{self.split}.txt")
            with open(objects_in_split_path, "r") as f:
                objects_in_split = f.read().splitlines()
            relpaths = [rpath for rpath in relpaths if rpath.split("/")[1] in objects_in_split]
        return relpaths
    
    def _load_transforms(self):
        self.transforms = np.array([None] * len(self.relpaths))
        self.transforms_paths = np.array([None] * len(self.relpaths))
        for idx, rpath in enumerate(self.relpaths):
            synset = rpath.split("/")[0]
            obj = rpath.split("/")[1]
            transforms_path = os.path.join(self.data_root, "img", synset, obj, "transforms.json")
            with open(transforms_path, "r") as f:
                transforms = json.load(f)
                self.transforms[idx] = transforms
                self.transforms_paths[idx] = transforms_path
    
    def _load_camera(self):     
        self.elevations = np.array([None] * len(self.relpaths))
        self.rotations = np.array([None] * len(self.relpaths))
        
        for idx, rpath in enumerate(self.relpaths):
            synset = rpath.split("/")[0]
            obj = rpath.split("/")[1]
            cam_idx = int(rpath.split("/")[2].split(".")[0])
            elevation_path = os.path.join(self.data_root, "camera", synset, obj, "elevation.npy")
            rotation_path = os.path.join(self.data_root, "camera", synset, obj, "rotation.npy")
            elevation_arr = np.load(elevation_path)
            rotation_arr = np.load(rotation_path)
            elevation = elevation_arr[cam_idx]
            rotation = rotation_arr[cam_idx]
            self.elevations[idx] = elevation
            self.rotations[idx] = rotation
            
    def count_png_files(self, directory):
        count = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.png'):
                    count += 1
        return count

    def _load_imgs(self, img_dir):
        self.relpaths = [None] * self.count_png_files(img_dir)
        index = 0
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.endswith('.png'):
                    self.relpaths[index] = os.path.relpath(os.path.join(root, file), img_dir)
                    index += 1
        return self.relpaths
                    
    def _load(self):
        self.data_root = self.config.get("data_root", "data/shapenet/processed_get3d")
        img_dir = os.path.join(self.data_root, "img")
        self._load_imgs(img_dir)                 
        l1 = len(self.relpaths)
        self.relpaths = self._filter_relpaths(self.relpaths)
        logging.info("Removed {} files from filelist during filtering.".format(l1 - len(self.relpaths)))
        self.relpaths = self._filter_split(self.relpaths)
        
        self.synsets = [rpath.split("/")[0] for rpath in self.relpaths]
        self.objects = [rpath.split("/")[1] for rpath in self.relpaths]
        self.class_labels = [f"{s}_{o}" for s, o in zip(self.synsets, self.objects)]
        self.abspaths = [os.path.join(os.getcwd(), self.data_root, "img", rpath) for rpath in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        unique_objects = np.unique(self.objects)
        unique_class_labels = np.unique(self.class_labels)
        
        self.synset2idx = {s: i for i, s in enumerate(unique_synsets)}
        self.object2idx = {s: i for i, s in enumerate(unique_objects)}
        self.class_label2idx = {s: i for i, s in enumerate(unique_class_labels)}
        
        if not self.keep_orig_class_label:
            self.class_labels = [self.class_label2idx[s] for s in self.class_labels]
            self.synsets = [self.synset2idx[s] for s in self.synsets]
            self.objects = [self.object2idx[s] for s in self.objects]
        else:
            raise NotImplementedError()

        self._load_transforms()
        self._load_camera()
                
        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "transforms": self.transforms,
            "transforms_path": self.transforms_paths,
            "elevations": self.elevations,
            "rotations": self.rotations,
            "abspaths": np.array(self.abspaths)
        }

        if self.process_images:
            self.size = retrieve(self.config, "size", default=1024)
            self.data = ImagePaths(self.abspaths,
                                   labels=labels,
                                   size=self.size,
                                   random_crop=self.random_crop,
                                   )
        else:
            self.data = labels
        
class ShapeNetTrain(ShapeNetBase):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        self.split = "train"
        super().__init__(**kwargs)
    
    def _prepare(self):
        self.random_crop = retrieve(self.config, "random_crop", default=False)
        
class ShapeNetValidation(ShapeNetBase):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.data_root = data_root
        self.process_images = process_images
        self.split = "validation"
        super().__init__(**kwargs)
        
    def _prepare(self):
        self.random_crop = retrieve(self.config, "random_crop", default=False)

class ShapeNetTest(ShapeNetBase):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.data_root = data_root
        self.process_images = process_images
        self.split = "test"
        super().__init__(**kwargs)

    def _prepare(self):
        self.random_crop = retrieve(self.config, "random_crop", default=False)

class ShapeNetTrainMini(ShapeNetBase):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        self.split = "train-mini"
        super().__init__(**kwargs)
    
    def _prepare(self):
        self.random_crop = retrieve(self.config, "random_crop", default=False)

class ShapeNetPose(Dataset):
    def __init__(self, size=None, euler_convention=None):
        """
        Initialize the ShapeNet dataset.

        Args:
            size (int): The maximum size of the images after rescaling.
        """
        self.base = self.get_base()
        
        assert size is not None, "size must be specified"
        self.size = size
        
        assert euler_convention is not None, "euler_angle_convention must be specified"
        assert euler_convention in ["XYZ", "ZYX"], f"Invalid euler_angle_convention {euler_convention}. Must be 'XYZ' or 'ZYX'."
        self.euler_convention = euler_convention
        
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

    def __len__(self):
        return len(self.base)
    
    def _get_object_pose_6d(self, idx):
            """
            Get the pose of the object at the given index as screw parameters.

            Parameters:
                idx (int): The index of the object.

            Returns:
                object_pose (np.ndarray): The pose of the object as screw parameters.
            """
            # Assuming rotation and elevation are given in degrees
            rotation_deg = self.base.rotations[idx]
            elevation_deg = self.base.elevations[idx]

            # Convert degrees to radians
            rotation_rad = radians(rotation_deg)
            elevation_rad = radians(elevation_deg)

            # Convert the Euler angles and translation to screw parameters
            roll = 0
            pitch = elevation_rad
            yaw = rotation_rad
            
            # set euler angles in order based on convention
            if self.euler_convention == "XYZ":
                euler_angle = torch.tensor([roll, pitch, yaw])
            elif self.euler_convention == "ZYX":
                euler_angle = torch.tensor([yaw, pitch, roll])
            else:
                raise ValueError(f"Invalid convention {self.euler_convention}. Must be 'XYZ' or 'ZYX'.")
            
            translation = torch.tensor([0, 0, 0])
            object_pose_screw = euler_angles_translation2se3_log_map(euler_angle, translation, self.euler_convention)
            assert object_pose_screw.shape[1] == POSE_6D_DIM, f"Expected 6D pose, got {object_pose_screw.shape[1]}D pose."
            return object_pose_screw
    
    def __getitem__(self, i):
        example = self.base[i]
        image = Image.open(example["file_path_"]) 
        if not image.mode == "RGBA":
            image = image.convert("RGBA")

        image = np.array(image).astype(np.uint8)

        image = self.image_rescaler(image=image)["image"]
        example["image_rgba"] = (image/127.5 - 1.0).astype(np.float32) # normalize to [-1, 1]
        example["image_rgb"] = example["image_rgba"][:, :, :3]
        example["image_mask"] = example["image_rgba"][:, :, 3]
        example["pose"] = self._get_object_pose_6d(i).reshape(-1)
        return example
        
class ShapeNetPoseTrain(ShapeNetPose):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_base(self):
        return ShapeNetTrain()

class ShapeNetPoseValidation(ShapeNetPose):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_base(self):
        return ShapeNetValidation()
    
class ShapeNetPoseTest(ShapeNetPose):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_base(self):
        return ShapeNetTest()

class ShapeNetPoseTrainMini(ShapeNetPose):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_base(self):
        return ShapeNetTrainMini()