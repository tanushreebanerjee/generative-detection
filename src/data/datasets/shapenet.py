# src/data/shapenet.py
import os
import json
import numpy as np
import numpy as np
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from taming.data.imagenet import retrieve, ImagePaths

class ShapeNetBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self.keep_orig_class_label = self.config.get("keep_orig_class_label", False)
        self.process_images = True  # if False we skip loading & processing images and self.data contains filepaths
        self._prepare()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths):
        ignore = set([])
        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
        return relpaths
    
    def _load_transforms(self):
        self.transforms = np.array([])
        self.transforms_paths = np.array([])
        for rpath in self.relpaths:
            synset = rpath.split("/")[0]
            obj = rpath.split("/")[1]
            transforms_path = os.path.join(self.data_root, "img", synset, obj, "transforms.json")
            with open(transforms_path, "r") as f:
                transforms = json.load(f)
                self.transforms = np.append(self.transforms, transforms)
                self.transforms_paths = np.append(self.transforms_paths, transforms_path)
    
    def _load_camera(self):        
        self.elevations = np.array([])
        self.rotations = np.array([])
        for rpath in self.relpaths:
            synset = rpath.split("/")[0]
            obj = rpath.split("/")[1]
            elevation_path = os.path.join(self.data_root, "camera", synset, obj, "elevation.npy")
            rotation_path = os.path.join(self.data_root, "camera", synset, obj, "rotation.npy")
            elevation = np.load(elevation_path)
            rotation = np.load(rotation_path)
            self.elevations = np.append(self.elevations, elevation)
            self.rotations = np.append(self.rotations, rotation)
     
    def _load(self):
        self.data_root = self.config.get("data_root", "data/processed/shapenet/processed_get3d")
    
        img_dir = os.path.join(self.data_root, "img")
        self.relpaths = [os.path.relpath(os.path.join(root, file), img_dir) for root, _, files in os.walk(img_dir) for file in files if file.endswith(".png")]
        l1 = len(self.relpaths)
        self.relpaths = self._filter_relpaths(self.relpaths)
        print("Removed {} files from filelist during filtering.".format(l1 - len(self.relpaths)))
        
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
        
        # divide into train, validation and test sets by object
        self.split = retrieve(self.config, "split", default={"train": 0.8, "validation": 0.1, "test": 0.1})
        # assign each object to a split
        self.split_idx = np.random.choice(3, len(self.objects), p=[self.split["train"], self.split["validation"], self.split["test"]])
        self.split_idx = np.array(self.split_idx)
        self.split_idx = np.array([self.split_idx[self.objects.index(o)] for o in self.objects])
                
        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "transforms": self.transforms,
            "transforms_path": self.transforms_paths,
            "elevations": self.elevations,
            "rotations": self.rotations,
            "abspaths": np.array(self.abspaths),
            "split_idx": np.array(self.split_idx),
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
        super().__init__(**kwargs)
    
    def _prepare(self):
        pass
        
class ShapeNetValidation(ShapeNetBase):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.data_root = data_root
        self.process_images = process_images
        super().__init__(**kwargs)
        
    def _prepare(self):
        pass
        
class ShapeNetTest(ShapeNetBase):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.data_root = data_root
        self.process_images = process_images
        super().__init__(**kwargs)
        
    def _prepare(self):
        pass
    