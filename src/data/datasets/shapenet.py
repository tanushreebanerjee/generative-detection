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
        self.split = None
        self.random_seed = retrieve(self.config, "random_seed", default=42)
        np.random.seed(self.random_seed)
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
    
    def _filter_split(self, relpaths):
        """Filters the given list of relative paths based on the specified split percentages."""
        if self.split is not None:
            assert self.split in ["train", "validation", "test"], f"Invalid split {self.split}."
            split_prop = retrieve(self.config, "split", default={"train": 0.8, "validation": 0.1, "test": 0.1})
            unique_objects = np.unique([rpath.split("/")[1] for rpath in relpaths])
            split_idx = np.random.choice(3, len(unique_objects), p=[split_prop["train"], split_prop["validation"], split_prop["test"]])
            split_idx = np.array(split_idx)
            split_idx = np.array([split_idx[list(unique_objects).index(o)] for o in [rpath.split("/")[1] for rpath in relpaths]])
            relpaths = [rpath for rpath, idx in zip(relpaths, split_idx) if idx == self.split]
        
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
        
    def _save_splits(self, unique_objects):
        split_dir = os.path.join(self.data_root, "splits")
        os.makedirs(split_dir, exist_ok=True)
        split_file = os.path.join(split_dir, f"{self.split}.txt")
        with open(split_file, "w") as f:
            for o in unique_objects:
                f.write(f"{o}\n")
    
    def _load(self):
        self.data_root = self.config.get("data_root", "data/processed/shapenet/processed_get3d")
    
        img_dir = os.path.join(self.data_root, "img")
        self.relpaths = [os.path.relpath(os.path.join(root, file), img_dir) for root, _, files in os.walk(img_dir) for file in files if file.endswith(".png")]
        l1 = len(self.relpaths)
        self.relpaths = self._filter_relpaths(self.relpaths)
        print("Removed {} files from filelist during filtering.".format(l1 - len(self.relpaths)))
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
            
        # save objects in split to file
        self._save_splits(unique_objects)
            
class ShapeNetTrain(ShapeNetBase):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        self.split = "train"
        super().__init__(**kwargs)
    
    def _prepare(self):
        self.random_crop = retrieve(self.config, "random_crop", default=True)
        
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
    