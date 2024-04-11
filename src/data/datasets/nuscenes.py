# src/data/nuscenes.py

from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset as MMDetNuScenesDataset
from mmdet3d.registry import DATASETS
from omegaconf import OmegaConf


NUM_CAMERAS = 6

CLASSES = ('car'),      # , 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 
                        #'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
PALETTE = [
            (255, 158, 0),  # Orange
            (255, 99, 71),  # Tomato
            (255, 140, 0),  # Darkorange
            (255, 127, 80),  # Coral
            (233, 150, 70),  # Darksalmon
            (220, 20, 60),  # Crimson
            (255, 61, 99),  # Red
            (0, 0, 230),  # Blue
            (47, 79, 79),  # Darkslategrey
            (112, 128, 144),  # Slategrey
        ]

@DATASETS.register_module()
class NuScenesBase(MMDetNuScenesDataset):
    
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        
        super().__init__(**self.config)  
        assert box_type_3d.lower() == 'camera', 'Only camera box type is supported'
        assert self.config["ann_file"] is not None, 'Annotation file is required'
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
   
    def _load_relpaths(self):
        data_list = self.load_data_list()
        num_img_files = len(data_list) * NUM_CAMERAS
        rel_paths = [None] * num_img_fileser
        index = 0
        for data_info in data_list:
            rel_paths[index] = data_info['img_path']
            index += 1
        return rel_paths
    
    def _load(self):
        self.rel_paths = self._load_relpaths()
    
    
@DATASETS.register_module()
class NuScenesTrain(NuScenesBase):
    METAINFO = {
        'classes': CLASSES,
        'version': 'v1.0-trainval',
        'palette': PALETTE,
    }
    def __init__(self, **kwargs):
        self.split = "train"
        super().__init__(**kwargs)
        
@DATASETS.register_module()     
class NuScenesValidation(NuScenesBase):
    METAINFO = {
        'classes': CLASSES,
        'version': 'v1.0-trainval',
        'palette': PALETTE,
    }
    def __init__(self, **kwargs):
        self.split = "validation"
        super().__init__(**kwargs)

@DATASETS.register_module()   
class NuScenesTest(NuScenesBase):
    METAINFO = {
        'classes': CLASSES,
        'version': 'v1.0-test',
        'palette': PALETTE,
    }
    def __init__(self, **kwargs):
        self.split = "test"
        super().__init__(**kwargs)

@DATASETS.register_module()
class NuScenesMini(NuScenesBase):
    METAINFO = {
        'classes': CLASSES,
        'version': 'v1.0-mini',
        'palette': PALETTE,
    }
    def __init__(self, **kwargs):
        self.split = "mini"
        super().__init__(**kwargs)    
    
class NuScenesPatch(MMDetNuScenesDataset):
    def __init__(self, **kwargs):
        self.base = self.get_base()
    
    def __len__(self):
        return len(self.base)
    
    def _get_object_pose_6d(self, idx):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
class NuScenesPatchTrain(NuScenesPatch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_base(self):
        return NuScenesTrain()
    
class NuScenesPatchValidation(NuScenesPatch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_base(self):
        return NuScenesValidation()
    
class NuScenesPatchTest(NuScenesPatch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_base(self):
        return NuScenesTest()

class NuScenesPatchMini(NuScenesPatch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_base(self):
        return NuScenesMini()
