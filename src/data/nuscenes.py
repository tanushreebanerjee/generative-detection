# src/data/nuscenes.py

from torch.utils.data import Dataset

class nuScenesBase(Dataset):
    def __init__(self, root_dir, version='v1.0-trainval', verbose=True):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    
    def _load_data(self):    
        pass
    
    def _load_image(self, sample):
        pass
    

class nuScenesTrain():
    def __init__(self, root_dir, version='v1.0-trainval', verbose=True):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    
    def _load_data(self):    
        pass
    
    def _load_image(self, sample):
        pass
    
class nuScenesValidation():
    def __init__(self, root_dir, version='v1.0-trainval', verbose=True):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    
    def _load_data(self):    
        pass
    
    def _load_image(self, sample):
        pass
    