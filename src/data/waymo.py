# src/data/waymo.py

from torch.utils.data import Dataset

class WaymoBase(Dataset):
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
    
class WaymoTrain():
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

class WaymoValidation():
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
    