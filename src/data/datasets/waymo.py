# src/data/waymo.py

from torch.utils.data import Dataset
from omegaconf import OmegaConf

class WaymoBase(Dataset):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
            self.process_images = True  # if False we skip loading & processing images and self.data contains filepaths
            self._prepare()
            self._load()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _prepare(self):
        raise NotImplementedError("Subclass must implement _prepare method")
    
    def _load(self):
        raise NotImplementedError("_load method must be implemented")

class WaymoTrain(WaymoBase):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        super().__init__(**kwargs)
    
    def _prepare(self):
        pass

class WaymoValidation(WaymoBase):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        super().__init__(**kwargs)
    
    def _prepare(self):
        pass