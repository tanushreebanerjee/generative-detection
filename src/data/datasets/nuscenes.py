# src/data/nuscenes.py

from torch.utils.data import Dataset
from omegaconf import OmegaConf
import cProfile, pstats, io
from pstats import SortKey

class nuScenesBase(Dataset):
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
        raise NotImplementedError("_filter_split method must be implemented")
    
    def _load(self):
        raise NotImplementedError("_load method must be implemented")

    def _load(self):
        raise NotImplementedError("_load method must be implemented")
    
class nuScenesTrain(nuScenesBase):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        self.split = "train"
        super().__init__(**kwargs)
    
    def _prepare(self):
        self.random_crop = retrieve(self.config, "random_crop", default=False)

class nuScenesValidation(nuScenesBase):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        super().__init__(**kwargs)
    
    def _prepare(self):
        pass
    
class nuScenesPatch(nuScenesBase):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        super().__init__(**kwargs)
    
    def _prepare(self):
        pass
    
class nuScenesPatchTrain(nuScenesPatch):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        super().__init__(process_images=process_images, data_root=data_root, **kwargs)
    
    def _prepare(self):
        pass
    
class nuScenesPatchValidation(nuScenesPatch):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        super().__init__(process_images=process_images, data_root=data_root, **kwargs)
    
    def _prepare(self):
        pass
    
class nuScenesPatchTest(nuScenesPatch):
    def __init__(self, process_images=True, data_root=None, **kwargs):
        super().__init__(process_images=process_images, data_root=data_root, **kwargs)
    
    def _prepare(self):
        pass
