from src.data.datasets.nuscenes import NuScenesTrain, NuScenesValidation
import tqdm
import numpy as np
import os

class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = np.array([])
        
    def get_stats(self):
        ret = {
            "mean": np.mean(self.history),
            "std": np.std(self.history),
        }
        return 

    def update(self, val, n=1):
        self.history.append(val)

def get_dataset_stats(dataset, save_dir="dataset_stats"):
    os.makedirs(save_dir, exist_ok=True)
    t1_meter = AverageMeter()
    t2_meter = AverageMeter()
    t3_meter = AverageMeter()
    v1_meter = AverageMeter()
    v2_meter = AverageMeter()
    v3_meter = AverageMeter()
    l_meter = AverageMeter()
    h_meter = AverageMeter()
    w_meter = AverageMeter()
    pbar = tqdm.tqdm(total=len(nusc_train))
    for i in range(len(dataset)):
        data = nusc_train[i]
        t1, t2, t3, v1, v2, v3 = data['pose_6d']
        l, h, w = data['bbox_size']
        t1_meter.update(t1)
        t2_meter.update(t2)
        t3_meter.update(t3)
        v1_meter.update(v1)
        v2_meter.update(v2)
        v3_meter.update(v3)
        l_meter.update(l)
        h_meter.update(h)
        w_meter.update(w)
        
        pbar.update(1)
        if i % 100 == 0:
            pbar.set_description(f"t1: {t1_meter.get_stats()} t2: {t2_meter.get_stats()} t3: {t3_meter.get_stats()} v1: {v1_meter.get_stats()} v2: {v2_meter.get_stats()} v3: {v3_meter.get_stats()} l: {l_meter.get_stats()} h: {h_meter.get_stats()} w: {w_meter.get_stats()}")
        
    # save stats to file and print to stdout
    stats = {
        "t1": t1_meter.get_stats(),
        "t2": t2_meter.get_stats(),
        "t3": t3_meter.get_stats(),
        "v1": v1_meter.get_stats(),
        "v2": v2_meter.get_stats(),
        "v3": v3_meter.get_stats(),
        "l": l_meter.get_stats(),
        "h": h_meter.get_stats(),
        "w": w_meter.get_stats(),
    }

    print(f"{dataset.__class__.__name__} stats:")
    for k, v in stats.items():
        print(f"{k}: {v}")
    with open(os.path.join(save_dir, f"{dataset.__class__.__name__}_stats.txt"), "w") as f:
        f.write(str(stats))    

def main():
    nusc_base_kwargs = {
        "label_names": ['car'],
        "data_root": "data/nuscenes",
        "pipeline": [],
        "box_type_3d": "Camera",
        "load_type": 'frame_based',
        "modality": dict(use_camera=True,use_lidar=False),
        "filter_empty_gt": False,
        "test_mode": False,
        "with_velocity": False,
        "use_valid_flag": False,
    }
    nusc_train = NuScenesTrain(**nusc_base_kwargs)
    nusc_val = NuScenesValidation(**nusc_base_kwargs)

    get_dataset_stats(nusc_train)
    get_dataset_stats(nusc_val)

if __name__ == "__main__":
    main()
