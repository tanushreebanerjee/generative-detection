from src.data.datasets.nuscenes import NuScenesTrain, NuScenesValidation
import tqdm
import numpy as np
import os
import torch
import json
import pickle as pkl

class AverageMeter():
    def __init__(self, history=None):
        self.reset()
        if history is not None:
            self.history = history

    def reset(self):
        self.history = np.array([])
        
    def get_stats(self):
        # return mean and logvar concatenated as "moments" in an array
        mean = np.mean(self.history)
        std = np.std(self.history)
        logvar = np.log(std**2)
        moments = np.array([mean, logvar])
        moments = torch.from_numpy(moments).float()
        return moments
    
    def update(self, val, n=1):
        # self.history is a numpy array
        self.history = np.append(self.history, val)
        
    def combine(self, other_meter_history):
        # combine self.history with other_meter_history
        self.history = np.append(self.history, other_meter_history)
        return self

def get_dataset_stats(dataset, save_dir="dataset_stats"):
    os.makedirs(save_dir, exist_ok=True)
    t1_meter = AverageMeter()
    t2_meter = AverageMeter()
    t3_meter = AverageMeter()
    l_meter = AverageMeter()
    h_meter = AverageMeter()
    w_meter = AverageMeter()
    pbar = tqdm.tqdm(total=len(dataset))
    for i in range(len(dataset)):
        data = dataset[i]
        t1, t2, t3, _, _, _ = data['pose_6d'][0]
        l, h, w = data['bbox_sizes']
        t1_meter.update(t1)
        t2_meter.update(t2)
        t3_meter.update(t3)
        l_meter.update(l)
        h_meter.update(h)
        w_meter.update(w)
        
        pbar.update(1)
        if i % 100 == 0:
            pbar.set_description(f"t1: {t1_meter.get_stats()} t2: {t2_meter.get_stats()} t3: {t3_meter.get_stats()} l: {l_meter.get_stats()} h: {h_meter.get_stats()} w: {w_meter.get_stats()}")
        
    # save stats to file and print to stdout
    stats = {
        "t1": t1_meter.get_stats(),
        "t2": t2_meter.get_stats(),
        "t3": t3_meter.get_stats(),
        "l": l_meter.get_stats(),
        "h": h_meter.get_stats(),
        "w": w_meter.get_stats(),
    }

    print(f"{dataset.__class__.__name__} stats:")
    for k, v in stats.items():
        print(f"{k}: {v}")
        
    with open(os.path.join(save_dir, f"{dataset.__class__.__name__}.pkl"), 'wb') as handle:
        pkl.dump(stats, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    meters_dict = {
        "t1": t1_meter,
        "t2": t2_meter,
        "t3": t3_meter,
        "l": l_meter,
        "h": h_meter,
        "w": w_meter,
    } 
    return stats, meters_dict

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

    train_stats, train_meters_dict = get_dataset_stats(nusc_train)
    val_stats, val_meters_dict = get_dataset_stats(nusc_val)
    
    # get combined stats from train_meters_list and val_meters_list
    combined_stats = {}
    for key, meter in train_meters_dict.items():
        meter = meter.combine(val_meters_dict[key].history)
        combined_stats[key] = meter.get_stats()
    print("Combined stats:")
    print(combined_stats)
    with open(os.path.join("dataset_stats", "combined.pkl"), 'wb') as handle:
        pkl.dump(combined_stats, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    main()
