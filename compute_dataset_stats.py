from src.data.datasets.nuscenes import NuScenesTrain, NuScenesValidation
import tqdm
import numpy as np
import os
import torch
import json
import pickle as pkl

class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.squared_sum = 0
        self.n = 0
        
    def get_stats(self):
        # return mean and logvar concatenated as "moments" in an array
        mean = self.sum / self.n
        squared_sum_expectation = self.squared_sum / self.n
        var = squared_sum_expectation - mean**2
        logvar = np.log(var + 1e-8) 
        moments = torch.tensor([mean, logvar], dtype=torch.float32)
        return moments
    
    def update(self, val):
        self.sum += val
        self.squared_sum += val**2
        self.n += 1
        
    def combine(self, other_meter):
        self.sum += other_meter.sum
        self.squared_sum += other_meter.squared_sum
        self.n += other_meter.n
        return self

def get_dataset_stats(dataset, save_dir="dataset_stats"):
    os.makedirs(save_dir, exist_ok=True)
    t1_meter = AverageMeter()
    t2_meter = AverageMeter()
    t3_meter = AverageMeter()
    v3_meter = AverageMeter()
    l_meter = AverageMeter()
    h_meter = AverageMeter()
    w_meter = AverageMeter()
    yaw_meter = AverageMeter()
    fill_factor_meter = AverageMeter()
    pbar = tqdm.tqdm(total=len(dataset))
    for i in range(len(dataset)):
        data = dataset[i]
        t1, t2, t3, v3 = data['pose_6d']
        l, h, w = data['bbox_sizes']
        yaw = data["yaw"]
        fill_factor = data["fill_factor"]
        t1_meter.update(t1)
        t2_meter.update(t2)
        t3_meter.update(t3)
        v3_meter.update(v3)
        l_meter.update(l)
        h_meter.update(h)
        w_meter.update(w)
        yaw_meter.update(yaw)
        fill_factor_meter.update(fill_factor)
        
        pbar.update(1)
        
    # save stats to file and print to stdout
    stats = {
        "t1": t1_meter.get_stats(),
        "t2": t2_meter.get_stats(),
        "t3": t3_meter.get_stats(),
        "v3": v3_meter.get_stats(),
        "l": l_meter.get_stats(),
        "h": h_meter.get_stats(),
        "w": w_meter.get_stats(),
        "yaw": yaw_meter.get_stats(),
        "fill_factor": fill_factor_meter.get_stats(),
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
        "v3": v3_meter,
        "l": l_meter,
        "h": h_meter,
        "w": w_meter,
        "yaw": yaw_meter,
        "fill_factor": fill_factor_meter,
    }
    
    for key, meter in meters_dict.items():
        print(f"{dataset.__class__.__name__} {key} meter:")
        print(f"{key}: sum={meter.sum}, squared_sum={meter.squared_sum}, n={meter.n}")
        print(f"{key}: mean={meter.sum/meter.n}, var={meter.squared_sum/meter.n - (meter.sum/meter.n)**2}")
        print(f"{key}: logvar={np.log(meter.squared_sum/meter.n - (meter.sum/meter.n)**2 + 1e-8)}")
    
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
    save_dir = "dataset_stats"
    train_stats, train_meters_dict = get_dataset_stats(nusc_train, save_dir=save_dir)
    val_stats, val_meters_dict = get_dataset_stats(nusc_val, save_dir=save_dir)
    
    # get combined stats from train_meters_list and val_meters_list
    combined_stats = {}
    for key, meter in train_meters_dict.items():
        meter = meter.combine(val_meters_dict[key])
        combined_stats[key] = meter.get_stats()
    print("Combined stats:")
    print(combined_stats)
    
    with open(os.path.join(save_dir, "combined_train_val.pkl"), 'wb') as handle:
        pkl.dump(combined_stats, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    main()
