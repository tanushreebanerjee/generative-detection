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
        mean = self.sum / self.n if self.n !=0 else 0
        squared_sum_expectation = self.squared_sum / self.n if self.n != 0 else 0
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
    label_names = dataset.label_names
    os.makedirs(save_dir, exist_ok=True)
    
    meters_dict = {label: {
        "t1": AverageMeter(),
        "t2": AverageMeter(),
        "t3": AverageMeter(),
        "v3": AverageMeter(),
        "l": AverageMeter(),
        "h": AverageMeter(),
        "w": AverageMeter(),
        "yaw": AverageMeter(),
        "fill_factor": AverageMeter()
    } for label in label_names}
    
    pbar = tqdm.tqdm(total=len(dataset))
    for i in range(len(dataset)):
        data = dataset[i]
        label = data['class_name']
        t1, t2, t3, v3 = data['pose_6d']
        l, h, w = data['bbox_sizes']
        yaw = data["yaw"]
        fill_factor = data["fill_factor"]
        
        meters_dict[label]['t1'].update(t1)
        meters_dict[label]['t2'].update(t2)
        meters_dict[label]['t3'].update(t3)
        meters_dict[label]['v3'].update(v3)
        meters_dict[label]['l'].update(l)
        meters_dict[label]['h'].update(h)
        meters_dict[label]['w'].update(w)
        meters_dict[label]['yaw'].update(yaw)
        meters_dict[label]['fill_factor'].update(fill_factor)
        
        pbar.update(1)
    
    # Save and print stats
    all_stats = {}
    for label, meters in meters_dict.items():
        stats = {k: v.get_stats() for k, v in meters.items()}
        all_stats[label] = stats
        print(f"{dataset.__class__.__name__} stats for {label}:")
        for k, v in stats.items():
            print(f"{k}: {v}")
        
        os.makedirs(os.path.join(save_dir, f"{dataset.__class__.__name__}"), exist_ok=True)
        with open(os.path.join(save_dir, f"{dataset.__class__.__name__}", f"{label}.pkl"), 'wb') as handle:
            pkl.dump(stats, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    return all_stats, meters_dict

def main():
    nusc_base_kwargs = {
        "label_names": ['car', 'truck', 'trailer', 
                        'bus', 'construction_vehicle', 
                        'bicycle', 'motorcycle', 
                        'pedestrian', 'traffic_cone', 'barrier'],
        "data_root": "data/nuscenes",
        "pipeline": [],
        "box_type_3d": "Camera",
        "load_type": 'frame_based',
        "modality": dict(use_camera=True,use_lidar=False),
        "filter_empty_gt": False,
        "test_mode": False,
        "with_velocity": False,
        "use_valid_flag": False,
        "patch_height": 256
        "patch_aspect_ratio": 1.0
        "perturb_center": False
        "perturb_scale": False
    }
    nusc_train = NuScenesTrain(**nusc_base_kwargs)
    nusc_val = NuScenesValidation(**nusc_base_kwargs)
    save_dir = "dataset_stats"
    train_stats, train_meters_dict = get_dataset_stats(nusc_train, save_dir=save_dir)
    val_stats, val_meters_dict = get_dataset_stats(nusc_val, save_dir=save_dir)
    
    # get combined stats from train_meters_list and val_meters_list
    combined_stats = {}
    label_names = nusc_base_kwargs["label_names"]
    for label in label_names:
        combined_stats[label] = {}
        for key in train_meters_dict[label].keys():
            combined_meter = train_meters_dict[label][key].combine(val_meters_dict[label][key])
            combined_stats[label][key] = combined_meter.get_stats()
            print("train sum: ", train_meters_dict[label][key].sum)
            print("val sum: ", val_meters_dict[label][key].sum)
            print("train n: ", train_meters_dict[label][key].n)
            print("val n: ", val_meters_dict[label][key].n)
            print("train squared sum: ", train_meters_dict[label][key].squared_sum)
            print("val squared sum: ", val_meters_dict[label][key].squared_sum)
    
    print("Combined stats:")
    for label, stats in combined_stats.items():
        print(f"Stats for {label}:")
        for k, v in stats.items():
            print(f"{k}: {v}")
    
    os.makedirs(os.path.join(save_dir, "combined"), exist_ok=True)
    with open(os.path.join(save_dir, "combined", f"all.pkl"), 'wb') as handle:
        pkl.dump(combined_stats, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    main()
