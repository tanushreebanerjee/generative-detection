from src.data.datasets.nuscenes import NuScenesValidationMini
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    item_idx = 35
    nusc_val = NuScenesValidationMini(**nusc_base_kwargs)
    nusc_item = nusc_val[item_idx]
    print("nusc_item", nusc_item)
    
if __name__ == "__main__":
    main()