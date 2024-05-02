from src.data.datasets.nuscenes import NuScenesValidationMini, LABEL_ID2NAME
from src.data.datasets.waymo import WaymoValidationMini, LABEL_ID2NAME
import tqdm


# Args are here: https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/waymo_dataset.py
waymo_base_kwargs = {
    "patch_height": 256,
    "patch_aspect_ratio": 1.3,
    "label_names": ['Car'],
    "data_root": "data/waymo",
    "pipeline": [],
    "box_type_3d": "Camera",
    "load_type": 'frame_based',
    "modality": dict(use_camera=True,use_lidar=False),
    "filter_empty_gt": False,
    "test_mode": False,
    # "with_velocity": False,
    # "use_valid_flag": False,
}

waymo_val = WaymoValidationMini(**waymo_base_kwargs)

# /n/fs/pci-sharedt/ij9461/try2/generative-detection/data