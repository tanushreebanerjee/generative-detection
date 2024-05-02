from src.data.datasets.nuscenes import NuScenesValidationMini, LABEL_ID2NAME
from src.data.datasets.waymo import WaymoValidationMini, LABEL_ID2NAME
import matplotlib.pyplot as plt
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

# ERROR: No such file or directory: 'data/waymo/waymo_mini_infos_val.pkl'
# Right now: data/processed/waymo/....
#   no pkl file in there

print(len(waymo_val))
print(waymo_val[5])

# mark patch_center_2d as red on full image
idx = 50

for idx in tqdm.tqdm(range(len(waymo_val))):
    full_img = waymo_val[idx].full_img.permute(1,2,0)

    # mark patch_center_2d as red on full image 
    patch_center_2d = waymo_val[idx].patch_center_2d # torch.Size([2])

idx = 11
patch = waymo_val[idx].patch.permute(1,2,0).numpy()
plt.imshow(patch)
plt.savefig('patch.png')
plt.show()

full_img = waymo_val[idx].full_img.permute(1,2,0)

# mark patch_center_2d as red on full image 
patch_center_2d = waymo_val[idx].patch_center_2d # torch.Size([2])

full_img = full_img.numpy()
full_img = full_img.copy() # (900, 1600, 3)

# make patch_center_2d as red with surrounding pixels red

patch_center_2d = patch_center_2d.numpy()
patch_center_2d = patch_center_2d.astype(int)
patch_center_2d = patch_center_2d.tolist()

for i in range(-5, 6):
    for j in range(-5, 6):
        full_img[patch_center_2d[1]+i, patch_center_2d[0]+j, :] = [1,0,0]
        
plt.imshow(full_img)
plt.savefig('patch2.png')
plt.show()

print("Image and Patch Shapes:", nusc_val[0].full_img.shape, nusc_val[0].patch.shape)
print("MIN AND MAX:", waymo_val[0].full_img.min(), waymo_val[0].full_img.max())