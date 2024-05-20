import torch
from src.util.cameras import PatchPerspectiveCameras, z_learned_to_world
import pickle as pkl

POSE_6D_DIM = 4
LHW_DIM = 3
FILL_FACTOR_DIM = 1

LABEL_NAME2ID = {
    'car': 0, 
    'truck': 1,
    'trailer': 2,
    'bus': 3,
    'construction_vehicle': 4,
    'bicycle': 5,
    'motorcycle': 6,
    'pedestrian': 7,
    'traffic_cone': 8,
    'barrier': 9,
    'background': 10
}

CLASS_NAMES = list(LABEL_NAME2ID.keys())
CLASS_IDS = list(LABEL_NAME2ID.values())

h_minmax_dir="dataset_stats/combined"
hmin_path = os.path.join(h_minmax_dir, "hmin.pkl")
hmax_path = os.path.join(h_minmax_dir, "hmax.pkl")
with open(hmin_path, "rb") as f:
    hmin_dict = pkl.load(f)
with open(hmax_path, "rb") as f:
    hmax_dict = pkl.load(f)

def get_world_coord_decoded_pose(decoded_pose_patch, camera, 
                                 patch_size, patch_center, 
                                 patch_resampling_factor,
                                 eps=None):
    
    # has a batch dim
    # first 4: t1, t2, t3, yaw
    # next 3: L, H, W
    # next 1: Fill factor
    # remaining: class probs
    # if no batch dim unsqueeze
    if len(decoded_pose_patch.shape) == 1:
        decoded_pose_patch = decoded_pose_patch.unsqueeze(0)
    
    pose_decoded = decoded_pose_patch[:, :POSE_6D_DIM]
    bbox_decoded = decoded_pose_patch[:, POSE_6D_DIM:POSE_6D_DIM+LHW_DIM]
    fill_factor_decoded = decoded_pose_patch[:, POSE_6D_DIM+LHW_DIM:POSE_6D_DIM+LHW_DIM+FILL_FACTOR_DIM]
    class_probs = decoded_pose_patch[:, POSE_6D_DIM+LHW_DIM+FILL_FACTOR_DIM:]
    
    # max id of class prob
    class_pred_id = torch.argmax(class_probs, dim=1)
    class_name = CLASS_NAMES[class_pred_id]
    
    if class_name == "background":
        return None
    
    # bbox has l, h, w, with l and w being ratios with respect to height. need to recover actual value
    length_ratio, height_absolute, width_ratio = bbox_decoded[:, 0], bbox_decoded[:, 1], bbox_decoded[:, 2]
    length_absolute = length_ratio * height_absolute
    width_absolute = width_ratio * height_absolute
    
    t1, t2, t3, yaw = pose_decoded[:, 0], pose_decoded[:, 1], pose_decoded[:, 2], pose_decoded[:, 3]
    
    # t1 and t2: in patch ndc. need to convert to world.
    point_patch = torch.stack([t1, t2, torch.ones_like(t1)], dim=1)
    point_world = camera.transform_points_world_from_patch_ndc(point_patch, 
                                                               patch_size, 
                                                               patch_center, eps=eps)
    
    # fill factor
    padding_pixels_resampled = fill_factor_decoded * patch_size[0]
    
    # t3: 
    z_learned = t3
    min_val = hmin_dict[class_name]
    max_val = hmax_dict[class_name]
    
    def get_zminmax(min_val, max_val, focal_length, patch_height):
            zmin = -(min_val * focal_length.squeeze()[0]) / (patch_height - padding_pixels_resampled)
            zmax = -(max_val * focal_length.squeeze()[0]) / (patch_height - padding_pixels_resampled)
            return zmin, zmax
    
    zmin, zmax = get_zminmax(min_val=min_val, max_val=max_val, 
                                focal_length=camera.focal_length, 
                                patch_height=patch_size[0])
    
    z_world = z_learned_to_world(z_learned=z_learned, zmin=zmin, zmax=zmax, 
                                    patch_resampling_factor=patch_resampling_factor)
    
    
    # return format: (x, y, z, l, h, w, yaw)
    
    x = point_world[:, 0]
    y = point_world[:, 1]
    z = z_world
    l = length_absolute
    h = height_absolute
    w = width_absolute
    yaw = yaw
    batch_size = point_world.shape[0]
    return_tensor = torch.zeros((batch_size, 7))
    return_tensor[:, 0] = x
    return_tensor[:, 1] = y
    return_tensor[:, 2] = z
    return_tensor[:, 3] = l
    return_tensor[:, 4] = h
    return_tensor[:, 5] = w
    return_tensor[:, 6] = yaw
    return return_tensor
