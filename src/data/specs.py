POSE_6D_DIM = 4
FILL_FACTOR_DIM=1
LHW_DIM = 3
BACKGROUND_CLASS_IDX = 10
BBOX_DIM = POSE_6D_DIM + LHW_DIM + FILL_FACTOR_DIM

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
    'background': BACKGROUND_CLASS_IDX,
}

LABEL_ID2NAME = {v: k for k, v in LABEL_NAME2ID.items()}


CAM_NAMESPACE = 'CAM'
CAMERAS = ["FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT"]
CAMERA_NAMES = [f"{CAM_NAMESPACE}_{camera}" for camera in CAMERAS]
CAM_NAME2CAM_ID = {cam_name: i for i, cam_name in enumerate(CAMERA_NAMES)}
CAM_ID2CAM_NAME = {i: cam_name for i, cam_name in enumerate(CAMERA_NAMES)}

Z_NEAR = 0.01
Z_FAR = 60.0

NUSC_IMG_WIDTH = 1600
NUSC_IMG_HEIGHT = 900

POSE_DIM = 4
LHW_DIM = 3
BBOX_3D_DIM = 7

PATCH_SIZES = [50, 100, 200, 400]