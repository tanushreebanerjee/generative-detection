# src/data/shapenet.py
import os
import json
import numpy as np
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class ShapeNetBase(Dataset):
    def __init__(self, root_dir, synset_id='02958343', transform=None):
        self.root_dir = root_dir
        self.synset_id = synset_id
        self.transform = transform
        self.data = self._load_data()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item_data = self.data[idx]
        img_path = item_data['img_path']
        img_rgb =  Image.open(img_path).convert('RGB')
                
        # Convert img_mask to a PyTorch tensor with the desired data type
        if self.transform:
            img_rgb = self.transform(img_rgb) # returns two crops of the same image
        
        sample = {
            'img_rgb': img_rgb,
            'camera_elevation': item_data['camera_elevation'],
            'camera_rotation': item_data['camera_rotation'],
            'transform_matrix': item_data['transform_matrix'],
            'object_code': item_data['object_code'],
            'aabb': item_data['aabb'],
            'camera_angle_x': item_data['camera_angle_x'],
            'img_path': item_data['img_path'],
            'synset_id': item_data['synset_id'],
            'label': item_data['label']
        }

        return sample
    
    def _load_data(self):
        data = []
        img_folder = os.path.join(self.root_dir, 'processed_get3d', 'img', self.synset_id)
        camera_folder = os.path.join(self.root_dir, 'processed_get3d', 'camera', self.synset_id)
        
        for idx, object_code in enumerate(os.listdir(img_folder)):
            object_folder = os.path.join(img_folder, object_code)
            if os.path.isdir(object_folder):
                transforms_file = os.path.join(object_folder, 'transforms.json')
                elevation_file = os.path.join(camera_folder, object_code, 'elevation.npy')
                rotation_file = os.path.join(camera_folder, object_code, 'rotation.npy')
                
                with open(transforms_file, 'r') as f:
                    transforms_data = json.load(f)
                    
                elevation = np.load(elevation_file)
                rotation = np.load(rotation_file)
               
                for frame in transforms_data['frames']:
                    file_path = os.path.join(object_folder, frame['file_path'])
                    data.append({
                        'img_path': file_path,
                        'camera_elevation': elevation,
                        'camera_rotation': rotation,
                        'transform_matrix': np.array(frame['transform_matrix']),
                        'object_code': object_code,
                        'aabb': transforms_data.get('aabb', []),
                        'camera_angle_x': transforms_data.get('camera_angle_x', 0),
                        'synset_id': self.synset_id,
                        'label': idx,
                    })

        return data
    
class ShapeNetTrain(ShapeNetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
class ShapeNetValidation(ShapeNetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    