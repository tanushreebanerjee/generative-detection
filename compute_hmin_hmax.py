import pickle as pkl
import os
import numpy as np

dataset_stats_path = "dataset_stats/combined/all.pkl"

with open(dataset_stats_path, "rb") as f:
    dataset_stats = pkl.load(f)
    
L_MIN = 0.5 
L_MAX = 3.0

hmin_dict = {}
hmax_dict = {}
for class_name in dataset_stats.keys():
    class_stats = dataset_stats[class_name] # mean, logvar
    height_stats = class_stats['h']
    mean = height_stats[0]
    std = np.sqrt(np.exp(height_stats[1]))

    new_H_MIN = mean - 2*std
    new_H_MAX = mean + 2*std
    hmin_dict[class_name] = new_H_MIN
    hmax_dict[class_name] = new_H_MAX

# save the new hmin and hmax dicts into a pkl file
h_minmax_dir = "dataset_stats/combined/"
os.makedirs(h_minmax_dir, exist_ok=True)
hmin_path = os.path.join(h_minmax_dir, "hmin.pkl")
hmax_path = os.path.join(h_minmax_dir, "hmax.pkl")
with open(hmin_path, "wb") as f:
    pkl.dump(hmin_dict, f)
with open(hmax_path, "wb") as f:
    pkl.dump(hmax_dict, f)
