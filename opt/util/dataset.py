from .llff_dataset import LLFFDataset
from .nerf_dataset import NeRFDataset
from os import path

def auto_dataset(root : str, *args, **kwargs):
    if path.isfile(path.join(root, 'poses_bounds.npy')):
        print("Detected LLFF dataset")
        return LLFFDataset(root, *args, **kwargs)
    elif path.isfile(path.join(root, 'transforms.json')) or \
         path.isfile(path.join(root, 'transforms_train.json')):
        print("Detected NeRF (Blender) dataset")
        return NeRFDataset(root, *args, **kwargs)

datasets = {
    'nerf': NeRFDataset,
    'llff': LLFFDataset,
    'auto': auto_dataset
}
