import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import os 

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class_map = {
    0: "Disturbed Galaxy",
    1: "Merging Galaxy",
    2: "Round Smooth Galaxy",
    3: "In-between Round Smooth Galaxy",
    4: "Cigar Shaped Smooth Galaxy",
    5: "Barred Spiral Galaxy",
    6: "Unbarred Tight Spiral Galaxy",
    7: "Unbarred Loose Spiral Galaxy",
    8: "Edge-on Galaxy without Bulge",
    9: "Edge-on Galaxy with Bulge"
}


high_level = {
    "elliptical": [2, 3, 4],
    "spiral": [5, 6, 7],
    "edge-on": [8, 9],
    "irregular/merger": [0, 1]
}


class GalaxyDataset(Dataset):
    def __init__(self, file_path, transform, split='train', ssl_mode=False):
        self.file_path = file_path
        self.transform = transform
        self.ssl_mode = ssl_mode
        
        with h5py.File(self.file_path, 'r') as f:
            total_size = len(f['images'])
            indices = np.arange(total_size)
            
            # 1. Shuffle the list of ALL possible indices
            np.random.seed(42) 
            np.random.shuffle(indices)

            # 2. Split that shuffled list into test and val sets
            split_idx = int(total_size * 0.7)
            split_idx_val = int(total_size * 0.85)
            if split == 'train':
                selected = indices[:split_idx]
            elif split == 'val':
                selected = indices[split_idx:split_idx_val]
            elif split == 'test':
                selected = indices[split_idx_val:]
            
            # 3. SORT the specific bucket we chose.
            self.selected_indices = np.sort(selected)
            
            # 4. Now h5py can grab the labels in one efficient sweep
            self.labels = np.array(f['ans'][self.selected_indices])
            
            # 5. Load the actual images into RAM
            print(f"Loading {split} images into memory...")
            self.images = np.array(f['images'])[self.selected_indices]
            print(f"Loaded {len(self.images)} images into RAM for {split} split.")
            
        self.hdf5_file = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Pull from RAM instead of disk directly
        image = self.images[idx]
        
        # SimCLR mode: returns two augmented versions of the same image
        if self.ssl_mode:
            views = self.transform(image)
            return views
        
        # Supervised mode
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.as_tensor(label)

    def get_weights(self):
        counts = np.bincount(self.labels)
        weights = len(self.labels) / (len(counts) * counts)
        return torch.tensor(weights, dtype=torch.float32)

    
