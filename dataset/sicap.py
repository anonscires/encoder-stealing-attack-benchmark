from torch.utils.data import Dataset
from torchvision import datasets
import h5py
import matplotlib as plt
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


class SicapSingleDataset(Dataset):
    def __init__(self, root, train=True, transform=None,
                 percent_data=1.0, corrupt_pct=0.0,clip_version="CLIP"):
        """
        corrupt_pct: fraction of examples whose labels
                    will be randomly “rotated” to a wrong label.
        """
        csv_file = os.path.join(
            root,
            "partition/Test/Train.xlsx" if train else "partition/Test/Test.xlsx"
        )
        self.clip_version = clip_version
        self.data_frame = pd.read_excel(csv_file)
        self.root  = os.path.join(root, "images")
        self.transform = transform
        
        # sample down for % data
        
        if not (percent_data == 1.0):
            sample_size = int(len(self.data_frame) * percent_data)
            self.data_frame = self.data_frame.sample(
                n=sample_size, random_state=42
            ).reset_index(drop=True)

        self.num_classes = 4  # assuming 5 label columns
        self.corrupt_pct  = corrupt_pct
        self.corrupted_map = {}  # idx -> true_label

        if corrupt_pct > 0:
            # choose which indices to corrupt
            
            n_corrupt = int(len(self.data_frame) * corrupt_pct)
            print("Corrupt percent",corrupt_pct,"no of corrupt",n_corrupt)
            corrupt_indices = np.random.choice(
                len(self.data_frame), size=n_corrupt, replace=False
            )
            for idx in corrupt_indices:
                true_lbl = self._get_label(idx)
                # pick a WRONG label uniformly
                wrong_lbl = np.random.choice(
                    [c for c in range(self.num_classes) if c != true_lbl]
                )
                self.corrupted_map[idx] = true_lbl
                # store in DataFrame so __getitem__ sees it
                self.data_frame.at[idx, 'corrupted_label'] = wrong_lbl
        print("corrupted map",self.corrupted_map)

    def _get_label(self, idx):
        row = self.data_frame.iloc[idx, 1:1+self.num_classes].values
        return int(row.argmax())

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # load image
        img_name = self.data_frame.iloc[idx, 0]
        img_path = os.path.join(self.root, img_name)
        image   = Image.open(img_path).convert("RGB")
        # Apply transformations if any
        
        if self.transform:
            if self.clip_version=="HF":
                image = self.transform(images=image, return_tensors="pt")
                # print(image,type(image),"inside quilt sataset")
                image = image['pixel_values'].squeeze()
            else:#openai clip and open_clip
                image = self.transform(image)

        # decide whether to use corrupted label
        if idx in self.corrupted_map:
            label = int(self.data_frame.at[idx, 'corrupted_label'])
        else:
            label = self._get_label(idx)

        return image, label

class SicapDataset(Dataset):
    def __init__(self, root, train=True, victim_transform=None, surrogate_transform=None,
                 percent_data=1.0, corrupt_pct=0.0,clip_version="CLIP"):
        """
        corrupt_pct: fraction of examples whose labels
                    will be randomly “rotated” to a wrong label.
        """
        csv_file = os.path.join(
            root,
            "partition/Test/Train.xlsx" if train else "partition/Test/Test.xlsx"
        )
        self.clip_version = clip_version
        self.data_frame = pd.read_excel(csv_file)
        self.root  = os.path.join(root, "images")
        self.victim_transform = victim_transform
        self.surrogate_transform = surrogate_transform

        # sample down for % data
        
        if not (percent_data == 1.0):
            sample_size = int(len(self.data_frame) * percent_data)
            self.data_frame = self.data_frame.sample(
                n=sample_size, random_state=42
            ).reset_index(drop=True)

        self.num_classes = 4  # assuming 5 label columns
        self.corrupt_pct  = corrupt_pct
        self.corrupted_map = {}  # idx -> true_label

        if corrupt_pct > 0:
            # choose which indices to corrupt
            
            n_corrupt = int(len(self.data_frame) * corrupt_pct)
            print("Corrupt percent",corrupt_pct,"no of corrupt",n_corrupt)
            corrupt_indices = np.random.choice(
                len(self.data_frame), size=n_corrupt, replace=False
            )
            for idx in corrupt_indices:
                true_lbl = self._get_label(idx)
                # pick a WRONG label uniformly
                wrong_lbl = np.random.choice(
                    [c for c in range(self.num_classes) if c != true_lbl]
                )
                self.corrupted_map[idx] = true_lbl
                # store in DataFrame so __getitem__ sees it
                self.data_frame.at[idx, 'corrupted_label'] = wrong_lbl
        print("corrupted map",self.corrupted_map)

    def _get_label(self, idx):
        row = self.data_frame.iloc[idx, 1:1+self.num_classes].values
        return int(row.argmax())

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # load image
        img_name = self.data_frame.iloc[idx, 0]
        img_path = os.path.join(self.root, img_name)
        image   = Image.open(img_path).convert("RGB")
        # Apply transformations if any
        
        if self.victim_transform:
            if self.clip_version=="HF":
                image_victim = self.victim_transform(images=image, return_tensors="pt")
                # print(image,type(image),"inside quilt sataset")
                image_victim = image_victim['pixel_values'].squeeze()
            else:#openai clip and open_clip
                image_victim = self.victim_transform(image)

        if self.surrogate_transform:
            if self.clip_version=="HF":
                image_thief = self.surrogate_transform(images=image, return_tensors="pt")
                # print(image,type(image),"inside quilt sataset")
                image_thief = image_thief['pixel_values'].squeeze()
            else:#openai clip and open_clip
                image_thief = self.surrogate_transform(image)

        # decide whether to use corrupted label
        if idx in self.corrupted_map:
            label = int(self.data_frame.at[idx, 'corrupted_label'])
        else:
            label = self._get_label(idx)

        return image_victim, image_thief, label
    
if __name__ == "__main__":
    root = "/cache/Shivam/clipadapter/datasets/sicapv2/SICAPv2/"
    victim_transform = None
    surrogate_transform = None
    clip_version="CLIP"
    percent_training_set=1.0
    corrupt_pct=0.0

    train_dataset = SicapDataset(root=root,
    train=True,
    victim_transform = None,
    surrogate_transform = None,
    percent_data=percent_training_set,
    corrupt_pct=corrupt_pct,
    clip_version=clip_version)
    
    test_dataset = SicapDataset(
    root=root,
    train=False,
    victim_transform = None,
    surrogate_transform = None,
    percent_data=1.0,
    corrupt_pct=0.0,clip_version=clip_version )
    
    val_dataset=None

