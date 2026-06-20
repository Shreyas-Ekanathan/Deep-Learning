import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from glob import glob

class MRI_CT_DATASET(Dataset):
    def __init__(self, mri_dir, ct_dir, mode="paired"):
        self.mri_files = sorted(glob(f"{mri_dir}/*.npy"))
        self.ct_files  = sorted(glob(f"{ct_dir}/*.npy"))
        self.mode = mode

    def __len__(self):
        return len(self.mri_files)

    def __getitem__(self, idx):
        mri = torch.tensor(np.load(self.mri_files[idx]), dtype=torch.float32).unsqueeze(0)
        ct  = torch.tensor(np.load(self.ct_files[idx]),  dtype=torch.float32).unsqueeze(0)
        if self.mode == "paired":
            return mri, ct
        elif self.mode == "mri":
            return mri
        elif self.mode == "ct":
            return ct