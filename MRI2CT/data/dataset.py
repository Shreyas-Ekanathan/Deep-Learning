import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from glob import glob

class MRI_CT_DATASET(Dataset):
    def __init__(self, mri_dir, ct_dir):
        self.mri_files = sorted(glob(f"{mri_dir}/*.npy"))
        self.ct_files  = sorted(glob(f"{ct_dir}/*.npy"))

    def __len__(self):
        return len(self.mri_files)

    def __getitem__(self, idx):
        mri = np.load(self.mri_files[idx])
        ct  = np.load(self.ct_files[idx])
        mri = torch.tensor(mri, dtype=torch.float32).unsqueeze(0)  #for channel
        ct  = torch.tensor(ct,  dtype=torch.float32).unsqueeze(0)
        return mri, ct
