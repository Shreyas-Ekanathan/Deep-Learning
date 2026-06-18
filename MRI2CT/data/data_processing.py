from pathlib import Path
import nibabel as nib
import numpy as np
from skimage.transform import resize

def preprocess_patient(patient_dir, out_dir, split="train"):
    patient_id = patient_dir.name
    mri = nib.load(patient_dir / "mr.nii.gz").get_fdata()
    ct = nib.load(patient_dir / "ct.nii.gz").get_fdata()
    mask = nib.load(patient_dir / "mask.nii.gz").get_fdata()

    # normalization
    mri = (mri - mri.mean()) / (mri.std() + 1e-8) 
    ct = np.clip(ct, -1000, 2000)
    ct = (ct + 1000) / 3000 * 2 - 1 # [-1, 1]

    # mask
    mri = mri * mask
    ct = ct * mask

    mri_out = Path(out_dir) / split / "mri"
    ct_out  = Path(out_dir) / split / "ct"
    mri_out.mkdir(parents=True, exist_ok=True)
    ct_out.mkdir(parents=True, exist_ok=True)

    for i in range(mri.shape[2]): #examining the top down view in slices, since the data is 3d
        # we take the cross sections as each slice
        if mask[:, :, i].sum() < 100: #too sparse
            continue
        mri_slice = resize(mri[:, :, i], (128, 128), anti_aliasing=True)
        ct_slice = resize(ct[:, :, i], (128, 128), anti_aliasing=True)
        np.save(mri_out / f"{patient_id}_slice_{i:03d}.npy", mri_slice.astype(np.float32))
        np.save(ct_out  / f"{patient_id}_slice_{i:03d}.npy", ct_slice.astype(np.float32))

data_root = Path("MRI2CT/data/train_raw/brain")
patients  = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name != "overview"])

split_idx = int(0.8 * len(patients))
for p in patients[:split_idx]:
    preprocess_patient(p, "MRI2CT/data", split="train")
for p in patients[split_idx:]:
    preprocess_patient(p, "MRI2CT/data", split="test")