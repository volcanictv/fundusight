"""Phase 7: cross-dataset DR validation — IDRiD dataset loader.

Reads data/IDRi/idrid_labels.csv (id_code, diagnosis -- same 0-4 ICDR
severity scale as APTOS, so the existing DR checkpoint's class indices
transfer directly, no relabeling needed) and matching JPEGs in
data/IDRi/Imagenes/Imagenes/. Deliberately not reusing
src.detection.dataset.AptosDataset -- that class hardcodes a .png extension
for APTOS's PNGs, while IDRiD ships JPEGs, and IDRiD's CSV additionally
carries a "Risk of macular edema" column plus several unnamed empty columns
(artifacts of a merged train+test CSV) that AptosDataset doesn't expect.
Otherwise the loader is functionally identical to AptosDataset.

Both the official IDRiD training set (id_code like IDRiD_001) and its
official test set (id_code like IDRiD_001test -- IDRiD's train and test
numbering both restart at 1, so the "test" suffix disambiguates them once
pooled into a single image folder) are present; this loader doesn't
distinguish between them since Phase 7's cross-dataset check evaluates the
whole labeled set as one held-out cross-dataset test, not a train/val/test
split of its own -- the model was never trained on any IDRiD image either
way.
"""

import os

import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms


class IDRiDDataset(Dataset):
    """Reads IDRiD's (id_code, diagnosis) label CSV and matching JPEGs."""

    def __init__(self, csv_path: str, img_dir: str, transform: transforms.Compose | None = None):
        self.df = pd.read_csv(csv_path, usecols=["id_code", "diagnosis"])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, f"{row['id_code']}.jpg")
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image, int(row["diagnosis"])
