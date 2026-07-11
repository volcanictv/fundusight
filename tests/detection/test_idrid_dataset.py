import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.detection.dataset import build_transforms
from src.detection.idrid_dataset import IDRiDDataset


def _make_fake_idrid(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows = []
    # id_codes mirror the real CSV's mix of plain and "test"-suffixed codes.
    for i, (id_code, diagnosis) in enumerate([("IDRiD_001", 0), ("IDRiD_002test", 3)]):
        image = np.random.default_rng(i).integers(0, 255, size=(50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{id_code}.jpg"), image)
        rows.append({"id_code": id_code, "diagnosis": diagnosis, "Risk of macular edema ": 1, "Unnamed: 3": None})

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(csv_path), str(img_dir)


def test_idrid_dataset_returns_tensor_and_label(tmp_path):
    csv_path, img_dir = _make_fake_idrid(tmp_path)
    dataset = IDRiDDataset(csv_path, img_dir, transform=build_transforms(train=False))

    assert len(dataset) == 2
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert label == 0

    # The "test"-suffixed id_code (disambiguating IDRiD's official test set,
    # which restarts numbering at 1 same as the training set) must resolve
    # to its own distinct file, not collide with the unsuffixed one.
    image, label = dataset[1]
    assert label == 3


def test_idrid_dataset_ignores_extra_csv_columns(tmp_path):
    csv_path, img_dir = _make_fake_idrid(tmp_path)
    dataset = IDRiDDataset(csv_path, img_dir)

    assert list(dataset.df.columns) == ["id_code", "diagnosis"]


def test_idrid_dataset_missing_image_raises(tmp_path):
    csv_path, img_dir = _make_fake_idrid(tmp_path)
    dataset = IDRiDDataset(csv_path, img_dir)
    dataset.df.loc[0, "id_code"] = "does_not_exist"

    with pytest.raises(FileNotFoundError):
        dataset[0]
