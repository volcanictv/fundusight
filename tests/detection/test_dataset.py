import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.detection.dataset import AptosDataset, build_transforms, compute_class_weights


def _make_fake_dataset(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows = []
    for i, diagnosis in enumerate([0, 2, 4]):
        id_code = f"img{i}"
        image = np.random.default_rng(i).integers(0, 255, size=(50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{id_code}.png"), image)
        rows.append({"id_code": id_code, "diagnosis": diagnosis})

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(csv_path), str(img_dir)


def test_aptos_dataset_returns_tensor_and_label(tmp_path):
    csv_path, img_dir = _make_fake_dataset(tmp_path)
    dataset = AptosDataset(csv_path, img_dir, transform=build_transforms(train=False))

    assert len(dataset) == 3
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert label == 0


def test_aptos_dataset_missing_image_raises(tmp_path):
    csv_path, img_dir = _make_fake_dataset(tmp_path)
    dataset = AptosDataset(csv_path, img_dir)
    dataset.df.loc[0, "id_code"] = "does_not_exist"

    with pytest.raises(FileNotFoundError):
        dataset[0]


def test_compute_class_weights_favors_minority_classes(tmp_path):
    csv_path, _ = _make_fake_dataset(tmp_path)
    # Duplicate the class-0 row several times to make it the dominant class.
    df = pd.read_csv(csv_path)
    df = pd.concat([df] + [df[df["diagnosis"] == 0]] * 5, ignore_index=True)
    df.to_csv(csv_path, index=False)

    weights = compute_class_weights(csv_path)
    class_labels = sorted(df["diagnosis"].unique())
    weight_by_class = dict(zip(class_labels, weights.tolist()))

    assert weight_by_class[0] < weight_by_class[2]
