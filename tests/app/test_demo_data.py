import cv2
import numpy as np
import pandas as pd

from src.app import demo_data
from src.detection.model import SEVERITY_LABELS


def test_list_demo_images_returns_empty_when_bundle_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_data, "_IMG_DIR", str(tmp_path / "no_such_images"))
    monkeypatch.setattr(demo_data, "_CSV_PATH", str(tmp_path / "no_such_images" / "labels.csv"))

    assert demo_data.list_demo_images() == []


def test_list_demo_images_reads_bundled_images(tmp_path, monkeypatch):
    img_dir = tmp_path / "demo_images"
    img_dir.mkdir()
    csv_path = img_dir / "labels.csv"

    rows = [("img_b", 1), ("img_a", 0), ("img_c", 2)]
    for id_code, _ in rows:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{id_code}.jpg"), image)
    pd.DataFrame(rows, columns=["id_code", "diagnosis"]).to_csv(csv_path, index=False)

    monkeypatch.setattr(demo_data, "_IMG_DIR", str(img_dir))
    monkeypatch.setattr(demo_data, "_CSV_PATH", str(csv_path))

    images = demo_data.list_demo_images()

    assert [item["diagnosis"] for item in images] == [0, 1, 2]  # sorted by severity
    assert images[0]["label"] == SEVERITY_LABELS[0]
    assert images[0]["id_code"] == "img_a"


def test_list_demo_images_skips_rows_with_missing_image_file(tmp_path, monkeypatch):
    img_dir = tmp_path / "demo_images"
    img_dir.mkdir()
    csv_path = img_dir / "labels.csv"
    pd.DataFrame([("missing_img", 0)], columns=["id_code", "diagnosis"]).to_csv(csv_path, index=False)

    monkeypatch.setattr(demo_data, "_IMG_DIR", str(img_dir))
    monkeypatch.setattr(demo_data, "_CSV_PATH", str(csv_path))

    assert demo_data.list_demo_images() == []


def test_load_demo_image_reads_bgr_array(tmp_path):
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), np.full((5, 5, 3), 200, dtype=np.uint8))

    image = demo_data.load_demo_image(str(path))

    assert image.shape == (5, 5, 3)
