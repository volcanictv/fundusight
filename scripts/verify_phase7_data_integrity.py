"""Integrity check for the Phase 7 datasets (ADAM, IDRiD, REFUGE2 glaucoma
labels) before any training code is written: every image actually decodes,
every label file's rows match real files on disk, and counts are what we
expect from the earlier presence check.

Investigation only -- doesn't change any pipeline code. Run with:

    .venv\\Scripts\\python.exe scripts\\verify_phase7_data_integrity.py
"""

import os

import cv2
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _check_images(paths: list[str], label: str) -> list[str]:
    """Returns list of paths that failed to decode or had a degenerate shape."""
    bad = []
    for path in paths:
        img = cv2.imread(path)
        if img is None or img.size == 0 or img.shape[0] < 10 or img.shape[1] < 10:
            bad.append(path)
    print(f"  {label}: {len(paths)} checked, {len(bad)} corrupt/unreadable")
    return bad


def check_adam() -> None:
    print("\n=== ADAM (AMD) ===")
    root = os.path.join(PROJECT_ROOT, "ADAM", "Training400")

    amd_dir = os.path.join(root, "AMD")
    non_amd_dir = os.path.join(root, "Non-AMD")
    disc_masks_dir = os.path.join(root, "Disc_Masks")

    amd_files = sorted(os.path.join(amd_dir, f) for f in os.listdir(amd_dir))
    non_amd_files = sorted(os.path.join(non_amd_dir, f) for f in os.listdir(non_amd_dir))
    mask_files = sorted(os.path.join(disc_masks_dir, f) for f in os.listdir(disc_masks_dir))

    assert len(amd_files) == 89, f"expected 89 AMD images, got {len(amd_files)}"
    assert len(non_amd_files) == 311, f"expected 311 Non-AMD images, got {len(non_amd_files)}"
    assert len(mask_files) == 400, f"expected 400 disc masks, got {len(mask_files)}"

    bad = []
    bad += _check_images(amd_files, "AMD/ images")
    bad += _check_images(non_amd_files, "Non-AMD/ images")
    bad += _check_images(mask_files, "Disc_Masks/")

    xlsx_path = os.path.join(root, "Fovea_location.xlsx")
    fovea_df = pd.read_excel(xlsx_path)
    assert len(fovea_df) == 400, f"expected 400 rows in Fovea_location.xlsx, got {len(fovea_df)}"
    all_image_names = {os.path.basename(f) for f in amd_files + non_amd_files}
    missing_from_disk = set(fovea_df["imgName"]) - all_image_names
    print(f"  Fovea_location.xlsx: 400 rows, {len(missing_from_disk)} reference a missing image file")
    if missing_from_disk:
        bad.append(f"Fovea_location.xlsx entries with no matching image: {sorted(missing_from_disk)[:10]}")

    if bad:
        print(f"  ADAM ISSUES: {bad}")
    else:
        print("  ADAM: all checks passed")


def check_idrid() -> None:
    print("\n=== IDRiD ===")
    root = os.path.join(PROJECT_ROOT, "data", "IDRi")
    img_dir = os.path.join(root, "Imagenes", "Imagenes")
    csv_path = os.path.join(root, "idrid_labels.csv")

    labels_df = pd.read_csv(csv_path)
    image_files = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))

    print(f"  labels csv: {len(labels_df)} rows, images on disk: {len(image_files)}")
    assert len(labels_df) == len(image_files), "label row count and image file count don't match"

    on_disk_ids = {os.path.splitext(os.path.basename(f))[0] for f in image_files}
    csv_ids = set(labels_df["id_code"])
    missing_images = csv_ids - on_disk_ids
    orphan_images = on_disk_ids - csv_ids
    print(f"  csv ids with no matching image file: {len(missing_images)}")
    print(f"  image files with no matching csv row: {len(orphan_images)}")

    diagnosis_range_ok = labels_df["diagnosis"].between(0, 4).all()
    edema_col = [c for c in labels_df.columns if "macular edema" in c.lower()][0]
    edema_range_ok = labels_df[edema_col].between(0, 2).all()
    print(f"  diagnosis values in [0,4]: {diagnosis_range_ok}, edema-risk values in [0,2]: {edema_range_ok}")

    bad = _check_images(image_files, "IDRiD images")

    issues = list(missing_images) + list(orphan_images) + bad
    if not diagnosis_range_ok or not edema_range_ok:
        issues.append("out-of-range label values")
    if issues:
        print(f"  IDRiD ISSUES: {issues[:10]}{'...' if len(issues) > 10 else ''}")
    else:
        print("  IDRiD: all checks passed")


def check_refuge2_glaucoma() -> None:
    print("\n=== REFUGE2 (for glaucoma classifier) ===")
    refuge_root = os.path.join(PROJECT_ROOT, "REFUGE2")
    merged_csv = os.path.join(refuge_root, "glaucoma_labels_merged.csv")
    merged_df = pd.read_csv(merged_csv)
    print(f"  glaucoma_labels_merged.csv: {len(merged_df)} rows")

    domain_to_folder = {"train": "train", "val": "val", "test": "test"}
    bad_paths = []
    missing_files = []
    for _, row in merged_df.iterrows():
        folder = domain_to_folder[row["domain"]]
        img_path = os.path.join(refuge_root, folder, "images", row["filename"])
        if not os.path.isfile(img_path):
            missing_files.append(img_path)
    print(f"  merged-csv rows with no matching image file on disk: {len(missing_files)}")

    all_referenced = [
        os.path.join(refuge_root, domain_to_folder[row["domain"]], "images", row["filename"])
        for _, row in merged_df.iterrows()
        if os.path.isfile(os.path.join(refuge_root, domain_to_folder[row["domain"]], "images", row["filename"]))
    ]
    bad_paths = _check_images(all_referenced, "referenced REFUGE2 images")

    dup_filenames = merged_df["filename"][merged_df["filename"].duplicated()].tolist()
    print(f"  duplicate filenames within merged csv: {len(dup_filenames)}")

    issues = missing_files + bad_paths + dup_filenames
    if issues:
        print(f"  REFUGE2 glaucoma ISSUES: {issues[:10]}{'...' if len(issues) > 10 else ''}")
    else:
        print("  REFUGE2 glaucoma labels: all checks passed")


def check_smdg19_metadata() -> None:
    print("\n=== SMDG-19 metadata (used only for its labels, not images) ===")
    csv_path = os.path.join(
        PROJECT_ROOT, "SMDG, A Standardized Fundus Glaucoma Dataset", "metadata - standardized.csv"
    )
    df = pd.read_csv(csv_path)
    refuge1 = df[df["names"].str.startswith(("REFUGE1-train", "REFUGE1-val"), na=False)]
    print(f"  metadata csv: {len(df)} total rows, {len(refuge1)} REFUGE1-train/val rows")
    dup_original_names = refuge1["original_name"][refuge1["original_name"].duplicated()].tolist()
    print(f"  duplicate original_name within REFUGE1 subset: {len(dup_original_names)}")
    if dup_original_names:
        print(f"  SMDG-19 ISSUES: duplicates {dup_original_names[:10]}")
    else:
        print("  SMDG-19 metadata: all checks passed")


if __name__ == "__main__":
    check_adam()
    check_idrid()
    check_refuge2_glaucoma()
    check_smdg19_metadata()
