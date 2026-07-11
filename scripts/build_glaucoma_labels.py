"""Merge REFUGE2's glaucoma labels (REFUGE2/Refuge2_test.csv -- despite the
name, it covers images from all three REFUGE2 domains: train/g-n-prefixed,
val/V-prefixed, test/T-prefixed) with SMDG-19's REFUGE1-train/REFUGE1-val
rows (SMDG-19 has no REFUGE2-test-domain coverage -- REFUGE1 predates the
REFUGE2 test-domain expansion).

Conflict resolution: Refuge2_test.csv has 68 duplicate filenames, 15 of
which disagree on the label. Where SMDG-19 covers the image (train/val
domain only), its label resolves the conflict; otherwise the row is
dropped rather than guessed at. SMDG-19 also fills in REFUGE1 images
Refuge2_test.csv never had a row for at all.

Investigation/prep only -- doesn't change any pipeline code, and Phase 7
training work hasn't started. Run with:

    .venv\\Scripts\\python.exe scripts\\build_glaucoma_labels.py
"""

import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFUGE2_CSV = os.path.join(PROJECT_ROOT, "REFUGE2", "Refuge2_test.csv")
SMDG_CSV = os.path.join(
    PROJECT_ROOT,
    "SMDG, A Standardized Fundus Glaucoma Dataset",
    "metadata - standardized.csv",
)
OUT_CSV = os.path.join(PROJECT_ROOT, "REFUGE2", "glaucoma_labels_merged.csv")


def _domain_for_filename(name: str) -> str:
    prefix = name[0]
    if prefix in ("g", "n"):
        return "train"
    if prefix == "V":
        return "val"
    if prefix == "T":
        return "test"
    raise ValueError(f"unrecognized filename prefix: {name!r}")


def load_refuge2_labels() -> tuple[dict[str, int], set[str]]:
    """Returns (name -> label for non-conflicting names, set of conflicting names)."""
    df = pd.read_csv(REFUGE2_CSV)
    df["basename"] = df["ImageName"].apply(lambda p: p.rsplit("/", 1)[-1])

    per_name_labels: dict[str, set[int]] = {}
    for name, label in zip(df["basename"], df["glaucoma"]):
        per_name_labels.setdefault(name, set()).add(int(label))

    resolved = {name: next(iter(labels)) for name, labels in per_name_labels.items() if len(labels) == 1}
    conflicts = {name for name, labels in per_name_labels.items() if len(labels) > 1}
    return resolved, conflicts


def load_smdg_refuge1_labels() -> dict[str, int]:
    df = pd.read_csv(SMDG_CSV)
    refuge1 = df[df["names"].str.startswith(("REFUGE1-train", "REFUGE1-val"), na=False)]

    train_rows = refuge1[refuge1["names"].str.startswith("REFUGE1-train")]
    val_rows = refuge1[refuge1["names"].str.startswith("REFUGE1-val")]
    assert len(train_rows) == 400, f"expected 400 REFUGE1-train rows, got {len(train_rows)}"
    assert len(val_rows) == 400, f"expected 400 REFUGE1-val rows, got {len(val_rows)}"
    for label, split_df in (("train", train_rows), ("val", val_rows)):
        counts = split_df["types"].value_counts().to_dict()
        assert counts.get(0) == 360 and counts.get(1) == 40, (
            f"REFUGE1-{label} label distribution {counts} != expected 360/40"
        )

    return dict(zip(refuge1["original_name"], refuge1["types"].astype(int)))


def main() -> None:
    refuge2_resolved, refuge2_conflicts = load_refuge2_labels()
    smdg_labels = load_smdg_refuge1_labels()

    print(f"REFUGE2 csv: {len(refuge2_resolved)} non-conflicting names, {len(refuge2_conflicts)} conflicting names")
    print(f"SMDG-19 REFUGE1-train/val: {len(smdg_labels)} labeled images")

    final: dict[str, tuple[int, str]] = {}
    for name, label in refuge2_resolved.items():
        final[name] = (label, "refuge2")

    resolved_conflicts = 0
    dropped_conflicts = 0
    for name in refuge2_conflicts:
        if name in smdg_labels:
            final[name] = (smdg_labels[name], "smdg_resolved_conflict")
            resolved_conflicts += 1
        else:
            dropped_conflicts += 1
    print(f"conflicts resolved via SMDG-19: {resolved_conflicts}, dropped unresolved: {dropped_conflicts}")

    added_from_smdg = 0
    for name, label in smdg_labels.items():
        if name not in final:
            final[name] = (label, "smdg_added")
            added_from_smdg += 1
    print(f"new images added from SMDG-19 (no REFUGE2 csv row at all): {added_from_smdg}")

    rows = []
    for name, (label, source) in final.items():
        rows.append({"filename": name, "domain": _domain_for_filename(name), "glaucoma": label, "source": source})
    merged = pd.DataFrame(rows).sort_values(["domain", "filename"]).reset_index(drop=True)
    merged.to_csv(OUT_CSV, index=False)

    print(f"\nFinal merged label count: {len(merged)}")
    print("\nCoverage by original domain (of 400 possible each):")
    for domain in ("train", "val", "test"):
        sub = merged[merged["domain"] == domain]
        n_glaucoma = int((sub["glaucoma"] == 1).sum())
        print(f"  {domain}: {len(sub)}/400  ({n_glaucoma} glaucoma / {len(sub) - n_glaucoma} non-glaucoma)")

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
