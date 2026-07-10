"""One-off diagnostic: is the CDR over-estimation bias (see
investigate_cdr_segmentation_bias.py: predicted disc/cup diameters ~97-99px
larger than ground truth, even with a perfect Stage 6.1 crop) driven by
genuine AREA over-segmentation, or by an irregular/elongated predicted
shape inflating the bounding-box-based vertical extent _vertical_extent()
uses, disproportionately to actual area overlap? These call for very
different fixes -- area over-segmentation implies the model itself is
miscalibrated (needs a confidence-threshold recalibration or a retrain);
a shape artifact implies compute_cdr()'s extent measurement (or the
cleanup step) needs to be more robust to a thin protrusion/tail on an
otherwise reasonably-sized blob, no retraining involved.

Investigation only -- doesn't change any pipeline code. Run with:

    .venv\\Scripts\\python.exe scripts\\investigate_cdr_shape_vs_area_bias.py
"""

import os
import sys

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.segmentation import optic_disc
from src.segmentation.optic_disc_dataset import DISC_ROI_WIDTH, OpticDiscDataset, build_pairs
from src.segmentation.optic_disc_infer import DEFAULT_WEIGHTS_PATH, load_optic_disc_model

REFUGE_ROOT = os.path.join(PROJECT_ROOT, "REFUGE2")


def _horizontal_extent(mask: np.ndarray) -> int:
    cols_with_mask = np.any(mask, axis=0)
    if not cols_with_mask.any():
        return 0
    col_indices = np.nonzero(cols_with_mask)[0]
    return int(col_indices[-1] - col_indices[0] + 1)


def _bbox_fill_ratio(mask: np.ndarray) -> float:
    """area / (vertical_extent * horizontal_extent) -- close to ~0.78 for a
    filled ellipse/circle in its own bounding box, much lower for an
    irregular blob with a thin protrusion stretching one dimension far
    past where most of the mask's area actually is.
    """
    v = optic_disc._vertical_extent(mask)
    h = _horizontal_extent(mask)
    if v == 0 or h == 0:
        return 0.0
    return float(mask.sum()) / (v * h)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_optic_disc_model(DEFAULT_WEIGHTS_PATH, device=str(device))

    test_pairs = build_pairs(REFUGE_ROOT)["test"]
    dataset = OpticDiscDataset(test_pairs, roi_width=DISC_ROI_WIDTH, train=False)
    print(f"Comparing area vs. bounding-box extent for pred/gt disc+cup on {len(dataset)} images...")

    disc_area_ratios, cup_area_ratios = [], []
    disc_diam_ratios, cup_diam_ratios = [], []
    pred_disc_fill, gt_disc_fill = [], []
    pred_cup_fill, gt_cup_fill = [], []

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            input_tensor, target = dataset[i]
            input_tensor = input_tensor.unsqueeze(0).to(device)

            logits = model(input_tensor)
            predicted_class = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_disc, pred_cup = optic_disc.clean_disc_cup_masks(predicted_class != 0, predicted_class == 2)

            target_np = target.numpy()
            gt_disc, gt_cup = target_np != 0, target_np == 2

            if gt_disc.sum() > 0 and pred_disc.sum() > 0:
                disc_area_ratios.append(pred_disc.sum() / gt_disc.sum())
                disc_diam_ratios.append(optic_disc._vertical_extent(pred_disc) / max(optic_disc._vertical_extent(gt_disc), 1))
            if gt_cup.sum() > 0 and pred_cup.sum() > 0:
                cup_area_ratios.append(pred_cup.sum() / gt_cup.sum())
                cup_diam_ratios.append(optic_disc._vertical_extent(pred_cup) / max(optic_disc._vertical_extent(gt_cup), 1))

            pred_disc_fill.append(_bbox_fill_ratio(pred_disc))
            gt_disc_fill.append(_bbox_fill_ratio(gt_disc))
            pred_cup_fill.append(_bbox_fill_ratio(pred_cup))
            gt_cup_fill.append(_bbox_fill_ratio(gt_cup))

    print(f"\n=== Area vs. bounding-box-extent comparison, {len(dataset)} images ===")
    print(f"mean disc AREA ratio (pred/gt)     = {np.mean(disc_area_ratios):.3f}")
    print(f"mean disc DIAMETER ratio (pred/gt) = {np.mean(disc_diam_ratios):.3f}")
    print(f"mean cup AREA ratio (pred/gt)      = {np.mean(cup_area_ratios):.3f}")
    print(f"mean cup DIAMETER ratio (pred/gt)  = {np.mean(cup_diam_ratios):.3f}")

    print(f"\nmean bbox-fill-ratio, predicted disc = {np.mean(pred_disc_fill):.3f}   ground-truth disc = {np.mean(gt_disc_fill):.3f}")
    print(f"mean bbox-fill-ratio, predicted cup  = {np.mean(pred_cup_fill):.3f}   ground-truth cup  = {np.mean(gt_cup_fill):.3f}")
    print("(lower fill ratio = more irregular/elongated shape relative to its own bounding box)")


if __name__ == "__main__":
    main()
