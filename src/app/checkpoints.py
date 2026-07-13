"""Deployment: fetch trained checkpoints from a GitHub Release at runtime.

Checkpoints are gitignored (see CLAUDE.md — trained weights are large and
never committed), so a fresh clone or a fresh cloud deployment starts with
none in checkpoints/. `main.py` calls `fetch_checkpoints()` once at startup
(cached via st.cache_resource so it only runs once per process) to pull the
five checkpoints inference actually needs from a GitHub Release's attached
assets; existing files are left untouched, so a local dev machine that
already trained its own checkpoints makes zero network calls.

`scripts/fetch_checkpoints.py` is the same download as a standalone CLI, for
pre-fetching outside the app (e.g. a deployment build step).
"""

import os

import requests

DEFAULT_REPO = "volcanictv/fundusight"
# v1.1.0 ships the ONH-cropped glaucoma checkpoint (2026-07-13). The tag bump
# is NOT cosmetic and must not be reverted independently of the code: that
# checkpoint classifies an optic-nerve-head CROP, while v1.0.0's classified a
# full fundus photo. Pairing v1.1.0's weights with pre-fix inference code (or
# vice versa) feeds the model an image of a kind it never trained on -- a
# silent train/inference mismatch that yields confident, meaningless glaucoma
# probabilities rather than an error. Weights and code ship together; see
# src/detection/onh_crop.py.
DEFAULT_TAG = "v1.1.0"

# Matches src/detection/infer.py, glaucoma_infer.py, amd_infer.py, and
# src/segmentation/vessel_infer.py, optic_disc_infer.py's DEFAULT_WEIGHTS_PATH
# constants — the only five files a deployed app needs to load.
# optic_disc_unet.provisional_domainsplit.pth (kept for comparison, see
# ROADMAP.md's Phase 6) is intentionally excluded: nothing in src/ loads it.
CHECKPOINT_FILES = [
    "dr_efficientnet_b0.pth",
    "glaucoma_efficientnet_b0.pth",
    "amd_efficientnet_b0.pth",
    "vessel_unet.pth",
    "optic_disc_unet.pth",
]

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINTS_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")


def fetch_checkpoints(repo: str = DEFAULT_REPO, tag: str = DEFAULT_TAG, dest_dir: str = CHECKPOINTS_DIR) -> list[str]:
    """Download any checkpoint missing from `dest_dir` from the GitHub Release
    at `repo`/`tag`. Returns the filenames actually downloaded (empty if
    everything was already present) — existing files are never re-fetched.

    A failed download (offline, release not published yet, rate-limited) is
    swallowed per-file rather than raised: the rest of the pipeline already
    treats a missing checkpoint as "fall back to the classical baseline" or
    "section unavailable" rather than a hard error (see vessel_infer.py /
    optic_disc_infer.py's *_auto() fallback convention), so a fetch failure
    should degrade the same way, not crash the whole app on startup.
    """
    os.makedirs(dest_dir, exist_ok=True)
    downloaded = []

    for filename in CHECKPOINT_FILES:
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path):
            continue

        url = f"https://github.com/{repo}/releases/download/{tag}/{filename}"
        tmp_path = dest_path + ".part"
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            os.replace(tmp_path, dest_path)
        except requests.exceptions.RequestException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            continue

        downloaded.append(filename)

    return downloaded
