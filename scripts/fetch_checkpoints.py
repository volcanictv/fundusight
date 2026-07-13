"""Deployment: download trained checkpoints from a GitHub Release (CLI entry
point). The app calls the same function itself at startup (see
src/app/checkpoints.py) — this script exists for pre-fetching outside the
app, e.g. as a deployment build step.

Usage:
    .venv\\Scripts\\python.exe scripts\\fetch_checkpoints.py
    .venv\\Scripts\\python.exe scripts\\fetch_checkpoints.py --tag v1.2.0 --repo volcanictv/fundusight

Uploading the assets (one-time, whenever a checkpoint is retrained) is a
manual step, not part of this script:
    gh release create v1.2.0 checkpoints/dr_efficientnet_b0.pth \\
        checkpoints/glaucoma_efficientnet_b0.pth checkpoints/amd_efficientnet_b0.pth \\
        checkpoints/vessel_unet.pth checkpoints/optic_disc_unet.pth \\
        checkpoints/disc_locator.pth \\
        --title "v1.2.0" --notes "Fundusight v1.2.0 checkpoints (vascular convergence prior; Stage 6.0 disc locator; glaucoma retrained on the new ONH crops)"

RETRAINED A CHECKPOINT? CUT A NEW TAG -- don't overwrite an existing release's
asset. The deployed app on `master` fetches from whatever DEFAULT_TAG that
branch's code carries, so clobbering a published asset swaps the weights under
code that is still live. That is how you get a silent train/inference mismatch
in production (v1.1.0's glaucoma checkpoint classifies an ONH crop; v1.0.0's
classified a full fundus photo -- feeding either one the other's input yields
confident, meaningless probabilities rather than an error). A new tag + a
DEFAULT_TAG bump in the same commit keeps weights and code shipping together.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app.checkpoints import DEFAULT_REPO, DEFAULT_TAG, fetch_checkpoints

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub owner/repo (default: {DEFAULT_REPO})")
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"Release tag (default: {DEFAULT_TAG})")
    args = parser.parse_args()

    print(f"Checking checkpoints/ against {args.repo}@{args.tag} ...")
    downloaded = fetch_checkpoints(repo=args.repo, tag=args.tag)
    if downloaded:
        print(f"Downloaded: {', '.join(downloaded)}")
    else:
        print("All checkpoints already present — nothing to download.")
