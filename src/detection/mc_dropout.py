"""Monte-Carlo Dropout uncertainty for the EfficientNet-B0 classifiers.

The three classifiers (DR / glaucoma / AMD) each keep the backbone's
Dropout(p=0.2) before the linear head -- detection/model.py replaces only
classifier[1] (the Linear), so classifier[0] (the Dropout) survives. Leaving
that dropout ACTIVE at inference and running several forward passes turns the
network into a cheap approximate-Bayesian ensemble: the spread of the predicted
class's probability across passes is a usable epistemic-uncertainty estimate
(Gal & Ghahramani, 2016). It needs no retraining.

This is an APPROXIMATION, not calibrated probability -- it captures the model's
own disagreement under dropout, not true predictive calibration, and is
surfaced in the UI/report worded as such.

The passes are drawn in ONE batched forward (the input replicated n_samples
times). Dropout masks are sampled independently per batch row, so a batch of N
yields N independent samples in a single pass -- far cheaper than N separate
forwards. BatchNorm stays in eval mode (running stats, not batch stats), so
batching does not shift the point prediction.
"""

import numpy as np
import torch

# 20 passes is the usual sweet spot in the MC-dropout literature: enough for a
# stable standard deviation, cheap enough to stay well within the pipeline's
# time budget as a single batched forward.
DEFAULT_MC_SAMPLES = 20


def enable_dropout(model: torch.nn.Module) -> None:
    """Put ONLY Dropout layers into train() mode, leaving BatchNorm and
    everything else in eval() -- so dropout samples, but running stats (and the
    deterministic point prediction) are unchanged."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_probabilities(model: torch.nn.Module, tensor: torch.Tensor, n_samples: int) -> np.ndarray:
    """(n_samples, num_classes) array of softmax probabilities from stochastic
    forward passes with dropout active.

    `tensor` is a single already-preprocessed input of shape (1, C, H, W).
    Restores the model to eval() before returning, so callers sharing a cached
    model (report/pipeline.py) are unaffected.
    """
    enable_dropout(model)
    try:
        batch = tensor.repeat(n_samples, 1, 1, 1)
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    finally:
        model.eval()
    return probabilities


def predicted_class_std(
    model: torch.nn.Module, tensor: torch.Tensor, class_idx: int, n_samples: int = DEFAULT_MC_SAMPLES
) -> float:
    """1-sigma spread of `class_idx`'s probability across MC-dropout passes --
    the single number the report/app show as '± x%'."""
    samples = mc_dropout_probabilities(model, tensor, n_samples)
    return float(samples[:, class_idx].std())
