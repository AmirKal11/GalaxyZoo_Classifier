# SimCLR Fine-Tuning Performance & Configuration (LP-FT Method)

This document records the exact configuration used to evaluate the 100-epoch SimCLR backbone on the downstream 10-class morphological classification task.

## Final Performance Metrics
* **Total Epochs:** 100
* **Best Validation Accuracy:** ~`81.9%`
* **Overall Macro F1-Score:** `0.7981`

### Phenomenon: Contrastive Feature Suppression
While the model slightly trailed the 85% supervised baseline from random initialization, the Confusion Matrix revealed a fascinating structural reality of Self-Supervised Learning.

Because SimCLR was aggressively trained with `GaussianBlur` and `ColorJitter` to prevent shortcut memorization on the telescope noise, it structurally generalized to **ignore high-frequency, wispy details** (which it perceived as blurry chaos). As a result:

* **Geometric Success:** The backbone was absolutely flawless at mapping dense, predictable Euclidean shapes, hitting `0.92 F1` on `Round Smooth` and `0.91 F1` on `Edge-On Bulge`.
* **Wispy Failure:** It struggled immensely with `Disturbed` (F1 `0.50`) and `Loose Spiral` (F1 `0.67`) because the high-frequency chaotic tails that define these classes were structurally suppressed during pre-training as "irrelevant blur noise."

## Configuration (`configs/config.yaml`)
* **Mode:** `loading_weights: True`, `freeze_backbone: False`
* **Batch Size:** `256`
* **Base Learning Rate (`learning_rate`):** `1e-4`
* **Weight Decay (`weight_decay`):** `1e-4`
* **Optimizer:** AdamW

## The LP-FT Optimization Strategy
To forcefully prevent the newly initialized classification head from firing massive, destructive error gradients back into the delicate SimCLR weights during the first epoch (Catastrophic Forgetting), the `train.py` script was wired to execute an automated **Linear Probe then Fine-Tune (LP-FT)** deployment:

1. **Epoch 0-19 (Linear Probe):**
   * **Action:** The entire SimCLR `ResNet` backbone was locked (`requires_grad=False`).
   * **Learning Rate:** A massive `1e-2` was manually prescribed.
   * **Result:** The classification head rapidly bent to map the 512-dimensional output vectors into 10 buckets. By Epoch 60, accuracy climbed rapidly to 80% without touching the backbone.

2. **Epoch 20-100 (Full Fine-Tune):**
   * **Action:** The backbone was un-vaulted globally (`requires_grad=True`).
   * **Learning Rate:** The optimizer cleanly transitioned to the configuration's gentle `1e-4` base learning rate.
   * **Result:** The gradients safely nudged the SimCLR layers to optimize for 10-class variance without overriding the deep positional filters.

## Saved Output
The weights resulting from this unified training pass exist at:
`models/best_model_linear_probe_masked.pth`

Its visual confusion matrix evaluation exists at:
`models/best_model_linear_probe_masked_cm.png`
