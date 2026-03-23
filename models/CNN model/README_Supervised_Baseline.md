# Supervised Baseline Configuration (Step 1)

This document records the hyperparameters used to train the CNN from scratch (random initialization) to establish the accuracy baseline for a fully supervised network without any self-supervised pre-training.

## Training Strategy (`configs/config.yaml`)
* **Mode:** Fully Supervised (From Scratch)
* **Epochs (`num_epochs`):** `100`

## Optimizer & Hyperparameters
* **Learning Rate (`learning_rate`):** `1e-3` *(High starting rate strictly required to map a completely randomized neural network to a 10-class problem)*
* **Weight Decay (`weight_decay`):** `0`
* **Batch Size:** `256`
* **Scheduler:** Cosine Annealing

## Data Augmentations
To ensure the baseline does not trivially overfit the 12,000 training images, we applied the exact structurally-robust `v2` augmentation stack perfected during our SSL testing. These transformations aggressively mutate the image while strictly pinning the target galaxy mathematically in the center of the frame:

1. `v2.ToImage()` & `v2.ToDtype()` 
2. `v2.RandomAffine(degrees=180, scale=(1.0, 1.5))` *(Rotates and zooms IN, mathematically preventing any black zero-padding from entering the tensor)*
3. `v2.CenterCrop(144)` *(Extracts the guaranteed-centered galaxy)*
4. `v2.RandomHorizontalFlip()` & `v2.RandomVerticalFlip()`
5. `v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0)`
6. `v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))` *(Smooths out unique telescope sensor noise to prevent shortcut memorization)*
7. `v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])`

## Objective
The final validation accuracy logged at Epoch 100 for this run represents your **"Supervised Baseline"**. Step 3 (SSL Transfer) will attempt to beat or match this accuracy using the 100-epoch SimCLR representations.
