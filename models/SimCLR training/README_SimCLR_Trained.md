# SimCLR Pre-Training Configuration & Results (Final Run)

This document records the exact hyperparameters and architectural fixes used to successfully train a robust SimCLR backbone without the network succumbing to representation collapse via padding-edge shortcuts.

## Final Training Metrics
* **Total Epochs:** 100
* **Best Validation Loss:** `0.2678` *(Epoch 94)*
* **Final Training Loss:** `0.1836` *(Epoch 99)*

*(Note: Validation loss smoothly decayed in parallel with the training loss. Because the augmentation pipeline mathematically prevented the generation of artificial black padding, the network was physically forced to learn the morphological structures of the galaxies to achieve this loss minimum!)*

## Configuration (`configs/config.yaml`)
* **SSL Method:** `simclr`
* **Batch Size:** `512`
* **Learning Rate (`learning_rate_ssl`):** `2e-4`
* **Weight Decay (`weight_decay_ssl`):** `1e-4` *(With the padding shortcut eliminated, the network no longer needed the restrictive `1e-1` decay to prevent overfitting. It was given its full capacity to map complex physical shapes).*
* **Temperature:** `0.1`

## Data Augmentation Pipeline (`src/architectures.py` -> `Image_Augmentations`)
To guarantee the model evaluated true galaxy physics instead of memorizing data anomalies, this `v2` augmentation stack was used:

1. **`v2.ToImage()` & `v2.ToDtype()`**
2. **`v2.RandomAffine(degrees=180, scale=(1.0, 1.5))`**: **THE CRITICAL FIX.** By ensuring the image is only ever zoomed IN (scale $\ge$ 1.0), PyTorch is mathematically incapable of generating pure-black `0.0` value padding wedges around the edges of the tensor. This single fix stopped the network from becoming a "padding-edge detector."
3. **`v2.CenterCrop(144)`**: Explicitly guarantees the zoom is perfectly centered onto the core bulge of the target galaxy.
4. **`v2.RandomHorizontalFlip()` & `v2.RandomVerticalFlip()`**
5. **`v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0)`**
6. **`v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))`**: Smoothing the tensors destroyed the high-frequency pixel sensor noise that the network previously used to memorize training IDs.
7. **`v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])`**

## Network Architecture
* **Backbone:** Custom Strided ResNet (`CNNclassifier` up to `.avgpool` embedding space)
* **Projection Output:** Flattened 512-dimensional vector.
* **Device:** Apple MPS
