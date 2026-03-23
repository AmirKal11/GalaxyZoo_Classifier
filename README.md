# GalaxyZoo Morphology Classifier

This repository provides a deep learning framework for classifying galaxy morphologies using the Galaxy Zoo dataset. The project implements a refined pipeline where self-supervised pre-training is used to enhance the performance of a supervised classification backbone.

## Directory Structure

The repository is organized to separate data handling, model architecture, and training logic:

```text
GalaxyZoo_Classifier/
├── configs/
│   └── config.yaml          # Hyperparameters and training configurations
├── data/
│   └── Galaxy10_DECals.h5   # HDF5 dataset containing galaxy images and labels
├── models/                  # Weights, logs, and performance visualizations
│   ├── CNN model/           # Supervised baseline outputs
│   ├── SimCLR training/     # SSL backbone and fine-tuned results
│   └── feature visualizations/ # Visualization of the learned features
├── src/                     # Source code directory
│   ├── architectures.py     # CNN and SimCLR backbone definitions
│   ├── dataset.py           # PyTorch Dataset for .h5 file streaming
│   ├── train.py             # Main script for Supervised and LP-FT training
│   ├── train_ssl.py         # SimCLR contrastive pre-training script
│   ├── utils.py             # Utilities for plotting and reporting
│   ├── visualize_augs.py    # Visualization of data augmentation logic
│   ├── visualize_cm.py      # Confusion matrix generation
│   └── visualize_features.py # Feature space analysis
├── Main_analysis.ipynb      # End-to-end comparative analysis
└── requirements.txt         # Project dependencies
```

## The Pipeline

The project follows a specific three-step scientific process:

### 1. Baseline CNN Training
A baseline CNN is trained using a specific set of augmentations. These augmentations (rotations, flips, color jitters, crops and gaussian blurs) are designed to improve the model's ability to generalize across different galactic orientations and lighting conditions. More details about these augmentations are expressed below.

### 2. SimCLR Pre-training
The SimCLR framework is trained using the **exact same set of augmentations** as the baseline CNN. This ensures a controlled comparison between the supervised baseline and the self-supervised approach. The model learns to map augmented views of the same galaxy to similar points in a latent space.

### 3. LP-FT Training Logic
To maximize the utility of the SimCLR-pretrained backbone, the pipeline implements a two-phase **Linear Probing then Full Fine-Tuning (LP-FT)** strategy:

* **Phase 1: Linear Probing (Epochs $0$–$20$):**
    * The pre-trained backbone is frozen (`requires_grad = False`), and only the fully connected (`fc`) classification head is trained.
    * A higher learning rate (starting at $10^{-2}$) is used to "warm up" the head without distorting the pre-trained features.
* **Phase 2: Full Fine-Tuning (Epochs $20+$):**
    * The entire network is unfrozen (`requires_grad = True`) to allow for end-to-end optimization.
    * The learning rate is reset to a lower base rate (e.g., $10^{-4}$ or $10^{-5}$) to subtly refine the backbone features for morphological classification.
* **Optimization:** Both phases utilize the **AdamW** optimizer combined with a **Cosine Annealing** learning rate scheduler.


### 4. Feature visualization

Although it isn't a mathematically rigorous method, the feature visualization helped identify whether the model was learning meaningful features or 'shortcut features' such as noise patterns or padding edges. Three visualization methods were used: 
* **GradCam** - the most intuitive method, it highlights the regions of the image that the model is focusing on when making a prediction. This helped us visualize and confirm that the model was focusing on the galaxy itself, and not the padding artifacts. It turned out to be especially useful in the SimCLR training process as the algorithm is very sensitive to the choice of augmentations. 
* **FeatureMaps** - this shows the evolution of the feature maps through the layers of the CNN. It helped us visualize the hierarchical feature learning of the model. 
* **Nearest Neighbors** - this shows the most similar images in the dataset to a given image in the learned feature space. It helped us visualize the feature space of the model and identify whether the model was learning meaningful features. Furthermore, it helped us identify 'problematic' classes, such as disturbed galaxies that don't have a clear morphology. 


---


## Training Methods

### Data Augmentation Consistency
As mentioned above, the augmentations consisted of rotations, flips, color jitters, crops and GaussianBlurs. 
This set of augmentations was carefully chosen to preserve the morphological features of the galaxies while introducing enough variability to prevent overfitting. 

For example - an aggresive gaussian blurring could blur the spiral arms of the galaxy or a random crop could cut off the galaxy's bulge. Additionally, the color of the galaxy is a strong indicator to it's age and star formation rate, so the color jitter is kept minimal to preserve this information. 

The project maintains strict consistency by using identical augmentation parameters for both the baseline supervised run and the SimCLR pre-training. This isolates the effect of the self-supervised objective on the model's final performance.

The performed augmentations can be viewed in 'models/augmentations_preview.png'


### Loss Function: NT-Xent
For the SimCLR phase, the project utilizes the **Normalized Temperature-scaled Cross-Entropy (NT-Xent)** loss.
* **Temperature ($\tau$):** Set to 0.1 for high-contrast feature mapping.
* **Optimization:** Calculated across a vectorized $2N \times 2N$ similarity matrix.

### Optimization & Fine-Tuning Schedule
* **Optimizer:** AdamW.
* **Warmup Phase:** Fine-tuning begins with a linear warmup to prevent the pre-trained weights from being distorted by large initial gradients during the switch to supervised classification.
* **Decay:** Cosine decay is applied post-warmup for precision.

---

## Evaluation
Success is measured by comparing the final accuracy of the fine-tuned SSL model against the initial Baseline CNN. This determines the "performance gain" provided by contrastive pre-training on astrophysical data.



## Discussion and Results

The two models (Baseline CNN and the SimCLR fine-tuned model) achieved similar results, with the Baseline CNN model outperforming the SimCLR model by a small margin. The two models performed well on most galaxies (around 80%-90% accuracy) but struggled with certain classes, such as disturbed and loose galaxies that don't have a clear morphology. The baseline CNN model performed better on those classes (60% accuracy compared to 53% in the disturbed class) which gave it the better overall results. The main challange was the inconsistency of the data itself - sometimes a galaxy was clear and large in the center of the frame and sometimes the galaxy was a relatively small patch in the center surrounded by other galaxies (a deep-field image). This made it harder for the network to learn meaningful features and keep the focus on the target galaxy and not other galaxies in the frame, although this artifact was consistent in all classes. 

