import h5py
import matplotlib.pyplot as plt
import torch
import numpy as np

# Import our custom transforms
from architectures import Image_Augmentations

def main():
   
    aug = Image_Augmentations(size=144)

    with h5py.File("data/Galaxy10_DECals.h5", "r") as f:
        # Get 5 widely spaced images to ensure variety
        images = f["images"][100:105]

    fig, axes = plt.subplots(5, 3, figsize=(10, 16))
    axes[0, 0].set_title("Original (256x256)")
    axes[0, 1].set_title("Augmented View 1")
    axes[0, 2].set_title("Augmented View 2")

    for i in range(5):
        img_np = images[i]
        
        # Plot Original
        axes[i, 0].imshow(img_np)
        axes[i, 0].axis("off")
        
        # Apply Base Transform (ToTensor: converts to [3, H, W] and normalizes to [0,1])
        base_tensor = base_transform(img_np)
        
        # Simulate BYOL creating two augmented views from the base tensor
        view1 = aug(base_tensor)
        view2 = aug(base_tensor)
        
        # Helper to un-normalize from [-1, 1] back to [0, 1] for matplotlib
        def unnormalize(t):
            t = t * 0.5 + 0.5 
            t = torch.clamp(t, 0, 1)
            # Convert to [H, W, C]
            return t.permute(1, 2, 0).numpy()
            
        axes[i, 1].imshow(unnormalize(view1))
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(unnormalize(view2))
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("augmentations_preview.png", dpi=150)
    print("Saved preview to augmentations_preview.png")

if __name__ == "__main__":
    # Ensure sys path knows about src if executed from root
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    main()
