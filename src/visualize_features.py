"""
Feature Visualization for SimCLR Backbone
==========================================
Three complementary techniques to understand what the network learned:
  1. Grad-CAM: heatmaps showing which spatial regions the backbone attends to
  2. Feature Maps: raw activations from each residual layer
  3. Nearest Neighbors: images that are closest in the learned feature space

Usage:
  python src/visualize_features.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import os

from dataset import GalaxyDataset, class_map
import architectures as arch


# ──────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────

def load_config(type, parameter):
    with open('configs/config.yaml', 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data[type][parameter]


def load_backbone(device):
    """Load CNNclassifier, strip the head, load SimCLR weights."""
    model = arch.CNNclassifier()

    # Replace head with identity so output = 512-dim features
    model.dropout = torch.nn.Identity()
    model.fc = torch.nn.Identity()

    ssl_weights = load_config('Training Parameters', 'loaded_model_name')
    print(f"Loading SimCLR weights from models/{ssl_weights}")
    state_dict = torch.load(f"models/{ssl_weights}", map_location=device)
    new_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def unnormalize(img_tensor):
    """Reverse the Normalize(0.5, 0.5, 0.5) to get pixel values back in [0,1]."""
    return (img_tensor * 0.5 + 0.5).clamp(0, 1)


def get_test_loader(batch_size=64):
    file_path = load_config('Data Parameters', 'file')
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((144, 144)),
        transforms.CenterCrop(144),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_dataset = GalaxyDataset(file_path=file_path, transform=val_transforms, split='test')
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ──────────────────────────────────────────────────────────
#  1. Grad-CAM
# ──────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for the SimCLR backbone.

    Since there is no classification head, we compute gradients w.r.t. a chosen
    feature dimension (the one with the highest activation for each image).
    This highlights which spatial regions contribute most to the dominant feature.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        # Register hooks on the target layer
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """Returns a batch of Grad-CAM heatmaps, shape (B, H, W)."""
        self.model.zero_grad()
        output = self.model(input_tensor)  # (B, 512)

        # Pick the strongest feature dimension for each image as the "target"
        target_indices = output.argmax(dim=1)  # (B,)

        # Backprop w.r.t. that feature
        one_hot = torch.zeros_like(output)
        for i in range(output.size(0)):
            one_hot[i, target_indices[i]] = 1.0
        output.backward(gradient=one_hot)

        # Grad-CAM computation
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP of gradients
        cam = (weights * self.activations).sum(dim=1)             # weighted sum
        cam = F.relu(cam)                                          # keep only positive

        # Normalize per image to [0, 1]
        B = cam.size(0)
        cam = cam.view(B, -1)
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam = cam.view(B, self.activations.size(2), self.activations.size(3))
        return cam


def visualize_gradcam(model, dataloader, device, num_images=8,
                      save_path='models/feature visualizations/gradcam_visualization_baseline_CNN.png'):
    """Generate Grad-CAM heatmaps for the first `num_images` test images."""
    print("\n=== Generating Grad-CAM Heatmaps ===")

    # Need gradients for Grad-CAM, so temporarily enable them
    model.eval()
    cam_extractor = GradCAM(model, target_layer=model.layer3)

    # Grab one batch
    images, labels = next(iter(dataloader))
    images = images[:num_images].to(device).requires_grad_(True)
    labels = labels[:num_images]

    # Generate heatmaps
    heatmaps = cam_extractor.generate(images)

    # Upsample heatmaps to image size
    heatmaps = F.interpolate(
        heatmaps.unsqueeze(1), size=(144, 144), mode='bilinear', align_corners=False
    ).squeeze(1).cpu().numpy()

    fig, axes = plt.subplots(2, num_images, figsize=(3.2 * num_images, 7))
    fig.suptitle("Grad-CAM: Where does the baseline CNN focus?",
                 fontsize=16, fontweight='bold', y=1.02)

    for i in range(num_images):
        orig = unnormalize(images[i].detach().cpu()).permute(1, 2, 0).numpy()
        heatmap = heatmaps[i]

        # Row 1: Original image
        axes[0, i].imshow(orig)
        axes[0, i].set_title(class_map[labels[i].item()], fontsize=8)
        axes[0, i].axis('off')

        # Row 2: Original + heatmap overlay
        axes[1, i].imshow(orig)
        axes[1, i].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved Grad-CAM visualization vanila CNN → {save_path}")


# ──────────────────────────────────────────────────────────
#  2. Intermediate Feature Maps
# ──────────────────────────────────────────────────────────

def visualize_feature_maps(model, dataloader, device, num_channels=8,
                           save_path='models/feature visualizations/feature_maps_visualization_baseline_CNN.png'):
    """
    For a single galaxy image, show the first `num_channels` activation maps
    from each of the 4 residual layers. This reveals how features evolve from
    low-level edges → high-level galactic structure.
    """
    print("\n=== Visualizing Intermediate Feature Maps ===")

    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Attach hooks to each residual layer
    layers = {
        'Layer 1 (64ch)': model.layer1,
        'Layer 2 (128ch)': model.layer2,
        'Layer 3 (256ch)': model.layer3,
        'Layer 4 (512ch)': model.layer4,
    }
    hooks = []
    for name, layer in layers.items():
        hooks.append(layer.register_forward_hook(hook_fn(name)))

    # Forward a single image
    images, labels = next(iter(dataloader))
    single_img = images[0:1].to(device)
    label = labels[0].item()

    with torch.no_grad():
        model(single_img)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Plot
    layer_names = list(layers.keys())
    fig, axes = plt.subplots(len(layer_names) + 1, num_channels,
                             figsize=(2.5 * num_channels, 2.5 * (len(layer_names) + 1)))
    fig.suptitle(f"Feature Map Evolution — {class_map[label]}",
                 fontsize=16, fontweight='bold', y=1.01)

    # Row 0: Show the original image in the first cell, rest empty
    orig = unnormalize(single_img[0].cpu()).permute(1, 2, 0).numpy()
    axes[0, 0].imshow(orig)
    axes[0, 0].set_title("Input Image", fontsize=9)
    for j in range(num_channels):
        axes[0, j].axis('off')

    # Rows 1-4: Activation maps
    for row, name in enumerate(layer_names, start=1):
        act = activations[name][0].cpu().numpy()  # (C, H, W)

        # Pick the channels with the highest mean activation (most "active")
        channel_means = act.mean(axis=(1, 2))
        top_channels = np.argsort(channel_means)[-num_channels:][::-1]

        for j, ch_idx in enumerate(top_channels):
            axes[row, j].imshow(act[ch_idx], cmap='inferno')
            axes[row, j].axis('off')
            if j == 0:
                axes[row, j].set_ylabel(name, fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved feature maps visualization → {save_path}")


# ──────────────────────────────────────────────────────────
#  3. Nearest Neighbors in Feature Space
# ──────────────────────────────────────────────────────────

def visualize_nearest_neighbors(model, dataloader, device, num_queries=4, k=6,
                                save_path='models/feature visualizations/nearest_neighbors_visualization_baseline_CNN.png'):
    """
    For `num_queries` random test images, find the `k` nearest neighbors in the
    512-dim SimCLR feature space using cosine similarity. If the network learned
    good representations, neighbors should share morphological traits.
    """
    print("\n=== Finding Nearest Neighbors in Feature Space ===")

    all_features = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            feats = model(images)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
            all_images.append(images.cpu())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_images = torch.cat(all_images, dim=0)

    print(f"Extracted features for {len(all_features)} images.")

    # Pick diverse query indices (one from each morphological group if possible)
    np.random.seed(42)
    # Try to sample one image per class for diversity
    unique_labels = np.unique(all_labels)
    query_indices = []
    for lbl in unique_labels[:num_queries]:
        candidates = np.where(all_labels == lbl)[0]
        query_indices.append(np.random.choice(candidates))

    # Compute cosine similarity for each query
    fig, axes = plt.subplots(num_queries, k + 1,
                             figsize=(2.8 * (k + 1), 3 * num_queries))
    fig.suptitle("Nearest Neighbors in baseline CNN Feature Space (Cosine Similarity)",
                 fontsize=16, fontweight='bold', y=1.02)

    for row, q_idx in enumerate(query_indices):
        query_feat = all_features[q_idx:q_idx+1]  # (1, 512)
        sims = cosine_similarity(query_feat, all_features)[0]  # (N,)
        sims[q_idx] = -1  # exclude self

        top_k = np.argsort(sims)[-k:][::-1]

        # Column 0: Query image
        query_img = unnormalize(all_images[q_idx]).permute(1, 2, 0).numpy()
        axes[row, 0].imshow(query_img)
        axes[row, 0].set_title(f"QUERY\n{class_map[all_labels[q_idx]]}", fontsize=8,
                               fontweight='bold', color='blue')
        axes[row, 0].axis('off')

        # Add blue border to query
        for spine in axes[row, 0].spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(3)
            spine.set_visible(True)

        # Columns 1..k: Neighbors
        for col, n_idx in enumerate(top_k, start=1):
            neighbor_img = unnormalize(all_images[n_idx]).permute(1, 2, 0).numpy()
            axes[row, col].imshow(neighbor_img)

            n_label = class_map[all_labels[n_idx]]
            sim_score = sims[n_idx]
            match = all_labels[n_idx] == all_labels[q_idx]

            # Green border = same class, red = different class
            border_color = 'green' if match else 'red'
            axes[row, col].set_title(f"{n_label}\nsim={sim_score:.3f}", fontsize=7)
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)
                spine.set_visible(True)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved nearest neighbors visualization → {save_path}")


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")


    model =  arch.CNNclassifier()
    model.load_state_dict(torch.load('models/CNN model/best_model_supervised_CNN.pth', map_location=device))
    model.to(device)
    model.eval()
    #model = load_backbone(device)
    test_loader = get_test_loader(batch_size=64)

    # 1. Grad-CAM
    visualize_gradcam(model, test_loader, device)

    # 2. Feature Maps
    visualize_feature_maps(model, test_loader, device)

    # 3. Nearest Neighbors
    visualize_nearest_neighbors(model, test_loader, device)



if __name__ == "__main__":
    main()
