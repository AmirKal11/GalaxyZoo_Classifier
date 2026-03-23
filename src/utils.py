import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, f1_score

GALAXY10_CLASSES = [
    "Disturbed", "Merging", "Round Smooth", 
    "In-between Smooth", "Cigar Smooth", 
    "Barred Spiral", "Tight Spiral", 
    "Loose Spiral", "Edge-on No Bulge", "Edge-on Bulge"
]

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path='metrics_plot.png'):
    """Plots loss and accuracy side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss Plot
    ax1.plot(train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', color='#ff7f0e', linestyle='--')
    ax1.set_title('Cross-Entropy Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Accuracy Plot
    ax2.plot(train_accs, label='Train Acc', color='#2ca02c', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', color='#d62728', linestyle='--')
    ax2.set_title('Classification Accuracy (%)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path='confusion_matrix.png'):
    """Generates a heatmap of predictions vs. ground truth."""
    if class_names is None or (class_names and "Class 0" in class_names):
        class_names = GALAXY10_CLASSES
        
    # Explicitly calculate CM for ALL classes to ensure labels match class_names
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalize by row (True labels) to see recall per class
    # Add small epsilon to avoid division by zero
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix (Recall)')
    plt.xlabel('Predicted Morphology')
    plt.ylabel('True Morphology')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

@torch.no_grad()
def get_all_predictions(model, dataloader, device):
    """Utility to gather all predictions for metrics calculation."""
    model.eval()
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    return np.array(all_labels), np.array(all_preds)


def print_performance_report(y_true, y_pred, class_names=None):
    """Prints F1-score and precision/recall summary."""
    if class_names is None or (class_names and "Class 0" in class_names):
        class_names = GALAXY10_CLASSES
        
    print("\n--- Galactic Morphology Performance Report ---")
    
    # Explicitly pass labels to ensure correct mapping to target_names
    labels = np.arange(len(class_names))
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names))
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Overall Macro F1-Score: {macro_f1:.4f}")