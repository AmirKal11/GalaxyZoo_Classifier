import torch
import yaml
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Local imports
import src.architectures as arch
from torchvision.transforms import v2
from src.dataset import GalaxyDataset
from src.utils import get_all_predictions, plot_confusion_matrix, print_performance_report

def load_config(type,parameter):
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config[type][parameter]

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_path = 'data/Galaxy10_DECals.h5'
    batch_size = 256
    
    # Matching deterministic val/test transformations
    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.CenterCrop(144),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print("Loading test dataset...")
    test_dataset = GalaxyDataset(file_path=file_path, transform=val_transforms, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Loading model architecture...")
    model = arch.CNNclassifier().to(device)

    # Determine which weights to load based on current config state
    model_path = '/Users/amir/Documents/Deep learning/GalaxyZoo_Classifier/models/best_model_linear_probe_masked.pth'
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded backbone weights from {model_path}!")

    model.eval()

    class_names = [
        "Disturbed", "Merging", "Round Smooth",
        "In-between Smooth", "Cigar Smooth",
        "Barred Spiral", "Tight Spiral",
        "Loose Spiral", "Edge-on No Bulge",
        "Edge-on Bulge"
    ]

    y_true, y_pred = get_all_predictions(model, test_dataloader, device)

    print("\n--- Classification Report ---")
    print_performance_report(y_true, y_pred, class_names)

    out_png = model_path.replace('.pth', '_cm.png')
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=out_png)

if __name__ == "__main__":
    main()
