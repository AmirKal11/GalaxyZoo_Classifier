import torch
from torch import nn 
import numpy as np
import yaml 
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt

### Local imports
import src.architectures as arch
from src.architectures import CircularMask, Image_Augmentations
from src.dataset import GalaxyDataset
from src.utils import plot_training_curves, get_all_predictions, plot_confusion_matrix, print_performance_report

def load_config(type,parameter):
    with open('configs/config.yaml', 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data[type][parameter]


class_names = [
    "Disturbed Galaxy",
    "Merging Galaxy",
    "Round Smooth Galaxy",
    "In-between Round Smooth Galaxy",
    "Cigar Shaped Smooth Galaxy",
    "Barred Spiral Galaxy",
    "Unbarred Tight Spiral Galaxy",
    "Unbarred Loose Spiral Galaxy",
    "Edge-on Galaxy without Bulge",
    "Edge-on Galaxy with Bulge"
]


def create_dataloader(file,batch_size):

    train_transforms = Image_Augmentations(size=144).transformations

    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.CenterCrop(144),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        #CircularMask(radius_fraction=0.4)
    ])
    
    

    train_dataset = GalaxyDataset(file_path = file,transform=train_transforms,split='train')
    val_dataset = GalaxyDataset(file_path = file,transform=val_transforms,split='val')
    test_dataset = GalaxyDataset(file_path = file,transform=val_transforms,split='test')
    weights = train_dataset.get_weights()
    train_dataloader = DataLoader(train_dataset, batch_size=load_config('Training Parameters','batch_size'), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=load_config('Training Parameters','batch_size'), shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=load_config('Training Parameters','batch_size'), shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader, weights




def train_one_epoch(model, dataloader, optimizer, criterion,device = torch.device('mps'),**kwargs):
    model.train()
    train_loss,train_acc =  0 , 0
    loading_weights = kwargs.get('loading_weights', False)
    freeze_backbone = True
    try:
        freeze_backbone = load_config('Training Parameters', 'freeze_backbone')
    except:
        pass
        
    if loading_weights and freeze_backbone:
        for name, module in model.named_modules():
            if "fc" not in name:  # Keep only your classifier head in train mode
                module.eval()

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output,target)
        _,pred = torch.max(output,1)
        train_acc += (pred == target).sum().item()
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        #if batch_idx % 200 == 0: print(f"Batch {batch_idx} Loss: {loss.item():.4f}")

    return train_loss/len(dataloader), train_acc/len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss,val_acc = 0 , 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output,target)
            _,pred = torch.max(output,1)
            val_acc += (pred == target).sum().item()
            val_loss += loss.item()
            #if batch_idx % 200 == 0: print(f"Batch {batch_idx} Loss: {loss.item():.4f}")
    return val_loss/len(dataloader), val_acc/len(dataloader.dataset)

        


def main():
    batch_size = load_config('Training Parameters', 'batch_size')
    file = load_config('Data Parameters', 'file')
    loading_weights = load_config('Training Parameters', 'loading_weights')
    
    train_dataloader, val_dataloader, test_dataloader, weights = create_dataloader(file,batch_size)
    freeze_backbone = load_config('Training Parameters', 'freeze_backbone')    
    
    device = torch.device('mps')
    class_weights = weights.to(device)

    model = arch.CNNclassifier()
    if loading_weights:
        print("--- Loading SSL Backbone Weights ---")
        state_dict = torch.load(f"models/{load_config('Training Parameters', 'loaded_model_name')}", map_location=device)
        
       

        # Strip prefixes correctly BEFORE loading
        new_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
        # Load weights once
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Load Status: {msg}")

        # Freeze Backbone if configured
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "fc" not in name: # Adjust 'fc' to whatever your head is named
                    param.requires_grad = False
            print("Backbone frozen. Training classification head only.")
        else:
            print("Backbone UNFROZEN. Fully fine-tuning the entire network.")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    base_lr = float(load_config('Training Parameters', 'learning_rate'))
    weight_decay = float(load_config('Training Parameters', 'weight_decay'))
    num_epochs = int(load_config('Training Parameters', 'num_epochs'))
    
    lp_ft_mode = loading_weights and not freeze_backbone
    if lp_ft_mode:
        print("Engaging LP-FT Mode: Linear Probing for 20 epochs, then Full Fine-Tuning.")
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
                
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-2, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-4)
    else:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=base_lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    model.to(device)
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):

        if lp_ft_mode and epoch == 20:
            print(f"\n--- Phase 2: Transitioning to Full Fine-Tuning at base LR={base_lr}! ---")
            for param in model.parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_epochs - 20), eta_min=1e-6)

        train_loss, train_acc = train_one_epoch(model, train_dataloader, optimizer, criterion, device, loading_weights=loading_weights)
        val_loss, val_acc = validate(model, val_dataloader, criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print("--- New Best Val Acc! Model Saved ---")
            
            save_name = 'best_model_linear_probe_masked.pth' if loading_weights else 'best_model_supervised_CNN_masked.pth'
            torch.save(model.state_dict(), f'models/{save_name}')
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
       
        print(f"In epoch number {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


    plot_training_curves(
        history['train_loss'], 
        history['val_loss'], 
        history['train_acc'], 
        history['val_acc'],
        save_path='models/training_metrics.png'
    )

    save_name = 'best_model_linear_probe.pth' if loading_weights else 'best_model_supervised_CNN.pth'
    model.load_state_dict(torch.load(f'models/{save_name}'))

    # Generate final reports
    y_true, y_pred = get_all_predictions(model, test_dataloader, device)
    print_performance_report(y_true, y_pred, class_names)
    cm_save_name = 'final_cm_linear_probe_masked.png' if loading_weights else 'final_cm_supervised_CNN_masked.png'
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=f'models/{cm_save_name}')
   
   
   
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()