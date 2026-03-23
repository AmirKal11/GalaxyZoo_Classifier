import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from matplotlib import pyplot as plt

from src.architectures import Image_Augmentations, SimCLRModel, CNNclassifier 
from src.dataset import GalaxyDataset
import numpy as np
import yaml

def load_config(type,parameter):
    with open('configs/config.yaml', 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data[type][parameter]



def create_ssl_dataloader(file_path, batch_size):    
    ssl_transform = Image_Augmentations(size=144)
    train_dataset = GalaxyDataset(file_path=file_path, transform=ssl_transform, split='train', ssl_mode=True)
    val_dataset = GalaxyDataset(file_path=file_path, transform=ssl_transform, split='val', ssl_mode=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_dataloader, val_dataloader


# ─────────────────────────────────────
#  SimCLR functions
# ─────────────────────────────────────

def calc_ntxent_loss(z_i, z_j, temperature):
    batch_size = z_i.shape[0]
    
    # Normalize features
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # [2N, feature_dim]
    z = torch.cat([z_i, z_j], dim=0)
    
    # Native Matmul for [2N, 2N] space
    sim_matrix = torch.matmul(z, z.T) / temperature
    
    # To drop the main diagonal, set it to -infinity so softmax ignores it 
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim_matrix.masked_fill_(mask, -1e9)
    
    # Define labels (The true matching pairs naturally sit at offset 'batch_size' in the 2N matrix)
    labels = torch.arange(2 * batch_size, device=z.device)
    labels = (labels + batch_size) % (2 * batch_size)
    
    # Natively supported on MPS without CPU shape syncs:
    return F.cross_entropy(sim_matrix, labels)

 

def train_one_epoch_simclr(model, dataloader, optimizer, device, temperature):
    model.train()
    train_loss = 0

    for batch_idx, views in enumerate(dataloader):
        view1, view2 = views[0].to(device), views[1].to(device)
        optimizer.zero_grad()
        
        z1 = model(view1)
        z2 = model(view2)

        loss = calc_ntxent_loss(z1, z2, temperature)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0: print(f"Batch {batch_idx} Loss: {loss.item():.4f}")

    return train_loss / len(dataloader)

def validate_simclr(model, loader, device, temperature):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for views in loader:
            v1, v2 = views[0].to(device), views[1].to(device)
            
            z1, z2 = model(v1), model(v2)
            current_loss = calc_ntxent_loss(z1, z2, temperature)
            
            total_loss += current_loss.item()
            
    return total_loss / len(loader)




# ─────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────

def main_ssl():
    method = load_config('SSL Parameters', 'ssl_method')
    base_lr = float(load_config('SSL Parameters', 'learning_rate_ssl'))
    warmup_epochs = 10
    total_epochs = load_config('SSL Parameters', 'num_epochs_ssl')
    file_path = load_config('Data Parameters', 'file')
    batch_size = load_config('SSL Parameters', 'ssl_batch_size')
    
    train_dataloader, val_dataloader = create_ssl_dataloader(file_path, batch_size)
    print(f'Created dataloaders for {method.upper()}')

    device = torch.device('mps')
    backbone = CNNclassifier()

    temperature = load_config('SSL Parameters', 'temperature')
    model = SimCLRModel(backbone=backbone)
    model.to(device)
    print('Initialized SimCLR model')

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=float(load_config('SSL Parameters', 'weight_decay_ssl')))

    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_cosine], 
        milestones=[warmup_epochs]
    )
    print('Initialized optimizers')

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(total_epochs):
        train_loss = train_one_epoch_simclr(model, train_dataloader, optimizer, device, temperature)
        val_loss = validate_simclr(model, val_dataloader, device, temperature)

        print('Computed loss')
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_name = f'models/best_{method}_backbone_{batch_size}_batch_{total_epochs}_epochs.pth'
            torch.save(backbone.state_dict(), save_name)
            print("--- New Best Val Loss! Backbone Saved ---")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
       
        print(f"In epoch number {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title(f'{method.upper()} Loss')
    plt.legend()
    plt.savefig(f'models/{method}_loss_{batch_size}_batch_{total_epochs}_epochs.png')
    
    plt.show()
    
if __name__ == "__main__":
    main_ssl()