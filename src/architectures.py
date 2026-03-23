import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import v2

class CircularMask:
    """Applies a smooth circular mask that fades edges to zero.
    This forces the network to focus on the centered galaxy and ignore
    background stars and noise at the image periphery."""
    def __init__(self, radius_fraction=0.45):
        self.radius_fraction = radius_fraction

    def __call__(self, img_tensor):
        _, H, W = img_tensor.shape
        cy, cx = H / 2, W / 2
        radius = self.radius_fraction * min(H, W)

        y = torch.arange(H).float().unsqueeze(1)
        x = torch.arange(W).float().unsqueeze(0)
        dist = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)

        # Smooth falloff: 1 inside radius, fades to 0 outside
        mask = torch.clamp(1.0 - (dist - radius) / (radius * 0.15), 0, 1)
        return img_tensor * mask.unsqueeze(0)


class Image_Augmentations():
    def __init__(self,size = 144):
      self.transformations = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomAffine(degrees=180, scale=(1.0, 1.5)), # Zooms IN only, preventing black padding shortcuts
            v2.CenterCrop(size),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0),
            v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)), # Smoothes out sensor noise shortcuts
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self, x):
        return [self.transformations(x),self.transformations(x)]



class SimCLRModel(nn.Module):
    def __init__(self, backbone, projection_dim=128):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = 512 


        ### Remove the final layer of the backbone and add a projector
        self.backbone.fc = nn.Identity() 
        self.backbone.dropout = nn.Identity() 
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, projection_dim))

    def forward(self, x):
        h = self.backbone(x) 
        z = self.projector(h)
        return torch.nn.functional.normalize(z, dim=1)




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # This is the 'Shortcut' logic
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # The key ResNet step: Add the input back to the output!
        out += self.shortcut(identity) 
        return self.gelu(out)



class CNNclassifier(nn.Module):
    def __init__(self,num_classes = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 1: Keeps channels at 64
        self.layer1 = ResidualBlock(64, 64, stride=1)
        
        # Layer 2: Doubles channels to 128, halves image size (stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        
        # Layer 3: Doubles channels to 256, halves image size (stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        
        # Layer 4: Doubles channels to 512 
        self.layer4 = ResidualBlock(256, 512, stride=2)

        # 3. The Head: Final classification as described in the paper [cite: 216, 525, 853]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)
 
      
    def forward(self, x):
        # First Block - The 'stem'
        x = self.maxpool(self.gelu(self.bn1(self.conv1(x))))
        
        # ResNet Blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        

        # Convert to vector and classify
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
           
       
        


