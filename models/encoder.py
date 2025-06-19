import torch
import torchvision.models
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, cfg=None):
        super(Encoder, self).__init__()
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        # Use VGG16_bn features up to layer 27 (conv5_3)
        self.vgg = nn.Sequential(*list(vgg16_bn.features.children())[:27])
        
        # Extra conv layers as in Pix2Vox - adjusted padding to get 20x20 output from 256x256 input
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=7),  # Padding adjusted for desired output size
            nn.BatchNorm2d(512),
            nn.ELU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=7),  # Padding adjusted for desired output size
            nn.BatchNorm2d(512),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True)
        )
        
        # Freeze VGG16 parameters (do not train)
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, rendering_images):
        """
        rendering_images: (B, V, 3, 256, 256)  # Changed from 512x512
        returns: (B, V, 256, 20, 20)  # Same output as before
        """
        B, V, C, H, W = rendering_images.size()
        # Permute to (V, B, C, H, W) for easier splitting on views
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        # Split into list of tensors, each (1, B, C, H, W)
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []
        for img in rendering_images:
            # Remove extra dimension: (B, C, H, W)
            img = img.squeeze(dim=0)
            feat = self.vgg(img)        # (B, 512, ~8, ~8) for 256x256 input
            feat = self.layer1(feat)    # (B, 512, with padding)
            feat = self.layer2(feat)    # (B, 512, with padding and maxpool)
            feat = self.layer3(feat)
            feat = F.interpolate(feat, size=(20, 20), mode='bilinear', align_corners=False)
# (B, 256, 20, 20)
            image_features.append(feat)
        # Stack views back: (V, B, 256, 20, 20) → permute → (B, V, 256, 20, 20)
        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        return image_features