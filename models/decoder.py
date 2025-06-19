import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # More conservative expansion from pseudo-3D to true 3D
        self.expand_to_3d = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3), stride=1, padding=1),  # Reduce channels early
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 128, kernel_size=(2, 1, 1), stride=(2, 1, 1)),  # [B, 128, 2, 20, 20]
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=(2, 1, 1), stride=(2, 1, 1)),   # [B, 64, 4, 20, 20]
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 64, kernel_size=(2, 1, 1), stride=(2, 1, 1)),    # [B, 64, 8, 20, 20]
            nn.ReLU(inplace=True)
        )
        
        # Progressive upsampling with controlled channel reduction
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 16, 40, 40]
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),  # [B, 16, 32, 80, 80]
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        
        # Instead of continuing to upsample spatially, resize to target
        self.final_resize = nn.Sequential(
            nn.Upsample(size=(32, 32, 32), mode='trilinear', align_corners=False),
            nn.Conv3d(16, 8, kernel_size=3, padding=1),  # [B, 8, 32, 32, 32]
            nn.ReLU(inplace=True)
        )
        
        # Final layer to generate volume
        self.layer5 = nn.Sequential(
            nn.Conv3d(8, 1, kernel_size=1),
            # nn.Sigmoid()
        )

    def forward(self, image_features):
        # image_features: [B, V, 256, 20, 20]
        B, V = image_features.shape[:2]
        
        # Process each view separately to manage memory
        gen_volumes = []
        raw_features = []
        
        for v in range(V):
            features = image_features[:, v]  # [B, 256, 20, 20]
            features = features.unsqueeze(2)  # [B, 256, 1, 20, 20]
            
            # Expand to 3D with reduced channels
            x = self.expand_to_3d(features)  # [B, 64, 8, 20, 20]
            
            # Progressive upsampling
            x = self.layer1(x)  # [B, 32, 16, 40, 40]
            x = self.layer2(x)  # [B, 16, 32, 80, 80]
            
            # Resize to final dimensions
            x = self.final_resize(x)  # [B, 8, 32, 32, 32]
            
            raw_feat = x  # Store before final layer
            
            # Generate volume
            volume = self.layer5(x)  # [B, 1, 32, 32, 32]
            
            # Combine raw features with volume
            raw_feat = torch.cat([raw_feat, volume], dim=1)  # [B, 9, 32, 32, 32]
            
            gen_volumes.append(volume.squeeze(1))  # [B, 32, 32, 32]
            raw_features.append(raw_feat)
            
            # Clear cache to free memory
            del features, x, volume
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Stack results
        gen_volumes = torch.stack(gen_volumes, dim=1)  # [B, V, 32, 32, 32]
        raw_features = torch.stack(raw_features, dim=1)  # [B, V, 9, 32, 32, 32]
        
        return raw_features, gen_volumes