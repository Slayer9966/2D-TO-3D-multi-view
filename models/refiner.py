import torch
import torch.nn as nn
import torch.nn.functional as F

class Refiner(nn.Module):
    def __init__(self, leaky_value=0.1, use_bias=True, n_vox=32, debug=False):
        super(Refiner, self).__init__()
        self.leaky_value = leaky_value
        self.n_vox = n_vox
        self.debug = debug
        
        # Encoder part - downsample with skip connections
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, padding=2),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(leaky_value),
            nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=4, padding=2),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(leaky_value),
            nn.MaxPool3d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=4, padding=2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(leaky_value),
            nn.MaxPool3d(kernel_size=2)
        )
        
        # Less aggressive bottleneck
        self.layer4 = nn.Sequential(
            nn.Linear(8192, 4096),  # Less compression
            nn.ReLU(),
            nn.Dropout(0.2)  # Add some regularization
        )
        self.layer5 = nn.Sequential(
            nn.Linear(4096, 8192),
            nn.ReLU()
        )
        
        # Decoder part - upsample with skip connections
        self.layer6 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=use_bias, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=use_bias, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=use_bias, padding=1),
            # No activation - BCEWithLogitsLoss expects raw logits
        )
        
        # Learnable residual weight instead of fixed 0.5
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, coarse_volumes):
        """
        Input: coarse_volumes [B, 32, 32, 32] - output from your Merger  
        Output: refined_volumes [B, 32, 32, 32] - refined version
        """
        # Reshape input to have channel dimension
        volumes_32_l = coarse_volumes.view(-1, 1, self.n_vox, self.n_vox, self.n_vox)
        
        if self.debug:
            print(f"Input stats: min={volumes_32_l.min():.4f}, max={volumes_32_l.max():.4f}, mean={volumes_32_l.mean():.4f}")
        
        # Encoder with skip connections
        volumes_16_l = self.layer1(volumes_32_l)
        volumes_8_l = self.layer2(volumes_16_l)
        volumes_4_l = self.layer3(volumes_8_l)
        
        if self.debug:
            print(f"Encoder output stats: min={volumes_4_l.min():.4f}, max={volumes_4_l.max():.4f}")
        
        # Bottleneck
        flatten_features = self.layer4(volumes_4_l.view(-1, 8192))
        flatten_features = self.layer5(flatten_features)
        
        if self.debug:
            print(f"Bottleneck stats: min={flatten_features.min():.4f}, max={flatten_features.max():.4f}")
        
        # Decoder with residual connections (U-Net style)
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 128, 4, 4, 4)
        volumes_8_r = volumes_8_l + self.layer6(volumes_4_r)
        volumes_16_r = volumes_16_l + self.layer7(volumes_8_r)
        
        # Final refined output (raw logits for BCEWithLogitsLoss)
        refined_output = self.layer8(volumes_16_r)
        
        if self.debug:
            print(f"Refined logits stats: min={refined_output.min():.4f}, max={refined_output.max():.4f}")
        
        # For the residual connection, we need to convert input to logits too
        # Since input is probabilities [0,1], convert to logits
        input_logits = torch.logit(torch.clamp(volumes_32_l, 1e-7, 1-1e-7))
        
        # Learnable residual connection in logit space
        volumes_32_r = input_logits + self.residual_weight * refined_output
        
        if self.debug:
            print(f"Final output stats: min={volumes_32_r.min():.4f}, max={volumes_32_r.max():.4f}")
            print(f"Residual weight: {self.residual_weight.item():.4f}")
        
        # Ensure output is in valid range [0, 1]
        volumes_32_r = torch.clamp(volumes_32_r, 0, 1)
        
        # Return in the same format as input
        return volumes_32_r.view(-1, self.n_vox, self.n_vox, self.n_vox)