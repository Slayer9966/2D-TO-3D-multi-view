import torch
import torch.nn as nn
import torch.nn.functional as F

class Merger(nn.Module):
    def __init__(self, leaky_value=0.1):
        super(Merger, self).__init__()
        self.leaky_value = leaky_value

        self.layer1 = nn.Sequential(
            nn.Conv3d(9, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(self.leaky_value)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(self.leaky_value)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(8, 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(self.leaky_value)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(4, 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(2),
            nn.LeakyReLU(self.leaky_value)
        )
        self.final_conv = nn.Conv3d(2, 1, kernel_size=3, padding=1)


    def forward(self, raw_features, gen_volumes):
        """
        raw_features: [B, V, 9, 32, 32, 32]
        gen_volumes: [B, V, 32, 32, 32]
        """

        n_views = gen_volumes.size(1)  # number of views
        raw_features_split = torch.split(raw_features, 1, dim=1)  # split by view
        volume_weights = []

        for i in range(n_views):
            # Remove the view dimension: [B, 9, 32, 32, 32]
            raw_feat = raw_features_split[i].squeeze(1)

            weight = self.layer1(raw_feat)
            weight = self.layer2(weight)
            weight = self.layer3(weight)
            weight = self.layer4(weight)
            weight = self.final_conv(weight)

            # weight shape: [B, 1, 32, 32, 32] → squeeze channel dim
            weight = weight.squeeze(1)  # [B, 32, 32, 32]

            volume_weights.append(weight)

        # Stack and permute to [B, V, 32, 32, 32]
        volume_weights = torch.stack(volume_weights, dim=1)

        # Softmax over views dimension (dim=1)
        volume_weights = torch.softmax(volume_weights, dim=1)

        # Weighted sum of volumes by weights
        weighted_volumes = gen_volumes * volume_weights

        merged_volume = torch.sum(weighted_volumes, dim=1)

        # Clamp output between 0 and 1

        return merged_volume
