import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class LightweightFeatureFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(LightweightFeatureFusion, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1, groups=in_channels, bias=False),
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels, bias=False),
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        return torch.cat([out1, out2], dim=1)

class UNetX(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: List[int] = [16, 32, 64, 128]):
        super(UNetX, self).__init__()

        self.encoder_blocks = nn.ModuleList()
        self.lff_modules = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        for i, feature in enumerate(features):
            in_feats = in_channels if i == 0 else features[i-1]
            self.encoder_blocks.append(ConvBlock(in_feats, feature))
            self.lff_modules.append(LightweightFeatureFusion(feature, feature))

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.upsamples = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.deep_supervision_outputs = nn.ModuleList()

        reversed_features = features[::-1]
        sum_of_lff_features = sum(features)

        for i in range(len(reversed_features) - 1):  # only 3 decoder stages
            up_in_channels = features[-1] * 2 if i == 0 else reversed_features[i - 1]
            up_out_channels = reversed_features[i]
            self.upsamples.append(
                nn.ConvTranspose3d(up_in_channels, up_out_channels, kernel_size=2, stride=2)
            )
            decoder_in_channels = up_out_channels + sum_of_lff_features
            self.decoder_convs.append(ConvBlock(decoder_in_channels, reversed_features[i]))
            self.deep_supervision_outputs.append(nn.Conv3d(reversed_features[i], out_channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        encoder_outputs = []
        lff_outputs = []

        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            encoder_outputs.append(x)
            lff_outputs.append(self.lff_modules[i](x))
            if i < len(self.encoder_blocks) - 1:
                x = self.pool(x)

        x = self.bottleneck(x)

        outputs_for_supervision = []

        for i in range(len(self.decoder_convs)):
            x = self.upsamples[i](x)
            feature_maps_to_fuse = [x]

            for lff_out in lff_outputs:
                target_size = x.shape[2:]
                if lff_out.shape[2:] != target_size:
                    lff_out_resized = F.interpolate(lff_out, size=target_size, mode='trilinear', align_corners=False)
                    feature_maps_to_fuse.append(lff_out_resized)
                else:
                    feature_maps_to_fuse.append(lff_out)

            fused = torch.cat(feature_maps_to_fuse, dim=1)
            x = self.decoder_convs[i](fused)
            outputs_for_supervision.append(self.deep_supervision_outputs[i](x))

        return [torch.sigmoid(out) for out in outputs_for_supervision]

# --- Test the Model ---
if __name__ == '__main__':
    input_channels = 1
    output_channels = 1
    feature_levels = [16, 32, 64, 128]

    model = UNetX(in_channels=input_channels, out_channels=output_channels, features=feature_levels)
    dummy_input = torch.randn(1, input_channels, 128, 128, 128)
    predictions = model(dummy_input)

    print("U-NetX Model Instantiated Successfully.")
    print(f"Number of deep supervision outputs: {len(predictions)}")

    for i, pred in enumerate(predictions):
        print(f"  - Supervision Output {i+1} shape: {pred.shape}")

    final_prediction = predictions[-1]
    print(f"\nShape of the final prediction mask: {final_prediction.shape}")
    print(f"Shape of the input tensor:        {dummy_input.shape}")

    assert final_prediction.shape[2:] == dummy_input.shape[2:], "Output spatial dimensions do not match input!"
    print("\nModel forward pass test successful.")
