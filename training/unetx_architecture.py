import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

# Define UNetX architecture for diagram
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class LightweightFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv3d(in_channels, out_channels//2, 1),
            nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=2, dilation=2, groups=in_channels),
            nn.Conv3d(in_channels, out_channels//2, 1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x)], dim=1)

class UNetX(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.lff_modules = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)

        for i, f in enumerate(features):
            self.encoder_blocks.append(ConvBlock(in_channels if i==0 else features[i-1], f))
            self.lff_modules.append(LightweightFeatureFusion(f, f))

        self.bottleneck = ConvBlock(features[-1], features[-1]*2)

        self.upsamples = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.deep_supervision_outputs = nn.ModuleList()

        reversed_features = features[::-1]
        sum_lff = sum(features)

        for i in range(len(reversed_features)-1):
            up_in = features[-1]*2 if i==0 else reversed_features[i-1]
            up_out = reversed_features[i]
            self.upsamples.append(nn.ConvTranspose3d(up_in, up_out, 2,2))
            self.decoder_convs.append(ConvBlock(up_out+sum_lff, reversed_features[i]))
            self.deep_supervision_outputs.append(nn.Conv3d(reversed_features[i], out_channels, 1))

    def forward(self, x):
        enc_outs, lff_outs = [], []

        for i, blk in enumerate(self.encoder_blocks):
            x = blk(x)
            enc_outs.append(x)
            lff_outs.append(self.lff_modules[i](x))
            if i < len(self.encoder_blocks)-1:
                x = self.pool(x)

        x = self.bottleneck(x)
        outputs = []

        for i in range(len(self.decoder_convs)):
            x = self.upsamples[i](x)
            to_fuse = [x] + [F.interpolate(lff, size=x.shape[2:], mode='trilinear', align_corners=False) if lff.shape[2:]!=x.shape[2:] else lff for lff in lff_outs]
            x = self.decoder_convs[i](torch.cat(to_fuse, dim=1))
            outputs.append(torch.sigmoid(self.deep_supervision_outputs[i](x)))

        return outputs

# Instantiate model and dummy input
model = UNetX()
dummy_input = torch.randn(1,1,64,64,64)

# Pass input through the model to get graph
outputs = model(dummy_input)
final_output = outputs[-1]

# Plot the architecture diagrammatically without torchviz/Graphviz
fig, ax = plt.subplots(figsize=(10,6))

layers = [
    "Input", "Encoder1", "LFF1", "Pool1",
    "Encoder2", "LFF2", "Pool2",
    "Encoder3", "LFF3", "Pool3",
    "Encoder4", "LFF4", "Bottleneck",
    "Upsample1", "Decoder1", "DeepSup1",
    "Upsample2", "Decoder2", "DeepSup2",
    "Upsample3", "Decoder3", "DeepSup3",
    "Output"
]

for i, layer in enumerate(layers):
    ax.text(0.5, 1 - i*0.05, layer, fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightblue'))

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.axis('off')
plt.title("UNetX Architecture Diagrammatic View")
plt.tight_layout()
plt.savefig("unetx_architecture_diagram.png")
plt.show()

print("UNetX architecture diagram saved as 'unetx_architecture_diagram.png'")
