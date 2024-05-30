import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, no_layers = 4, bottleneck = "conv"):
        super(UNet, self).__init__()

        #Define the encoder layers
        self.enc = nn.ModuleList()
        self.enc.append(self.conv_block(in_channels=in_channels, out_channels=64))
        for i in range(no_layers-1):
            self.enc.append(self.conv_block(in_channels=2**(i+6), out_channels=2**(i+7)))

        # Define the bottleneck layer
        if bottleneck == "conv":
            self.bottleneck = self.conv_block(in_channels=2**(no_layers+5), out_channels=2**(no_layers+6))

        # Define the decoder layers
        self.dec = nn.ModuleList()
        for j in range(no_layers, 0, -1):
            self.dec.append(self.upconv_block(in_channels=2**(j+6), out_channels=2**(j+5)))
            self.dec.append(self.conv_block(in_channels=2**(j+6), out_channels=2**(j+5)))


        # Define the final output layer
        self.final = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc_outputs = []
        ip = x
        for l in self.enc:
            ip = l(ip)
            enc_outputs.insert(0, ip)
            ip = F.max_pool2d(ip, 2)

        # Bottleneck
        ip = self.bottleneck(ip)

        for idx, l in enumerate(self.dec):
            if idx%2 == 0:
                ip = l(ip)
                ip = torch.cat((ip, enc_outputs[idx // 2]), dim=1)
            else:
                ip = l(ip)

        out = self.final(ip)
        return out