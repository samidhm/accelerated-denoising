import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Define the encoder layers
        self.enc1 = self.conv_block(in_channels=in_channels, out_channels=64)
        self.enc2 = self.conv_block(in_channels=64, out_channels=128)
        self.enc3 = self.conv_block(in_channels=128, out_channels=256)
        self.enc4 = self.conv_block(in_channels=256, out_channels=512)

        # Define the bottleneck layer
        self.bottleneck = self.conv_block(in_channels=512, out_channels=1024)

        # Define the decoder layers
        self.upconv4 = self.upconv_block(in_channels=1024, out_channels=512)
        self.dec4 = self.conv_block(in_channels=1024, out_channels=512)
        self.upconv3 = self.upconv_block(in_channels=512, out_channels=256)
        self.dec3 = self.conv_block(in_channels=512, out_channels=256)
        self.upconv2 = self.upconv_block(in_channels=256, out_channels=128)
        self.dec2 = self.conv_block(in_channels=256, out_channels=128)
        self.upconv1 = self.upconv_block(in_channels=128, out_channels=64)
        self.dec1 = self.conv_block(in_channels=128, out_channels=64)

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
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # Final output layer
        out = self.final(dec1)
        return out

model = UNet(8, 3)
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 image
output = model(input_tensor)
print(output)  # Should be torch.Size([1, 3, 64, 64])
