import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image

from unet import UNet
from dataset import CustomImageDataset

def create_dataset(test_txt, data_path, batch_size=32):
    test_dataset = CustomImageDataset(test_txt, data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

test_loader = create_dataset("data/test.txt", "data/raw_data")

device = torch.device("cuda")
model = UNet(10, 3).to(device)

model.load_state_dict(torch.load('unet_denoising.pth'))
model.eval()

output_dir = 'denoised_test_images'
os.makedirs(output_dir, exist_ok=True)

test_files = open("data/test.txt", "r").read().strip().split("\n")

with torch.no_grad():
    for idx, (noisy_image, clean_image, albedo_image_test), in enumerate(test_loader):
        noisy_image = noisy_image.to('cuda')
        albedo_image_test = albedo_image_test.to('cuda')
        outputs_temp = model(noisy_image)
        albedo_image_test = torch.sub(albedo_image_test, 0.0001)
        #outputs = torch.mul(outputs_temp,albedo_image_test)
        outputs = torch.mul(outputs_temp,1)
        for i in range(outputs.size(0)):
            denoised_img = outputs[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
            denoised_img = (denoised_img * 255).astype(np.uint8)  # Convert to uint8
            img = Image.fromarray(denoised_img)
            img.save(os.path.join(output_dir, f'{test_files[idx * test_loader.batch_size + i]}'))
        
print(f"Denoised test images saved to {output_dir}")
