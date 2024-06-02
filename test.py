import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import argparse

from unet import UNet
from dataset import CustomImageDataset
import argparse
from tqdm import tqdm

from model_stats import SizeEstimator

def create_dataset(test_txt, data_path, batch_size=32):
    test_dataset = CustomImageDataset(test_txt, data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def create_datasets(train_txt, val_txt, test_txt, data_path, batch_size=32):
    train_dataset = CustomImageDataset(train_txt, data_path)
    val_dataset = CustomImageDataset(val_txt, data_path)
    test_dataset = CustomImageDataset(test_txt, data_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


#Get arguments for network configuration from command line
parser = argparse.ArgumentParser(description="UNet model training loop")
parser.add_argument("quantize", type=str, help="Choose quantization mode", default="none")
    
args = parser.parse_args()

train_loader, val_loader, test_loader = create_datasets("data/train.txt", "data/val.txt","data/test.txt", "data/raw_data")


device = torch.device("cuda")

#Load model checkpoint
model = UNet(16, 3).to(device)
model.load_state_dict(torch.load('unet_denoising.pth'))
model.eval()

print('Weights before quantization')
print(model.bottleneck.weight())

#Quantize model
if args.quantize == "ptdq":
    torch.ao.quantization.quantize_dynamic(model, dtype=torch.qint8, inplace=True)
    print('Weights after quantization')
    print(torch.int_repr(model.bottleneck.weight()))
elif args.quantize == "ptsq":
    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    torch.ao.quantization.prepare(model, inplace=True)
    #Calibrate by running one batch from validation set
    with torch.no_grad():
        print("Running calibration")
        for (noisy_image, _) in tqdm(test_loader):
            model(noisy_image)
            break
    
    torch.ao.quantization.convert(model, inplace= True)
    print('Weights after quantization')
    print(torch.int_repr(model.bottleneck.weight()))
else:
    pass

output_dir = 'denoised_test_images'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

test_files = open("data/test.txt", "r").read().strip().split("\n")
latencies = []

with torch.no_grad():
    #for idx, (noisy_image, _) in enumerate(test_loader):
    for idx, (noisy_image, clean_image), in enumerate(test_loader):
        noisy_image = noisy_image.to('cuda')
        
        t = time.time()
        outputs = model(noisy_image)
        latencies.append(time.time() - t)

        for i in range(outputs.size(0)):
            denoised_img = outputs[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
            denoised_img = (denoised_img * 255).astype(np.uint8)  # Convert to uint8
            img = Image.fromarray(denoised_img)
            img.save(os.path.join(images_dir, f'{test_files[idx * test_loader.batch_size + i]}'))
        
print(f"Denoised test images saved to {output_dir}")
