import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import argparse

from unet import UNet
from dataset import CustomImageDataset

from model_stats import SizeEstimator

def create_dataset(test_txt, data_path, batch_size=32):
    test_dataset = CustomImageDataset(test_txt, data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

parser = argparse.ArgumentParser(description="UNet inference training script")

parser.add_argument("-e", "--experiment", default="unet_denoising", type=str, help="Name of the experiment to evaluate")
parser.add_argument("-b", "--batch_size", default=1024, type=int, help="Inference batch size")

args = parser.parse_args()

# TODO: Infer model size and dataset features
test_loader = create_dataset("data/test.txt", "data/raw_data", args.batch_size)

device = torch.device("cuda")
model = UNet(13, 3).to(device)

model.load_state_dict(torch.load(f"{args.experiment}.pth"))
model.eval()

output_dir = os.path.join('evaluation_results', args.experiment)
images_dir = os.path.join(output_dir, args.experiment, "images")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

test_files = open("data/test.txt", "r").read().strip().split("\n")
latencies = []

with torch.no_grad():
    for idx, (noisy_image, _) in enumerate(test_loader):
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