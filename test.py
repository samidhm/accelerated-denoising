import json
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import argparse

from unet import UNet
import argparse
from tqdm import tqdm

from utils import *

from model_stats import SizeEstimator


#Get arguments for network configuration from command line
parser = argparse.ArgumentParser(description="UNet model training loop")
parser.add_argument("-q", "--quantize", type=str, help="Choose quantization mode", default="none", choices=["none", "ptdq", "ptsq"])
parser.add_argument("-e", "--experiment", default="unet_denoising", type=str, help="Name of the experiment to evaluate")
parser.add_argument("-b", "--batch_size", default=1260, type=int, help="Inference batch size")

args = parser.parse_args()

folder = f"results/{args.experiment}"

config = json.load(open(f"{folder}/config.json"))

train_loader, val_loader, test_loader, num_features = \
        create_datasets("data/train.txt", "data/val.txt","data/test.txt", "data/raw_data", config["features"], args.batch_size)


device = torch.device("cuda")

#Load model checkpoint
model = UNet(num_features, 3, config["n"]).to(device)
model.load_state_dict(torch.load(f"{folder}/checkpoint.pth"))
model.eval()

print('Weights before quantization')
print(model.bottleneck[0].weight)

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
    
    torch.ao.quantization.convert(model, inplace=True)
    print('Weights after quantization')
    print(torch.int_repr(model.bottleneck.weight()))
else:
    pass

output_dir = f"{folder}/eval_quant_{args.quantize}"
images_dir = f"{output_dir}/images"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

test_files = open("data/test.txt", "r").read().strip().split("\n")
latencies = []
psnr_values = []

with torch.no_grad():
    for idx, (noisy_image, gold) in enumerate(test_loader):
        noisy_image = noisy_image.to('cuda')
        
        t = time.time()
        outputs = model(noisy_image)
        latencies.append(time.time() - t)

        for i in range(outputs.size(0)):
            psnr_values.append(psnr(outputs[i].cpu(), gold))
            #denoised_img = outputs[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
            #denoised_img = (denoised_img * 255).astype(np.uint8)  # Convert to uint8
            denoised_img = outputs[i].to(torch.float32).cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
            denoised_img = (denoised_img * 255).astype(np.uint8)
            img = Image.fromarray(denoised_img)
            img.save(os.path.join(images_dir, f'{test_files[idx * test_loader.batch_size + i]}'))

out = {
    "batch_size": args.batch_size,
    "avg_batch_latency": np.mean(latencies),
    "model_size": os.path.getsize(f"{folder}/checkpoint.pth") / 1e3,
    "avg_psnr": np.mean(psnr_values)
}    

with open(f"{output_dir}/stats.json", 'w') as fp:
    json.dump(out, fp)

print(f"Denoised test results saved to {output_dir}")
