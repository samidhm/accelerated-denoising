from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import argparse
from utils import *
from tqdm import tqdm
import os
import json

from unet import UNet

#Get arguments for network configuration from command line
parser = argparse.ArgumentParser(description="UNet model training loop")
parser.add_argument("-n", "--num_layers", type=int, default=4, help="No of encoder and decoder layers in the UNet network")
parser.add_argument("-b", "--bottleneck", type=str, default="conv", help="Choose architecture of bottleneck layer, conv")
parser.add_argument("-f", "--features", nargs='+', default=["depth", "normal", "relative_normal", "albedo", "roughness"], help="Features to include")
parser.add_argument("-t", "--tag", type=str, default="", help="Tag to be added for while saving files")
parser.add_argument("-a", "--alpha", type=str, default=0.5, help="Coefficient for L1 loss")

args = parser.parse_args()

# Paths to the text files
split_file_folder = "data"
train_txt = f"{split_file_folder}/train.txt"
val_txt = f"{split_file_folder}/val.txt"
test_txt = f"{split_file_folder}/test.txt"

# Paths to the image folders
data_path = "data/raw_data"
train_loader, val_loader, test_loader, num_features = create_datasets(train_txt, val_txt, test_txt, data_path, args.features)

device = torch.device("cuda")
# Define the model, loss function, and optimizer
model = UNet(num_features, 3, args.num_layers, args.bottleneck).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 5
patience = 5  # Early stopping patience
best_loss = float('inf')
patience_counter = 0

training_loss = []
validation_loss = []
validation_psnr = []

# For saving the best model
best_model_wts = copy.deepcopy(model.state_dict())

model.train()

# Training loop
for epoch in range(num_epochs):
    adjust_learning_rate(optimizer, epoch)
    running_loss = 0.0
    t = time()
    for noisy_image, clean_image in tqdm(train_loader):
        # Move tensors to the appropriate device
        noisy_image, clean_image = noisy_image.to(device), clean_image.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(noisy_image)
        #print('SAMIDH :::', outputs.dtype)
        
        loss = args.alpha * l1_norm(outputs, clean_image) + (1-args.alpha) * HFEN(outputs, clean_image)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate the loss
        running_loss += loss.item() * noisy_image.size(0)
    
    # Calculate training loss
    epoch_loss = running_loss / len(train_loader.dataset)
    training_loss.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Time: {time()-t}s')

    # Validation phase
    model.eval()
    val_loss = 0.0
    psnr_values = []
    with torch.no_grad():
        for noisy_image, clean_image in tqdm(val_loader):
            noisy_image, clean_image = noisy_image.to(device), clean_image.to(device)
            outputs = model(noisy_image)
            loss = args.alpha * l1_norm(outputs, clean_image) + (1-args.alpha) * HFEN(outputs, clean_image)
            val_loss += loss.item() * noisy_image.size(0)
            psnr_values += [psnr(outputs[i], clean_image[i]) for i in range(outputs.size(0))]
    
    # Calculate validation loss
    val_loss /= len(val_loader.dataset)
    validation_loss.append(val_loss)
    validation_psnr.append(np.mean(psnr_values))
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation PSNR: {validation_psnr[-1]}')
    
    # Check for early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# Load the best model weights
model.load_state_dict(best_model_wts)

os.makedirs("results", exist_ok=True)
feature_string = '_'.join([x[:3] for x in args.features])

if args.tag != "":
    experiment_name = f"{args.tag}_"
else:
    experiment_name = ""

experiment_name += f"n_{args.num_layers}_alpha_{args.alpha:.2f}_feat_{feature_string}"

folder = f"results/{experiment_name}"
os.makedirs(folder, exist_ok=True)

# Save the model
torch.save(model.state_dict(), f"{folder}/checkpoint.pth")

config = {
    "n": args.num_layers,
    "features": args.features,
    "tag": args.tag,
    "alpha": args.alpha
}

with open(f"{folder}/config.json", 'w') as fp:
    json.dump(config, fp)

open(f"{folder}/training_loss.txt", "w").write("\n".join([str(x) for x in training_loss]))
open(f"{folder}/validation_loss.txt", "w").write("\n".join([str(x) for x in validation_loss]))
open(f"{folder}/validation_psnr.txt", "w").write("\n".join([str(x) for x in validation_psnr]))

