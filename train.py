import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import copy

from unet import UNet
from dataset import CustomImageDataset

def LoG(img):
	weight = [
		[0, 0, 1, 0, 0],
		[0, 1, 2, 1, 0],
		[1, 2, -16, 2, 1],
		[0, 1, 2, 1, 0],
		[0, 0, 1, 0, 0]
	]
	weight = np.array(weight)

	weight_np = np.zeros((1, 1, 5, 5))
	weight_np[0, 0, :, :] = weight
	weight_np = np.repeat(weight_np, img.shape[1], axis=1)
	weight_np = np.repeat(weight_np, img.shape[0], axis=0)

	weight = torch.from_numpy(weight_np).type(torch.FloatTensor).to('cuda:0')

	return torch.nn.functional.conv2d(img, weight, padding=1)

def HFEN(output, target):
	return torch.sum(torch.pow(LoG(output) - LoG(target), 2)) / torch.sum(torch.pow(LoG(target), 2))


def l1_norm(output, target):
	return torch.sum(torch.abs(output - target)) / torch.numel(output)


def create_datasets(train_txt, val_txt, test_txt, data_path, batch_size=32):
    train_dataset = CustomImageDataset(train_txt, data_path)
    val_dataset = CustomImageDataset(val_txt, data_path)
    test_dataset = CustomImageDataset(test_txt, data_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Paths to the text files
split_file_folder = "data"
train_txt = f"{split_file_folder}/train.txt"
val_txt = f"{split_file_folder}/val.txt"
test_txt = f"{split_file_folder}/test.txt"

# Paths to the image folders
data_path = "data/raw_data"
train_loader, val_loader, test_loader = create_datasets(train_txt, val_txt, test_txt, data_path)

device = torch.device("cpu")
# Define the model, loss function, and optimizer
model = UNet(13, 3).to(device)
criterion = nn.MSELoss()  # Mean Squared Error loss for denoising
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 50
patience = 5  # Early stopping patience
best_loss = float('inf')
patience_counter = 0

# For saving the best model
best_model_wts = copy.deepcopy(model.state_dict())

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for noisy_image, clean_image in train_loader:
        # Move tensors to the appropriate device
        noisy_image, clean_image = noisy_image.to(device), clean_image.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(noisy_image)
        loss = 0.8 * l1_norm(outputs, clean_image) + 0.2 * HFEN(outputs, clean_image)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate the loss
        running_loss += loss.item() * noisy_image.size(0)
    
    # Calculate training loss
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy_image, clean_image in val_loader:
            noisy_image, clean_image = noisy_image.to(device), clean_image.to(device)
            outputs = model(noisy_image)
            loss = criterion(outputs, clean_image)
            val_loss += loss.item() * noisy_image.size(0)
    
    # Calculate validation loss
    val_loss /= len(val_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
    
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

# Save the model
torch.save(model.state_dict(), 'unet_denoising.pth')