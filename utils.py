import torch 
import math
import numpy as np

from dataset import CustomImageDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

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

	weight = torch.from_numpy(weight_np).type(torch.FloatTensor).to('cuda')

	return torch.nn.functional.conv2d(img, weight, padding=1)

def HFEN(output, target):
	return torch.sum(torch.pow(LoG(output) - LoG(target), 2)) / torch.sum(torch.pow(LoG(target), 2))


def l1_norm(output, target):
	return torch.sum(torch.abs(output - target)) / torch.numel(output)

def l2_norm(output, target):
     return F.mse_loss(output, target)


def create_datasets(train_txt, val_txt, test_txt, inference_txt, data_path, features, batch_size=32):
    train_dataset = CustomImageDataset(train_txt, data_path, features)
    val_dataset = CustomImageDataset(val_txt, data_path, features)
    test_dataset = CustomImageDataset(test_txt, data_path, features)
    inference_dataset = CustomImageDataset(inference_txt, data_path, features)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, inference_loader, train_dataset.num_features


def adjust_learning_rate(optimizer, epoch):
    initial_lr = 0.001
    if epoch < 10:
        lr = initial_lr * (10 ** (epoch / 9))  # Geometric progression
    else:
        lr = initial_lr / np.sqrt(epoch + 1 - 9)  # 1/√t schedule
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"Epoch {epoch+1}, Learning rate: {lr}")
