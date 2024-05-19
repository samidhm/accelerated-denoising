import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, txt_file, data_path):
        self.data_path = data_path
        self.input_folders = ['/samples_1', '/depth', '/diffuse_color', '/glossy_color', '/normal']
        self.input_folders = [f"{data_path}/{x}" for x in self.input_folders]
        self.output_folder = f"{data_path}/samples_512"
        self.image_names = self._read_txt_file(txt_file)

    def _read_txt_file(self, txt_file):
        with open(txt_file, 'r') as file:
            image_names = file.read().splitlines()
        return image_names

    def _load_image(self, folder, image_name, grayscale=False):
        image_path = os.path.join(folder, image_name)
        image = Image.open(image_path)
        if grayscale:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        image = transforms.ToTensor()(image)
        return image

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        # Load input images
        input_images = []
        for folder in self.input_folders:
            grayscale = folder.endswith('depth')
            img = self._load_image(folder, image_name, grayscale)
            input_images.append(img)
        
        # Concatenate input images along the channel dimension
        input_tensor = torch.cat(input_images, dim=0)

        # Load output image
        output_image = self._load_image(self.output_folder, image_name)

        return input_tensor, output_image

def create_datasets(train_txt, val_txt, test_txt, data_path, batch_size=32):
    train_dataset = CustomImageDataset(train_txt, data_path)
    val_dataset = CustomImageDataset(val_txt, data_path)
    test_dataset = CustomImageDataset(test_txt, data_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Paths to the text files
split_file_folder = "dataset"
train_txt = f"{split_file_folder}/train.txt"
val_txt = f"{split_file_folder}/val.txt"
test_txt = f"{split_file_folder}/test.txt"

# Paths to the image folders
data_path = "dataset/data"
train_loader, val_loader, test_loader = create_datasets(train_txt, val_txt, test_txt, data_path)

# Example usage
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)
    break
