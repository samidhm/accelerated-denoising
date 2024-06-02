import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, txt_file, data_path, include=["depth", "normal", "relative_normal", "albedo", "roughness"]):
        self.data_path = data_path
        self.image_names = self._read_txt_file(txt_file)
        self.include = include

        self.num_features = 3

        sizes = {"depth": 1, "normal": 3, "relative_normal": 3, "albedo": 3, "roughness": 3}
        self.num_features += sum([sizes[x] for x in include])

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
        input_images = [self._load_image(f"{self.data_path}/samples_1", image_name)]

        if "depth" in self.include:
            input_images.append(self._load_image(f"{self.data_path}/depth", image_name, True))
        
        normal_image = self._load_image(f"{self.data_path}/normal", image_name)
        if "normal" in self.include:
            input_images.append(normal_image)

        if "relative_normal" in self.include:
            view_matrix = np.load(f"{self.data_path}/view_space_matrix/{image_name[:-3]}npy")
            view_matrix = torch.tensor(view_matrix, dtype=input_images[-1].dtype).reshape(1, 1, 3, 3)

            relative_normal = normal_image.transpose(-1, -3)
            current_shape = relative_normal.shape
            relative_normal = relative_normal.reshape(*current_shape, 1)
            relative_normal = (view_matrix @ relative_normal).reshape(*current_shape)
            relative_normal = torch.nn.functional.normalize(relative_normal, dim=-1)
            relative_normal = relative_normal.transpose(-1, -3) 

            assert relative_normal.shape == normal_image.shape
            input_images.append(relative_normal)

        if "albedo" in self.include:
            input_images.append(self._load_image(f"{self.data_path}/diffuse_color", image_name) + \
                                self._load_image(f"{self.data_path}/glossy_color", image_name))
        
        
        # Concatenate input images along the channel dimension
        input_tensor = torch.cat(input_images, dim=0)

        # Load output image
        output_image = self._load_image(f"{self.data_path}/samples_512", image_name)

        return input_tensor, output_image


