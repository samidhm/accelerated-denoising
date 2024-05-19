## Description

This folder contains a dataset of rendered 64x64 images from the [Amazon Lumberyard Bistro Scene](https://developer.nvidia.com/orca/amazon-lumberyard-bistro)'s exterior 
for various camera positions and angles. Specifically, the `BistroExterior.fbx` scene is rendered at various sampling rates. The images rendered at `sample rate = 1` are considered noisy, while the high-quality versions are rendered using `sample rate = 512`.
Refer to the script for the exact configurations and other specifications. 
The generated images are stored and grouped by the number of samples per pixel in the respective folders (`samples_1` and `samples_512`). 
Other useful information such as the z depth,the absolute normal, the diffuse color, the glossy color and the view-space transformation matrix for the camera is also collected for each configuration.

The data has been generated and can be found [here](https://drive.google.com/file/d/185EjQrve5o6ZpfUDyv-Et5ZUEaEKsE7X/view).

There are two scripts also present in this folder.

1. `render.py`: The script used to generate the dataset. The script requires Blender to be installed and configured appropriately.

## Usage
```
Blender -b -P render.py -- [-h] [-f FBX_SCENE_PATH] [-o OUTPUT_FOLDER] [-s START] [-e END] [-g]

Script to render images for the Amazon Lumberyard Bistro script

options:
  -h, --help            show this help message and exit
  -f FBX_SCENE_PATH, --fbx_scene_path FBX_SCENE_PATH
                        Path to the BistroExterior.fbx file
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Folder where the rendering output is saved
  -s START, --start START
                        The first index (inclusive) of images to be rendered (as per the script's own rendering order)
  -e END, --end END     The last index (inclusive) of images to be rendered (as per the script's own rendering order)
  -g, --gpu             Use GPU (Nvidia/CUDA only)
```

2. `gen_split.py`: A script which generates 3 text files containing the names of images corresponding to a training,
validation and test split. The data is chosen such that each set has unique points from all sections of the scene.

## Usage
```
python gen_split.py
```
