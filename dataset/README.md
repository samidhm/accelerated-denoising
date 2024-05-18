## Description

This script creates the dataset for our experiments, using Blender.
The script renders 64x64 images from the [Amazon Lumberyard Bistro Scene](https://developer.nvidia.com/orca/amazon-lumberyard-bistro)'s exterior 
for various camera positions and angles. Specifically, the script loads the `BistroExterior.fbx` file and renders several images at various sampling rates.
Refer to the script for the exact configurations and other specifications. 
The generated images are stored in the provided output folder and grouped by the number of samples per pixel.

The data has been generated and can be found [here](https://drive.google.com/file/d/13zf6nw3t_bk1mYhGTgfbS2rGK1Y2Et8f/view?usp=sharing).

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
