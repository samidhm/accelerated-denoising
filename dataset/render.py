import bpy
import sys
import time
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to render images for the Amazon Lumberyard Bistro script")

    parser.add_argument("-f", "--fbx_scene_path", default="./BistroExterior.fbx", type=str, help="Path to the BistroExterior.fbx file")
    parser.add_argument("-o", "--output_folder", default=".", type=str, help="Folder where the rendering output is saved")
    parser.add_argument("-s", "--start", type=int, default=0, help="The first index (inclusive) of images to be rendered " \
                                                                   "(as per the script's own rendering order)")
    parser.add_argument("-e", "--end", type=int, default= None, help="The last index (inclusive) of images to be rendered " \
                                                                     "(as per the script's own rendering order)")
    parser.add_argument("-g", "--gpu", action='store_true', help="Use GPU (Nvidia/CUDA only)")

    # Parse arguments after --
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


pi_by_180 = (3.14159 / 180)

# Handpicked (x, y) coordinates which are good camera positions for the Bistro exterior scene
points = [(23, 60), (20, 50), (13, 40), (7, 30), (-2, 20), (-10, 10), (-12, 4), (-12, -2), (-10, -6), (-5, -10)]
points += [(-5, -10), (-5, -10), (5, -14), (9, -16), (13, -18), (35, -26), (40, -28), (45, -30), (50, -32), (53, -33)]
points += [(56, -34), (60, -36), (60, -36), (70, -25), (77, -25), (85, -30), (85, -35), (84, -42), (80, -52), (76, -57)]
points += [(70, -59), (65, -55), (60, -50), (-25, 13), (-35, 23), (-39, 30), (-31, 37)]

sample_count = [1, 3, 5, 512]
heights = [i for i in range(2, 17)]
camera_euler_rotations_x =  [60, 90, 120]
camera_euler_rotations_z =  [i for i in range(0, 360, 45)]

total_images_to_render = len(points) * len(sample_count) * \
                       len(heights) * len(camera_euler_rotations_x) * len(camera_euler_rotations_z)



if __name__ == "__main__":
    args = parse_arguments()

    bpy.ops.import_scene.fbx(filepath = args.fbx_scene_path)

    camera = bpy.data.objects.get('Camera')
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.cycles.use_adaptive_sampling = False
    bpy.context.scene.render.resolution_x = 64
    bpy.context.scene.render.resolution_y = 64

    if args.gpu:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.scene.cycles.device = 'GPU'

        # Enable all available GPUs
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            if device.type == 'CUDA':
                device.use = True

    cnt = 0
    start = args.start
    end = args.end if args.end is not None else total_images_to_render

    print(f"Rendering {end-start+1} images, with indices {start}..{end}\n", file=sys.stderr)

    for samples in sample_count:
        bpy.context.scene.cycles.samples = samples
        folder_path = args.output_folder + '/pix_' + str(samples)
        
        for x, y in points:
            camera.location[0] = x
            camera.location[1] = y
            
            for z in heights:
                camera.location[2] = z

                for eu_x in camera_euler_rotations_x:
                    for eu_z in camera_euler_rotations_z:
                        
                        cnt += 1
                        if cnt < start or cnt > end:
                            print(f"Skipping image {cnt}/{total_images_to_render}", file=sys.stderr)
                            continue

                        camera.rotation_euler[0] = eu_x * pi_by_180
                        camera.rotation_euler[1] = 0
                        camera.rotation_euler[2] = eu_z * pi_by_180

                        file_name = f'image_x_{x}_y_{y}_z_{z}_eux_{eu_x}_euz_{eu_z}'
                        bpy.context.scene.render.filepath = f'{folder_path}/{file_name}.png'

                        print(f"Rendering image {cnt}/{total_images_to_render}, with parameters " \
                               f"x_{x}_y_{y}_z_{z}_eux_{eu_x}_euz_{eu_z}, samples={samples}", file=sys.stderr)

                        t = time.time()
                        
                        bpy.ops.render.render(write_still=True)
                        
                        print(f"Rendered {cnt}/{total_images_to_render}, in {time.time() - t} sec.\n", file=sys.stderr)
