import itertools


if __name__ == "__main__":
    pi_by_180 = (3.14159 / 180)

    # Handpicked (x, y) coordinates which are good camera positions for the Bistro exterior scene
    train_points = [(23, 60), (20, 50), (-10, 10), (-12, -2), (-5, -10), (5, -14), (35, -26)] 
    train_points += [(45, -30), (50, -32), (53, -33), (56, -34), (60, -36), (60, -36), (77, -25)] 
    train_points += [(85, -35), (80, -52), (76, -57), (65, -55), (60, -50), (-39, 30), (-31, 37)]
    
    val_points = [(13, -18), (85, -30), (-2, 20), (13, 40), (-12, 4), (9, -16), (-35, 23)]
    
    test_points = [(-10, -6), (84, -42), (7, 30), (-25, 13), (40, -28), (70, -25), (70, -59)]

    inference_points = train_points + val_points + test_points

    heights = [i for i in range(2, 17)]
    camera_euler_rotations_x =  [60, 90, 120]
    camera_euler_rotations_z =  [i for i in range(0, 360, 45)]


    for points, filename in [(train_points, "train"), (val_points, "val"), (test_points, "test"), (inference_points, "inference")]:
        axes = [points, heights, camera_euler_rotations_x, camera_euler_rotations_z]
        names = []
        for (x, y), z, eu_x, eu_z in itertools.product(*axes):
            names.append(f"image_x_{x}_y_{y}_z_{z}_eux_{eu_x}_euz_{eu_z}.png")
        
        open(filename + ".txt", "w").write("\n".join(names))
        