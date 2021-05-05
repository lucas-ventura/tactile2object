import os
import open3d as o3d
import numpy as np
import pickle
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser(
    description='Generate data for Convolutional Occupancy Networks'
)

# TODO: delete default
parser.add_argument('--pressupre_pth', type=str,
                    default="C:/Users/lucas/Desktop/UPC/MIT/manopth/outputs/graspit_to_mano/ycb/",
                    help="Path to the directory with all the folders with the pickle pressure files")
parser.add_argument('--ycb_pth', type=str,
                    default="C:/Users/lucas/Desktop/UPC/MIT/tactile2object/ycb",
                    help="PATH/TO/occupancy_networks/data/ycb")

get_obj_name = lambda name: name[:len(name) - 14]


def main(args):
    pressure_pointcloud_path = os.path.join(args.ycb_pth, "5_pressure_pointcloud")

    if not os.path.exists(pressure_pointcloud_path):
        os.makedirs(pressure_pointcloud_path)

    # Convert pressure info to pointcloud
    print("Saving pressure pointclouds")
    obj_names = []

    for obj_folder in os.listdir(args.pressupre_pth):
        pkls_path = os.path.join(args.pressupre_pth, obj_folder)
        obj_name = get_obj_name(obj_folder)
        obj_names.append(obj_name)

        # Get sensor info from every pkl file
        all_sensors_xyz = []
        all_sensors_pressure = []

        for pkl_name in os.listdir(pkls_path):
            pkl_path = os.path.join(pkls_path, pkl_name)

            pressure_info = pickle.load(open(pkl_path, "rb"))

            # Load the sensors pressure and take only the ones that have pressure greater than 0
            sensors_pressure = pressure_info['sensors_pressure']
            idx_sensors = np.nonzero(sensors_pressure)

            sensors_xyz = pressure_info['sensors_xyz'][idx_sensors]
            sensors_pressure = sensors_pressure[idx_sensors]

            all_sensors_xyz.append(sensors_xyz)
            all_sensors_pressure.append(sensors_pressure)

        sensors_xyz = np.concatenate(all_sensors_xyz)
        # sensors_pressure = np.concatenate(all_sensors_pressure)

        # Point cloud of the geneated object
        pcd_gen = o3d.geometry.PointCloud()
        pcd_gen.points = o3d.utility.Vector3dVector(sensors_xyz)

        pcd_gen.estimate_normals()
        pcd_gen.orient_normals_consistent_tangent_plane(50)

        ply_pth = os.path.join(pressure_pointcloud_path, f"{obj_name}.ply")
        o3d.io.write_point_cloud(ply_pth, pcd_gen)

    # Generate data for Convolutional Occupancy Networks
    # It generates the pointcloud.npz files with the pressure info
    print("Converting data for Convolutional Occupancy Networks")
    pressure_points_path = os.path.join(ycb_dir, "6_pressure_pointcloud")

    if not os.path.exists(pressure_points_path):
        os.makedirs(pressure_points_path)

    for obj_name in obj_names:
        pc_pth = os.path.join(ycb_dir, "4_pointcloud", f"{obj_name}.npz")
        pc_npz = np.load(pc_pth)

        translation = pc_npz['loc'].tolist()
        scale = pc_npz['scale'].item()

        ply_pth = os.path.join(pressure_pointcloud_path, f"{obj_name}.ply")
        pcd_pressure = o3d.io.read_point_cloud(ply_pth)

        points = (np.asarray(pcd_pressure.points) - translation) / scale
        normals = np.asarray(pcd_pressure.normals)

        npz_pth = os.path.join(pressure_points_path, obj_name)
        np.savez(npz_pth,
                 points=points,
                 normals=normals,
                 loc=pc_npz['loc'],
                 scale=pc_npz['scale']
                 )


    # Create dataset
    print("Generating ycb_con folder")
    dataset_pth = os.path.join(args.ycb_pth, "ycb_con")

    if not os.path.exists(dataset_pth):
        os.makedirs(dataset_pth)

    for obj_name in obj_names:
        obj_dir = os.path.join(dataset_pth, obj_name)

        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)

        src_pth_ps = os.path.join(args.ycb_pth, "4_points", f"{obj_name}.npz")
        dst_pth_ps = os.path.join(obj_dir, "points.npz")
        copyfile(src_pth_ps, dst_pth_ps)

        src_pth_pcd = os.path.join(pressure_points_path, f"{obj_name}.npz")
        dst_pth_pcd = os.path.join(obj_dir, "pointcloud.npz")
        copyfile(src_pth_pcd, dst_pth_pcd)


    # Create splits
    print("Create splits")
    n = len(obj_names)
    n_train, n_val = int(n * 0.6), int(n * 0.2)
    n_test = n - n_train - n_val

    np.random.shuffle(obj_names)

    training, val, test = obj_names[:n_train], obj_names[n_train:(n_train + n_val)], obj_names[-n_test:]
    train_pth = os.path.join(dataset_pth, 'train.lst')
    val_pth = os.path.join(dataset_pth, 'val.lst')
    test_pth = os.path.join(dataset_pth, 'test.lst')

    with open(train_pth, 'w') as filehandle:
        for obj_name in training:
            filehandle.write('%s\n' % obj_name)

    with open(val_pth, 'w') as filehandle:
        for obj_name in val:
            filehandle.write('%s\n' % obj_name)

    with open(test_pth, 'w') as filehandle:
        for obj_name in test:
            filehandle.write('%s\n' % obj_name)


if __name__ == '__main__':
    main(parser.parse_args())
