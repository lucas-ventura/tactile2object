import os
import pymeshlab
import argparse

parser = argparse.ArgumentParser(description='Convert objects from ply to off')
parser.add_argument('--ycb_in', type=str,
                    default='C:/Users/lucas/Desktop/UPC/MIT/graspit/YCB_dataset/models/ycb',
                    help="Path to models/ycb")
parser.add_argument('--ycb_out', type=str,
                    default='ycb',
                    help="Where to save the data")
parser.add_argument('--scale', default=1000, help="Scale to apply to each model")
parser.add_argument('-v', '--vertices', type=int, default=16, choices=[16, 64, 512], help="Number of k vertices to use")

def main(args):
    # Create directories
    dirs = ["0_in", "1_scaled", "1_transform", "2_depth", "2_watertight", "4_points", "4_pointcloud", "4_watertight_scaled"]

    for dir_name in dirs:
        dir_pth = os.path.join(args.ycb_out, dir_name)

        if not os.path.exists(dir_pth):
            os.makedirs(dir_pth)

    # Convert to off
    for obj_name in os.listdir(args.ycb_in):
        ply_pth = os.path.join(args.ycb_in, obj_name, "google_{}k".format(args.vertices), 'nontextured.ply')

        if os.path.exists(ply_pth):
            out_pth = os.path.join(args.ycb_out, "0_in", f"{obj_name}.off")

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(ply_pth)

            # Scale object
            ms.transform_scale_normalize(axisx=args.scale, uniformflag=True)

            ms.save_current_mesh(out_pth,
                                 save_vertex_color=False,
                                 save_vertex_coord=False,
                                 save_face_color=False)

      
if __name__ == '__main__':
    main(parser.parse_args())
