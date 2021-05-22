import os
import pymeshlab
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--pth_off', type=str,
                    default='ycb_objects',
                    help='Path to ycb folder containing objects in off format.')
parser.add_argument('--pth_out', type=str,
                    default='ycb_converted',
                    help='Path to output folder with the converted objects.')
parser.add_argument('--out_format', type=str,
                    default='ply', choices=['ply', 'stl', 'obj'],
                    help='File output format.')

def main(args):
    if not os.path.exists(args.pth_out):
        os.makedirs(args.pth_out)

    off_objs = [obj for obj in os.listdir(args.pth_off) if ".off" in obj]

    for off_obj in off_objs:
        off_pth = os.path.join(args.pth_off, off_obj)

        # Load object
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(off_pth)

        # Save object
        out_obj = off_obj.replace(".off", "." + args.out_format)
        out_pth = os.path.join(args.pth_out, out_obj)
        ms.save_current_mesh(out_pth, save_face_color=False)


if __name__ == '__main__':
    main(parser.parse_args())