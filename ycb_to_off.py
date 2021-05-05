import os
import pymeshlab

# TODO: test

parser = argparse.ArgumentParser(description='Convert objects from ply to off')

parser.add_argument('--ycb_pth', type=str,
                    default='',
                    help="Path to ycb folder")  
parser.add_argument('--scale', default=1000, help="scales to apply to each model")

def main(args):
  # Create directories
  dirs = ["0_in", "1_scaled", "1_transform", "2_depth", "2_watertight", "4_points", "4_pointcloud", "4_watertight_scaled"]
  
  for dir_name in dirs:
    dir_pth = os.path.join(args.ycb_pth, dir_name)
    if not os.path.exists(dir_pth):
      os.makedirs(dir_pth)

  # Convert to off
  out_dir = os.path.join(args.ycb_pth, "0_in")

  for dir_name in os.listdir(args.ycb_pth):
      ply_pth = os.path.join(args.ycb_pth, dir_name, "google_16k", "nontextured.ply")
      out_pth = os.path.join(out_dir, f"{dir_name}.off")

      try:
          ms = pymeshlab.MeshSet()
          ms.load_new_mesh(ply_pth)
          
          # Scale object
          ms.transform_scale_normalize(axisx=args.scale, uniformflag=True)
          
          ms.save_current_mesh(out_pth,
                               save_vertex_color = False,
                               save_vertex_coord = False, 
                               save_face_color = False)
      except:
          pass

      
if __name__ == '__main__':
    main(parser.parse_args())
