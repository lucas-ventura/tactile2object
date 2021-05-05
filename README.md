# tactile2object


## Generate grasps
First, we need to generate the grasps using the graspIt software. You can generate grasps from the YCB dataset following the instructions [here](https://github.com/lucas-ventura/mano_grasp#generate-grasps-from-ybc-dataset).

## Prepare objects for the [Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks) model
We will follow the instructions from [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks). Please read the [readme](https://github.com/autonomousvision/occupancy_networks/tree/master/external/mesh-fusion) and try their examples. 

If you have probleams with `meshlabserver`, you can use `pymeshlab` instead. You will need to do this modifications to the `3_simplify.py` script:
```diff
+    import pymeshlab
```
```diff
-    os.system('meshlabserver -i %s -o %s -s %s' % (
-       filepath,
-       os.path.join(self.options.out_dir, ntpath.basename(filepath)),
-       self.simplification_script
-       ))
+    ms = pymeshlab.MeshSet()
+    ms.load_new_mesh(filepath)
+    ms.load_filter_script(self.simplification_script)
+    ms.apply_filter_script()
+    ms.save_current_mesh(os.path.join(self.options.out_dir, ntpath.basename(filepath)))
```

After checking that you can reproduce the watertight and simplified meshes from the demo examples we can do the same with our data.


**1. Run the script `ycb_to_off.py`. This will convert the ycb objects to `.off`.**

**2. Declare environment variables:**
```bash
export MESHFUSION_PATH=PATH_TO/external/mesh-fusion
export build_path_c=PATH_TO_YCB
```

**3. Scale meshes:**
```bash
python $MESHFUSION_PATH/1_scale.py \
    --in_dir $build_path_c/0_in \
    --out_dir $build_path_c/1_scaled \
    --t_dir $build_path_c/1_transform
```

**4. Create depths maps:**
```bash
python $MESHFUSION_PATH/2_fusion.py \
  --mode=render \
  --in_dir $build_path_c/1_scaled \
  --out_dir $build_path_c/2_depth
```

**5. Produce watertight meshes:**
```bash
python $MESHFUSION_PATH/2_fusion.py \
  --mode=fuse \
  --in_dir $build_path_c/2_depth \
  --out_dir $build_path_c/2_watertight \
  --t_dir $build_path_c/1_transform
```

**6. Process watertight meshes:**
```bash
python sample_mesh.py PATH_TO_YCB/2_watertight \
      --n_proc $NPROC --resize \
      --bbox_in_folder PATH_TO_YCB/0_in \
      --pointcloud_folder PATH_TO_YCB/4_pointcloud \
      --points_folder PATH_TO_YCB/4_points \
      --mesh_folder PATH_TO_YCB/4_watertight_scaled \
      --packbits --float16
```
