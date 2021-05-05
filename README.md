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


**1. Run the script `ycb_to_off.py`. This will scale and convert the ycb objects to `.off`.**

**2. Run ``build_ybc.sh``**

Move the script to the scripts folder:
```bash
mv build_ybc.sh PATH/TO/occupancy_networks/scripts
```

Run the script:
```bash
bash build_ybc.sh
```
This will scale the meshes, create the depth maps, and produce and process the watertight meshes


**3. Run ``grasps_to_con.py``:**
```bash
grasps_to_con.py \
    --pressupre_pth PATH/TO/manopth/outputs/graspit_to_mano/ycb/ \
    --ycb_pth PATH/TO/occupancy_networks/data/ycb
```
This will generate the npz files needed to run the model from Convolutional Occupancy Networks.


**4. Create symbolic link to dataset:**
```bash
cd PATH/TO/convolutional_occupancy_networks
ln -s PATH/TO/occupancy_networks/data/ycb/ycb_con/ data/
```