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

1. Run the script `ycb_to_off.py`. This will convert the ycb objects to `.off`.
