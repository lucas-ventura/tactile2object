#!/bin/bash

# Path to where 0_in is located
build_path_c=../data/ybc
MESHFUSION_PATH=../external/mesh-fusion
NPROC=12
export DISPLAY=:2

if [[ ! -d $build_path_c/0_in ]]; then
    echo "0_in not in directory!"
    exit
fi


mkdir -p $build_path_c/1_scaled \
         $build_path_c/1_transform \
         $build_path_c/2_depth \
         $build_path_c/2_watertight \
         $build_path_c/4_points \
         $build_path_c/4_pointcloud \
         $build_path_c/4_watertight_scaled 


echo "Scaling meshes"
python $MESHFUSION_PATH/1_scale.py \
    --n_proc $NPROC \
    --in_dir $build_path_c/0_in \
    --out_dir $build_path_c/1_scaled \
    --t_dir $build_path_c/1_transform

# You might need to remove the DISPLAY=:2
echo "Create depths maps"
python $MESHFUSION_PATH/2_fusion.py \
    --mode=render --n_proc $NPROC \
    --in_dir $build_path_c/1_scaled \
    --out_dir $build_path_c/2_depth

echo "Produce watertight meshes"
python $MESHFUSION_PATH/2_fusion.py \
    --mode=fuse --n_proc $NPROC \
    --in_dir $build_path_c/2_depth \
    --out_dir $build_path_c/2_watertight \
    --t_dir $build_path_c/1_transform

echo "Process watertight meshes"
python sample_mesh.py $build_path_c/2_watertight \
    --n_proc $NPROC --resize \
    --bbox_in_folder $build_path_c/0_in \
    --pointcloud_folder $build_path_c/4_pointcloud \
    --points_folder $build_path_c/4_points \
    --mesh_folder $build_path_c/4_watertight_scaled \
    --packbits --float16
