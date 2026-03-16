# bash scripts/vis_dataset.sh

source /home/hxd/miniconda3/bin/activate vis_dp3


dataset_path=/home/hxd/桌面/dp3_bin/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/training_data_example

vis_cloud=0
cd /home/hxd/桌面/dp3_bin/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
python vis_dataset.py --dataset_path $dataset_path \
                    --use_img 1 \
                    --vis_cloud ${vis_cloud} \
                    --use_pc_color 1 \
                    --downsample 1 \