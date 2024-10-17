# CUDA_VISIBLE_DEVICES=7 nohup python -u main_benchmark.py --diffusion_model /dev/shm/alexJiang/source/stable-diffusion-v1-4 --output_dir /dev/shm/alexJiang/output/DiffDisEval/SD1_4 --dataset_root /dev/shm/alexJiang/eval_dataset/large_500_300_150_50/ --negative_token --num_cpu 16 --num_gpu 1 --num_runs 100 --seed 3407 > SD1_4.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python -u main_benchmark.py --diffusion_model /dev/shm/alexJiang/source/stable-diffusion-v1-4 --output_dir /dev/shm/alexJiang/output/DiffDisEval/SD1_4 --dataset_root /dev/shm/alexJiang/eval_dataset/small_200_30_20_10/ --negative_token --num_cpu 16 --num_gpu 1 --num_runs 2 --seed 3407 > SD1_4.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_benchmark.py --diffusion_model /dev/shm/alexJiang/source/stable-cascade --output_dir /dev/shm/alexJiang/output/DiffDisEval/stable-cascade --dataset_root /dev/shm/alexJiang/eval_dataset/large_500_300_150_50/ --seed 3407 --single_run > SC.log 2>&1 &