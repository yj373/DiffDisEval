# CUDA_VISIBLE_DEVICES=0 nohup python -u main_classic_eval.py --diffusion_model_name stable-diffusion-v1-4 --output_dir /dev/shm/alexJiang/output/DiffDisEval/SD1_4/generated_images_1 --seed 3407 > SD1_4_eval_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python -u main_classic_eval.py --diffusion_model_name stable-cascade --output_dir /dev/shm/alexJiang/output/DiffDisEval/stable-cascade/generated_images_1 --generate_images --seed 3407 > SC_eval.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main_classic_eval.py --diffusion_model_name stable-cascade --output_dir /dev/shm/alexJiang/output/DiffDisEval/stable-cascade/generated_images_2 --generate_images --seed 340 > SC_eval_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u main_classic_eval.py --diffusion_model_name stable-cascade --output_dir /dev/shm/alexJiang/output/DiffDisEval/stable-cascade/generated_images_3 --generate_images --seed 34 > SC_eval_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u main_classic_eval.py --diffusion_model_name stable-cascade --output_dir /dev/shm/alexJiang/output/DiffDisEval/stable-cascade/generated_images_4 --generate_images --seed 3 > SC_eval_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -u main_classic_eval.py --diffusion_model_name stable-diffusion-v1-4 --output_dir /dev/shm/alexJiang/output/DiffDisEval/SD1_4/generated_images_2 --generate_images --seed 340 > SD1_4_eval_1.log 2>&1 &&
CUDA_VISIBLE_DEVICES=7 nohup python -u main_classic_eval.py --diffusion_model_name stable-diffusion-v1-4 --output_dir /dev/shm/alexJiang/output/DiffDisEval/SD1_4/generated_images_3 --generate_images --seed 34 > SD1_4_eval_1.log 2>&1 &&
CUDA_VISIBLE_DEVICES=7 nohup python -u main_classic_eval.py --diffusion_model_name stable-diffusion-v1-4 --output_dir /dev/shm/alexJiang/output/DiffDisEval/SD1_4/generated_images_4 --generate_images --seed 3 > SD1_4_eval_1.log 2>&1 &


