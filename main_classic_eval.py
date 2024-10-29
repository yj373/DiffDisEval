import torch
import os
import shutil
import argparse
# import ImageReward as RM
# import hpsv2
import numpy as np

from diffusers import StableDiffusionPipeline, DiffusionPipeline, StableCascadeDecoderPipeline, StableCascadePriorPipeline


PROMPTS = [
    "there are two buses parked in a parking lot next to each other",
    "there is a small pony that is standing on the ground",
    "there is a dog that is sitting in a dog house",
    "aerial view of a neighborhood with a red tree in the foreground",
    "aerial view of a house with a yellow roof and red trees",
    "there is a small bird sitting on a branch in a tree",
    "cars are driving down the street in a city with tall trees",
    "cars and trucks driving down a street in a city",
    "there are three cows standing in a field of tall grass",
    "boats are in the water near a small island with mountains in the background",
    "there is a bike that is parked on a trail in the woods",
    "there is a train that is going down the tracks with people watching",
    "there is a man and woman hugging on the beach",
    "there is a red couch in a living room with pictures on the wall",
    "there is a computer desk with a monitor and a keyboard",
    "there are many bottles of beer and some food on a table",
    "there is a plate of food on a table with a glass of water",
    "there are two planes flying in the sky together",
    "there is a potted plant sitting on a table next to a window",
    "yellow school bus parked in a parking lot at night",
    "there is a dog sitting in a living room with a christmas chair",
    "there is a motorbike parked on the street in front of a building",
    "there is a white cow with horns standing in a field",
    "there is a black and white cat laying on a bed",
    "yellow bird with a white beak sitting on a branch in a cage",
    "there is a cat that is laying on the pillow on a chair",
    "there are many sheep standing in a field of grass together",
    "woman standing in front of a christmas tree in a hotel lobby",
    "there is a very large engine that is on display in a museum",
    "there is a couch and a table in a room",
    "there is a table with a white table cloth and a white table cloth",
    "there is a bottle of champagne sitting on a shelf",
    "there are two computer monitors on a desk with a keyboard and mouse",
    "an aerial view of a plane flying over a field with a building in the background",
    "vegetations and motorcycles driving down a street with a line of trees",
    "there is a man walking across the street with a backpack",
    "aerial view of a city with a red tree and a street",
    "aerial view of a residential neighborhood with a red tree in the foreground",
    "there are many person riding bikes on the street in the city",
    "aerial view of a house in a red forest with a roof"
]
MODEL_ROOT = "/dev/shm/alexJiang/source"


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images')
    parser.add_argument(
        "--diffusion_model_name",
        type=str,
        default="stable-diffusion-v1-4",
        required=True,
        help="diffusion model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/dev/shm/alexJiang/output/DiffDisEval/SD1_4/generated_images",
        help="directory containing the generated images",
    )
    parser.add_argument(
        "--generate_images",
        action="store_true",
        help="whether to generate new images"
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=4,
        help="number of generated images for each prompt",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=25,
        help="number of prompts used to synthesize images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="random seed"
    )
    args = parser.parse_args()
    return args


def same_seeds(seed):
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True 


def generate_images(args):
    model_id = os.path.join(MODEL_ROOT, args.diffusion_model_name)
    prompts = PROMPTS[:args.num_prompts]
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)
    if args.diffusion_model_name in ["SD1_4", "SD1_5", "openjourney"]:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        for i, prompt in enumerate(prompts):
            for j in range(args.num_images_per_prompt):
                image = pipe(prompt, height=512, width=512, num_inference_steps=50).images[0]
                image.save(os.path.join(args.output_dir, "{}-{}.png".format(i, j)))

    elif args.diffusion_model_name.startswith("SDXL"):
        base_id = os.path.join(model_id, "stable-diffusion-xl-base-1.0")
        base = DiffusionPipeline.from_pretrained(base_id, torch_dtype=torch.float16, use_safetensors=True)
        base = base.to("cuda")

        refiner_id = os.path.join(model_id, "stable-diffusion-xl-refiner-1.0")
        refiner = DiffusionPipeline.from_pretrained(refiner_id, text_encoder_2=base.text_encoder_2, vae=base.vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        refiner = refiner.to("cuda")

        for i, prompt in enumerate(prompts):
            for j in range(args.num_images_per_prompt):
                image = base(
                    prompt=prompt,
                    height=512,
                    width=512,
                    num_inference_steps=50,
                    denoising_end=0.8,
                    output_type="latent",
                ).images
                image = refiner(
                    prompt=prompt,
                    height=512,
                    width=512,
                    num_inference_steps=50,
                    denoising_start=0.8,
                    image=image,
                ).images[0]
                image.save(os.path.join(args.output_dir, "{}-{}.png".format(i, j)))

    elif args.diffusion_model_name.startswith("stable-cascade"):
        prior_id = os.path.join(model_id, "stable-cascade-prior")
        prior = StableCascadePriorPipeline.from_pretrained(prior_id, variant="bf16", torch_dtype=torch.bfloat16).to("cuda")

        decoder_id = os.path.join(model_id, "stable-cascade")
        decoder = StableCascadeDecoderPipeline.from_pretrained(decoder_id, variant="bf16", torch_dtype=torch.float16).to("cuda")

        for i, prompt in enumerate(prompts):
            for j in range(args.num_images_per_prompt):
                prior_output = prior(
                    prompt=prompt,
                    height=512,
                    width=512,
                    negative_prompt="",
                    guidance_scale=7.5,
                    num_images_per_prompt=1,
                    num_inference_steps=40
                )
                decoder_output = decoder(
                    image_embeddings=prior_output.image_embeddings.to(torch.float16),
                    prompt=prompt,
                    negative_prompt="",
                    guidance_scale=0.0,
                    output_type="pil",
                    num_inference_steps=10
                ).images[0]
                decoder_output.save(os.path.join(args.output_dir, "{}-{}.png".format(i, j)))
    else:
        print("unsupported diffusion model: {}".format(args.diffusion_model_name))


# def image_reward(args, dir_list):
#     prompts = PROMPTS[:args.num_prompts]
#     model = RM.load("ImageReward-v1.0")
#     score_list = [0] * len(dir_list)
#     for i, prompt in enumerate(prompts):
#         for j in range(args.num_images_per_prompt):
#             img_list = [os.path.join(dir, '{}-{}.png'.format(i, j)) for dir in dir_list]
#             with torch.no_grad():
#                 ranking, rewards = model.inference_rank(prompt, img_list)
#                 print("\nPreference predictions:\n")
#                 print(f"ranking = {ranking}")
#                 print(f"rewards = {rewards}")
#                 for index in range(len(img_list)):
#                     score = model.score(prompt, img_list[index])
#                     print(f"{img_list[index]}: {score:.2f}")
#                     score_list[index] += score / 1000.
#     print(score_list)


# def hps(args, dir_list):
#     prompts = PROMPTS[:args.num_prompts]
#     score_list = [0] * len(dir_list)
#     for i, prompt in enumerate(prompts):
#         for j in range(args.num_images_per_prompt):
#             img_list = [os.path.join(dir, '{}-{}.png'.format(i, j)) for dir in dir_list]
#             scores = hpsv2.score(img_list, prompt, hps_version="v2.1")
#             for index in range(len(img_list)):
#                 score_list[index] += scores[index]
#     print(score_list)


def main():
    args = parse_args()
    same_seeds(args.seed)
    if args.generate_images:
        generate_images(args)
    else:
        assert os.path.isdir(args.output_dir) and len(os.listdir(args.output_dir)) == args.num_prompts * args.num_images_per_prompt

    # image_reward(args, [args.output_dir])
    # hps(args, [args.output_dir])


if __name__ == '__main__':
    main()
    
    

