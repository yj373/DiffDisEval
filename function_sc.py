import os
import time
import shutil
import nltk
import torch
import torchvision
import json
import safetensors
import math
import numpy as np
import torch.nn.functional as F

from transformers import BlipProcessor, BlipForConditionalGeneration, EfficientNetModel
from diffusers import DDIMScheduler, DDPMWuerstchenScheduler, StableCascadeDecoderPipeline, StableCascadePriorPipeline
from typing import Optional, Union, Tuple, List, Callable, Dict
from einops import rearrange
from torchvision import transforms
from ray import train
from PIL import Image
from math import ceil
from torch import nn
from sklearn.metrics import f1_score, roc_curve, auc
from visualization import create_palette, show_cross_attention, show_cam_on_image
from attention import AttentionStore, aggregate_all_attention, aggregate_all_attention_sc
from function import VOC_label_map, Cityscape_label_map, Vaihingen_label_map, Kvasir_label_map, analysis, same_seeds

GUIDANCE_SCALE = 7.5
# GUIDANCE_SCLAE = 4.0
MAX_NUM_WORDS = 77

def load_effnet(path):
  checkpoint = {}
  with safetensors.safe_open(path, framework="pt", device="cpu") as f:
    for key in f.keys():
      checkpoint[key] = f.get_tensor(key)
  return checkpoint


# EfficientNet
class EfficientNetEncoder(nn.Module):
    def __init__(self, c_latent=16):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s(weights='DEFAULT').features.eval()
        self.mapper = nn.Sequential(
            nn.Conv2d(1280, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent, affine=False),  # then normalize them to have mean 0 and std 1
        )

    def forward(self, x):
        return self.mapper(self.backbone(x))


# encode images for SC
def encode_imgs_prior(imgs, effnet):
  latents = effnet(imgs)
  return latents


def encode_imgs_decoder(imgs, vqgan):
  latents = vqgan.encode(imgs).latents
  return latents


@torch.no_grad()
def register_attention_control_sc(model, controller, prior=True):
    def ca_forward(self, place_in_unet):

        def forward(x, kv):
            # print("kv shape", kv.shape) # prior: torch.Size([2, 85, 2048])
            # print("x shape", x.shape) # prior: torch.Size([2, 2048, 24, 24])
            kv = self.kv_mapper(kv) # encoder_hidden_states
            norm_x = self.norm(x) # hidden_states

            attn = self.attention
            batch_size, channel, height, width = norm_x.shape
            hidden_states_attn = norm_x.view(batch_size, channel, height * width).transpose(1, 2)
            q = attn.to_q(hidden_states_attn)
            if attn.norm_cross:
              print('norm_cross is True')
              kv = attn.norm_encoder_hidden_states(kv)
            k = attn.to_k(kv)
            v = attn.to_v(kv)

            inner_dim = k.shape[-1]
            head_dim = inner_dim // attn.heads

            q = q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            scale_factor = 1 / math.sqrt(q.size(-1))
            attn_cross = q @ k.transpose(-2, -1) * scale_factor
            attn_cross = attn_cross[0, :, :, :].squeeze()
            attn_cross = torch.softmax(attn_cross, dim=-1)

            k = attn.to_k(hidden_states_attn)
            v = attn.to_v(hidden_states_attn)

            inner_dim = k.shape[-1]
            head_dim = inner_dim // attn.heads
            k = k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            attn_self = q @ k.transpose(-2, -1) * scale_factor
            attn_self = attn_self[0, :, :, :].squeeze()
            attn_self = torch.softmax(attn_self, dim=-1)

            attn_self = controller(attn_self, False, place_in_unet)
            attn_cross = controller(attn_cross, True, place_in_unet)
            if self.self_attn:
                batch_size, channel, _, _ = x.shape
                kv = torch.cat([norm_x.view(batch_size, channel, -1).transpose(1, 2), kv], dim=1)
            out = x + attn(norm_x, encoder_hidden_states=kv)
            return out

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'SDCascadeAttnBlock':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    if prior:
      sub_nets = model.prior.named_children()
    else:
      sub_nets = model.decoder.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            num = register_recr(net[1], 0, "down")
            # print('down num', num)
            cross_att_count += num
        elif "up" in net[0]:
            num = register_recr(net[1], 0, "up")
            # print('up num', num)
            cross_att_count += num
        elif "mid" in net[0]:
            num = register_recr(net[1], 0, "mid")
            # print('mid num', num)
            cross_att_count += num

    controller.num_att_layers = cross_att_count


def encode_prompt(
    model,
    prompt,
    batch_size
):
    text_inputs = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    device = model.device
    text_input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask
    text_encoder_output = model.text_encoder(
        text_input_ids.to(device), attention_mask=attention_mask.to(device), output_hidden_states=True
    )
    prompt_embeds = text_encoder_output.hidden_states[-1]
    prompt_embeds_pooled = text_encoder_output.text_embeds.unsqueeze(1)

    prompt_embeds = prompt_embeds.to(dtype=model.text_encoder.dtype, device=device)
    prompt_embeds_pooled = prompt_embeds_pooled.to(dtype=model.text_encoder.dtype, device=device)
    prompt_embeds = prompt_embeds.repeat_interleave(1, dim=0)
    prompt_embeds_pooled = prompt_embeds_pooled.repeat_interleave(1, dim=0)

    # do classifier-free guidance
    uncond_tokens = [""] * batch_size
    uncond_input = model.tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    negative_prompt_embeds_text_encoder_output = model.text_encoder(
        uncond_input.input_ids.to(device),
        attention_mask=uncond_input.attention_mask.to(device),
        output_hidden_states=True,
    )
    negative_prompt_embeds = negative_prompt_embeds_text_encoder_output.hidden_states[-1]
    negative_prompt_embeds_pooled = negative_prompt_embeds_text_encoder_output.text_embeds.unsqueeze(1)

    # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
    seq_len = negative_prompt_embeds.shape[1]
    negative_prompt_embeds = negative_prompt_embeds.to(dtype=model.text_encoder.dtype, device=device)
    negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
    negative_prompt_embeds = negative_prompt_embeds.view(batch_size, seq_len, -1)

    seq_len = negative_prompt_embeds_pooled.shape[1]
    negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.to(
        dtype=model.text_encoder.dtype, device=device
    )
    negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.repeat(1, 1, 1)
    negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.view(
        batch_size, seq_len, -1
    )
    return prompt_embeds, prompt_embeds_pooled, negative_prompt_embeds, negative_prompt_embeds_pooled


def init_latent_prior(latent, model, height, width, generator, batch_size):
    # print(height, model.config.resolution_multiple)
    latents_shape = (
        batch_size,
        model.prior.config.in_channels,
        ceil(height / model.config.resolution_multiple),
        ceil(width / model.config.resolution_multiple),
    )
    if latent is None:
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=model.device,
            dtype=model.dtype
        )
    else:
        # if latent.shape != latents_shape:
        #   raise ValueError(f"Unexpected latents shape, got {latent.shape}, expected {latents_shape}")
        latents = latent.to(model.device)

    latents = latents * model.scheduler.init_noise_sigma
    return latents


@torch.no_grad()
def diffusion_step_prior(
    prior,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    height: int = 512,
    width: int = 512,
):
    register_attention_control_sc(prior, controller, prior=True)
    batch_size = len(prompt)
    device = prior.prior.device
    dtype=next(prior.prior.parameters()).dtype

    (
        prompt_embeds,
        prompt_embeds_pooled,
        negative_prompt_embeds,
        negative_prompt_embeds_pooled
    ) = encode_prompt(
        model=prior,
        prompt=prompt,
        batch_size=batch_size
    )
    # print(prompt_embeds.shape) # torch.Size([1, 77, 1280])
    # print(prompt_embeds_pooled.shape) # torch.Size([1, 1, 1280])
    # print(negative_prompt_embeds.shape)
    # print(negative_prompt_embeds_pooled.shape)
    # print('>>>')

    image_embeds_pooled = torch.zeros(
        1,
        1,
        prior.prior.config.clip_image_in_channels,
        device=device,
        dtype=dtype
    )
    uncond_image_embeds_pooled = torch.zeros(
        1,
        1,
        prior.prior.config.clip_image_in_channels,
        device=device,
        dtype=dtype
    )
    image_embeds = torch.cat([image_embeds_pooled, uncond_image_embeds_pooled], dim=0)

    text_encoder_hidden_states = (
        torch.cat([prompt_embeds, negative_prompt_embeds]) if negative_prompt_embeds is not None else prompt_embeds
    )
    text_encoder_pooled = (
        torch.cat([prompt_embeds_pooled, negative_prompt_embeds_pooled])
        if negative_prompt_embeds is not None
        else prompt_embeds_pooled
    )

    prior.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = prior.scheduler.timesteps

    latents = init_latent_prior(latent, prior, height, width, generator, batch_size)
    if isinstance(prior.scheduler, DDPMWuerstchenScheduler):
        timesteps = timesteps[:-1]

    if hasattr(prior.scheduler, "betas"):
        alphas = 1.0 - prior.scheduler.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
    else:
        alphas_cumprod = []

    for i, t in enumerate(timesteps):
        if not isinstance(prior.scheduler, DDPMWuerstchenScheduler):
            if len(alphas_cumprod) > 0:
                timestep_ratio = prior.get_timestep_ratio_conditioning(t.long().cpu(), alphas_cumprod)
                timestep_ratio = timestep_ratio.expand(latents.size(0)).to(dtype).to(device)
            else:
                timestep_ratio = t.float().div(prior.scheduler.timesteps[-1]).expand(latents.size(0)).to(dtype)
        else:
            timestep_ratio = t.expand(latents.size(0)).to(dtype)

        print("text_encoder_pooled.shape", text_encoder_pooled.shape) # torch.Size([2, 1, 1280])
        print("text_encoder_hidden_states.shape", text_encoder_hidden_states.shape) # torch.Size([2, 77, 1280])
        print("image_embeds.shape", image_embeds.shape) # torch.Size([2, 1, 768])
        print("latents.shape", latents.shape) # torch.Size([1, 16, 24, 24])
        print("timestep_ratio.shape", timestep_ratio.shape) # torch.Size([1])
        predicted_image_embedding = prior.prior(
            sample=torch.cat([latents] * 2),
            timestep_ratio= torch.cat([timestep_ratio] * 2),
            clip_text_pooled=text_encoder_pooled,
            clip_text=text_encoder_hidden_states, ######################
            clip_img=image_embeds,
            return_dict=False,
        )[0]
        break


def init_latent_decoder(latent, model, image_embeddings, generator, batch_size):
    _, channels, height, width = image_embeddings.shape
    latents_shape = (
        batch_size,
        4,
        int(height * model.config.latent_dim_scale),
        int(width * model.config.latent_dim_scale),
    )
    if latent is None:
        latents = torch.randn(latents_shape, generator=generator, device=model.device, dtype=model.dtype)
    else:
        # print(latents_shape, latent.shape)
        # if latents.shape != latents_shape:
        #   raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
        latents = latent.to(model.device)
    return latents


@torch.no_grad()
def diffusion_step_decoder(
    decoder,
    prompt: List[str],
    image_embeddings, # Image embeddings extracted from an image or generated by a Prior model
    controller,
    num_inference_steps: int = 50,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    height: int = 512,
    width: int = 512,
):

    register_attention_control_sc(decoder, controller, prior=False)
    batch_size = len(prompt)
    device = decoder.decoder.device
    dtype=next(decoder.decoder.parameters()).dtype

    (
        _,
        prompt_embeds_pooled,
        _,
        negative_prompt_embeds_pooled
    ) = encode_prompt(
        model=decoder,
        prompt=prompt,
        batch_size=batch_size
    )

    if isinstance(image_embeddings, list):
        image_embeddings = torch.cat(image_embeddings, dim=0)

    prompt_embeds_pooled = torch.cat([prompt_embeds_pooled, negative_prompt_embeds_pooled])
    effnet = torch.cat([image_embeddings, torch.zeros_like(image_embeddings)])

    decoder.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = decoder.scheduler.timesteps

    latents = init_latent_decoder(latent, decoder, image_embeddings, generator, batch_size)
    decoder._num_timesteps = len(timesteps[:-1])

    for i, t in enumerate(timesteps[:-1]):
        timestep_ratio = t.expand(latents.size(0)).to(decoder.dtype)

        predicted_latents = decoder.decoder(
            sample=torch.cat([latents] * 2),
            timestep_ratio=torch.cat([timestep_ratio] * 2),
            clip_text_pooled=prompt_embeds_pooled,
            effnet=effnet,
            return_dict=False,
        )[0]
        break


def generate_att_sc(
    t,
    prior,
    decoder,
    sc_latent,
    noise_prior,
    vqgan_latent,
    noise_decoder,
    prompts,
    controller_prior,
    controller_decoder,
    pos_positions,
    device,
    height_sc: int = 768,
    width_sc: int = 768,
    height_vqgan: int = 1024,
    width_vqgan: int = 1024,
    verbose: bool = False,
    neg_positions = [],
    neg_weight: float = 1.0,
    alpha_prior: float = 8.0,
    beta_prior: float = 0.4,
    alpha_decoder: float = 8.0,
    beta_decoder: float = 0.4,
    decoder_weight = [0.5, 0.5],
    cls_name: str = '',
    all_masks: bool = False, # generate a mask for all tokens
):
    controller_prior.reset()
    controller_decoder.reset()
    g_cpu = torch.Generator(4307)

    latents_prior = prior.scheduler.add_noise(sc_latent, noise_prior, torch.tensor(t, device=device))
    diffusion_step_prior(
        prior,
        prompts,
        controller_prior,
        num_inference_steps=t,
        generator=g_cpu,
        latent=latents_prior,
        height=height_sc,
        width=width_sc,
    )

    layers = ("mid", "up", "down")
    imgs = [] # a list of tensors [24, 24, 85]
    cross_attention_maps_prior = aggregate_all_attention_sc(prompts, controller_prior, layers, True, 0) # here 0 means the first peompt
    for idx in range(len(cross_attention_maps_prior)):
        out_att = cross_attention_maps_prior[idx].permute(2, 0, 1).float()
        att_max = torch.amax(out_att, dim=(1,2), keepdim=True)
        att_min = torch.amin(out_att, dim=(1,2), keepdim=True)
        out_att = (out_att - att_min) / (att_max - att_min)
        imgs.append(out_att)
    self_attention_maps_prior = aggregate_all_attention_sc(prompts, controller_prior, ("up", "mid", "down"), False, 0)

    if verbose:
        # att_map_img_prior = Image.fromarray((att_map_prior.cpu().detach().numpy()*255).astype(np.uint8), mode="L")
        palette = create_palette('viridis')
        # palette_image = att_map_img_prior.convert("P")
        # palette_image.putpalette(palette)
        # display(palette_image)
        print("prior: 24x24 cross att map")
        show_cross_attention(prompts, prior.tokenizer, controller_prior, palette, res=24, from_where=layers, cls_name=cls_name)
    if not all_masks:
        cross_att_map = torch.stack(imgs).sum(0)[pos_positions[0]].mean(0).view(24*24, 1)
        for pos in pos_positions[1:]:
            cross_att_map += torch.stack(imgs).sum(0)[pos].mean(0).view(24*24, 1)
        if len(pos_positions) > 1:
            cross_att_map /= len(pos_positions)

        if len(neg_positions) > 0:
            cross_att_map_neg = torch.zeros_like(cross_att_map)
            for pos in neg_positions:
                cross_att_map_neg += torch.stack(imgs).sum(0)[pos].mean(0).view(24*24, 1)
            cross_att_map_neg /= len(neg_positions)
            cross_att_map -= neg_weight * cross_att_map_neg

        self_att = self_attention_maps_prior[-1].view(24*24, 24*24).float()
        self_att = self_att / self_att.max()
        cross_att_map = torch.matmul(self_att, cross_att_map)

        att_map_prior = cross_att_map.view(24, 24)
        att_map_prior = F.interpolate(att_map_prior.unsqueeze(0).unsqueeze(0),
                                size=(height_vqgan, width_vqgan),
                                mode='bilinear',
                                align_corners=False).squeeze().squeeze()
        att_map_prior = (att_map_prior - att_map_prior.min()) / (att_map_prior.max() - att_map_prior.min())
        att_map_prior = F.sigmoid(alpha_prior * (att_map_prior - beta_prior))
        att_map_prior = (att_map_prior - att_map_prior.min()) / (att_map_prior.max() - att_map_prior.min())
        att_map_prior = [att_map_prior]
        del cross_att_map
        torch.cuda.empty_cache()
    else:
        assert len(imgs) == 1
        cross_att_maps = imgs[0]
        att_map_prior = []
        tokens = prior.tokenizer.encode(prompts[0])
        print("num tokens", len(tokens))
        for i in range(len(tokens)):
            cross_att_map = cross_att_maps[i].mean(0).view(24*24, 1)
            self_att = self_attention_maps_prior[-1].view(24*24, 24*24).float()
            self_att = self_att / self_att.max()
            cross_att_map = torch.matmul(self_att, cross_att_map)
            att_map_prior_ = cross_att_map.view(24, 24)
            att_map_prior_ = F.interpolate(att_map_prior_.unsqueeze(0).unsqueeze(0),
                                size=(height_vqgan, width_vqgan),
                                mode='bilinear',
                                align_corners=False).squeeze().squeeze()
            att_map_prior_ = (att_map_prior_ - att_map_prior_.min()) / (att_map_prior_.max() - att_map_prior_.min())
            att_map_prior_ = F.sigmoid(alpha_prior * (att_map_prior_ - beta_prior))
            att_map_prior_ = (att_map_prior_ - att_map_prior_.min()) / (att_map_prior_.max() - att_map_prior_.min())
            att_map_prior.append(att_map_prior_)

    att_map = att_map_prior
    return att_map


def stable_cascade_inference(
    img_path,
    cls_name,
    device,
    blip_device,
    processor,
    model,
    prior,
    decoder,
    effnet,
    t: int = 100,
    verbose: bool = False,
    seed: int = 3407,
    negative_token: bool = False,
    neg_weight=1.0,
    alpha_prior=8.0,
    beta_prior=0.55,
    all_masks=False,
):
    with torch.no_grad():
        same_seeds(seed)

        input_img = Image.open(img_path).convert("RGB")

        trans = []
        trans.append(transforms.ToTensor())
        trans = transforms.Compose(trans)

        img_tensor = (trans(input_img).unsqueeze(0)).to(device)

        height_sc = 768 # height of the image input to the semantic compressor
        width_sc = 768 # width of the image input to the semantic compressor
        height_vqgan = 1024
        width_vqgan = 1024

        rgb_sc = F.interpolate(img_tensor, (height_sc, width_sc), mode='bicubic', align_corners=False).bfloat16()
        rgb_vqgan = F.interpolate(img_tensor, (height_vqgan, width_vqgan), mode='bicubic', align_corners=False).bfloat16()

        sc_latent = encode_imgs_prior(rgb_sc, effnet)
        noise_prior = torch.randn_like(sc_latent).to(device)
        raw_image = input_img

        # vqgan_latent = encode_imgs_decoder(rgb_vqgan, decoder.vqgan)
        # noise_decoder = torch.randn_like(vqgan_latent).to(device)

        text = f"a photograph of {cls_name}"
        inputs = processor(raw_image, text, return_tensors="pt").to(blip_device) # processor: Blip processor

        # use blip and "++" emphasizing semantic information of target categories
        out = model.generate(**inputs)
        texts = processor.decode(out[0], skip_special_tokens=True)
        texts = text +"++"+ texts[len(text):] # ", highly realistic, artsy, trending, colorful"

        prompts = [texts]
        # print("**** blip_prompt: "+texts+"****")
        tokenizer = prior.tokenizer
        token_ids = tokenizer.encode(texts)
        tokens = [tokenizer.decode(int(_)) for _ in token_ids]
        tagged_tokens = nltk.tag.pos_tag(tokens)
        pos_positions = []   # pos of targer class word
        neg_positions = []
        pos_start = False
        neg_start_pos = -1
        for i, (word, tag) in enumerate(tagged_tokens):
            if word == 'of':
                pos_start = True
                continue
            if word == '++':
                neg_start_pos = i + 1
                break
            if pos_start:
                # pos_positions.append([i + 2])
                pos_positions.append([i])
        # pos_positions = [[7]]
        print('positive tokens:', pos_positions, 'cls_name: ', cls_name)
        if negative_token:
            for i, (word, tag) in enumerate(tagged_tokens[neg_start_pos:-1]):
                if tag.startswith('N'):
                    if word != cls_name:
                        # print(word)
                        neg_positions.append([i + neg_start_pos])
      

        controller_prior = AttentionStore()
        controller_decoder = AttentionStore()

        mask = generate_att_sc(
            t,
            prior,
            decoder,
            sc_latent,
            noise_prior,
            None,
            None,
            prompts,
            controller_prior,
            controller_decoder,
            pos_positions,
            device,
            height_sc=height_sc,
            width_sc=width_sc,
            height_vqgan=height_vqgan,
            width_vqgan=width_vqgan,
            verbose=verbose,
            neg_positions = neg_positions,
            neg_weight=neg_weight,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            alpha_decoder=8.0,
            beta_decoder=0.4,
            decoder_weight=[0.5, 0.5],
            cls_name=cls_name,
            all_masks=all_masks,
        )
        
        for i, m in enumerate(mask):
            mask[i] = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(raw_image.size[1],raw_image.size[0]), mode='bilinear', align_corners=False).squeeze().squeeze()
            
        if verbose:
            cam = show_cam_on_image(raw_image, mask[0])
            print("visual_cam of the first mask")
            pil_img = Image.fromarray(cam[:,:,::-1])
            pil_img.save('cam_{}.pdf'.format(cls_name))
        del img_tensor
        torch.cuda.empty_cache()

    return mask


def domain_test(
    processor,
    model,
    prior,
    decoder,
    effnet,
    blip_device,
    device,
    images_dir,
    result_dir,
    label_map,
    augmented_label_file,
    dataset_root_length,
    augmented_label: bool = False,
    thres_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    t: int = 100,
    seed: int = 3407,
    neg_weight=1.0,
    alpha_prior=8.0,
    beta_prior=0.55,
    single_image=False,
    all_masks=False,
):
    start = time.time()

    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    os.mkdir(os.path.join(result_dir, 'mask'))
    for thres in thres_list:
        os.mkdir(os.path.join(result_dir, '{}'.format(thres)))
    cls_arr_dir = images_dir.replace("images", "class_array")
    augmented_label_path = images_dir.replace("images", augmented_label_file)

    with open(augmented_label_path, 'r') as f:
        label_data = json.load(f)

    size = 0
    verbose = single_image
    print('>>> seed: {}'.format(seed))
    for img_file in os.listdir(images_dir):
        if not (img_file.endswith('.png') or img_file.endswith('.tif') or img_file.endswith('.jpg')):
            continue
        if single_image and not (img_file == '22.png'):
            continue
            print(img_file)
        img_path = os.path.join(images_dir, img_file)
        size += 1

        seg_classes = label_data[img_path[dataset_root_length:]]

        for cls_name in seg_classes.keys():
            mask = stable_cascade_inference(
                img_path,
                cls_name,
                device,
                blip_device,
                processor,
                model,
                prior,
                decoder,
                effnet,
                verbose=verbose,
                seed=seed,
                negative_token=True,
                neg_weight=neg_weight,
                alpha_prior=alpha_prior,
                beta_prior=beta_prior,
                all_masks=all_masks,
            )
            for mask_idx, m in enumertate(mask):
                with open(os.path.join(result_dir, 'mask', '{}_{}_{}.npy'.format(img_file.split('.')[0], cls_name, mask_idx)), 'wb') as f:
                    np.save(f, m)
                for mask_threshold in thres_list:
                    mask_binary = np.where(m > mask_threshold, 255, 0)
                    mask_binary_img = Image.fromarray(mask_binary.astype(np.uint8))
                    mask_binary_img.save(os.path.join(result_dir, '{}'.format(mask_threshold), '{}_{}_{}.png'.format(img_file.split('.')[0], cls_name, mask_idx)))

            if augmented_label:
                for aug_cls_name in seg_classes[cls_name]:
                    mask = stable_cascade_inference(
                        img_path,
                        aug_cls_name,
                        device,
                        blip_device,
                        processor,
                        model,
                        prior,
                        decoder,
                        effnet,
                        verbose=verbose,
                        seed=seed,
                        negative_token=True,
                        neg_weight=neg_weight,
                        alpha_prior=alpha_prior,
                        beta_prior=beta_prior,
                        all_masks=all_masks,
                    )
                    for mask_idx, m in enumertate(mask):
                        with open(os.path.join(result_dir, 'mask', '{}_{}_{}.npy'.format(img_file.split('.')[0], aug_cls_name, mask_idx)), 'wb') as f:
                            np.save(f, m)
                        for mask_threshold in thres_list:
                            mask_binary = np.where(m > mask_threshold, 255, 0)
                            mask_binary_img = Image.fromarray(mask_binary.astype(np.uint8))
                            mask_binary_img.save(os.path.join(result_dir, '{}'.format(mask_threshold), '{}_{}_{}.png'.format(img_file.split('.')[0], aug_cls_name, mask_idx)))

    ds_name = images_dir.split('/')[-2]
    print(">>>>>>>>>> dataset: {}, size: {}, test time: {:.2f}s".format(ds_name, size, time.time() - start))


def stable_cascade_func(config, args=None):
    model_key = args.diffusion_model
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    prior_id = os.path.join(model_key, "stable-cascade-prior")
    decoder_id = os.path.join(model_key, "stable-cascade")
    prior = StableCascadePriorPipeline.from_pretrained(prior_id, variant="bf16", torch_dtype=torch.bfloat16).to(device)
    prior.scheduler = DDIMScheduler.from_pretrained(prior_id, subfolder="scheduler",
                                                     beta_start=0.00085,beta_end=0.012,
                                                     steps_offset=1)

    processor = BlipProcessor.from_pretrained(args.BLIP)
    model = BlipForConditionalGeneration.from_pretrained(args.BLIP).to(device) 
    effnet = EfficientNetEncoder()
    effnet_ckpt = load_effnet(os.path.join(decoder_id, "effnet_encoder.safetensors"))
    effnet.load_state_dict(effnet_ckpt if 'state_dict' not in effnet_ckpt else effnet_ckpt['state_dict'])
    effnet = effnet.bfloat16().to(device)

    dataset_names = ["VOC2012", "Cityscape", "Vaihingen", "Kvasir-SEG"]
    datasets = [os.path.join(args.dataset_root, ds) for ds in dataset_names] if not args.single_image else [os.path.join(args.dataset_root, "VOC2012")]
    label_maps = [VOC_label_map, Cityscape_label_map, Vaihingen_label_map, Kvasir_label_map] if not args.single_image else [VOC_label_map]

    res_dir = "results_0.9_{}_{:.2f}_{:.2f}_{:.2f}".format(int(config["t"]), config["neg_weight"], config["alpha_prior"], config["beta_prior"])
    root_dir = os.path.join(args.output_dir, res_dir)
    thres_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if os.path.isdir(root_dir):
        shutil.rmtree(root_dir)
    os.mkdir(root_dir)
    augmented_label_file = 'aug_label_blip_bert_0.9.json'  
    dataset_root_length = len(args.dataset_root)

    for ds, label_map in zip(datasets, label_maps):
        images_dir = os.path.join(ds, "images")
        result_dir = os.path.join(root_dir, ds.split('/')[-1])

        domain_test(
            processor,
            model,
            prior,
            None,
            effnet,
            device,
            device,
            images_dir,
            result_dir,
            label_map,
            augmented_label_file,
            dataset_root_length,
            augmented_label=True,
            thres_list= thres_list,
            t=int(config["t"]),
            seed=args.seed,
            neg_weight=config["neg_weight"],
            alpha_prior=config["alpha_prior"],
            beta_prior=config["beta_prior"],
            single_image=args.single_image,
            all_masks=args.all_masks,
        )

    if not args.single_image:
        results_dir_list = [os.path.join(root_dir, ds) for ds in dataset_names]
        segmentations_dir_list = [os.path.join(ds, "segmentations") for ds in datasets]
        f1_auc, f1_optim, iou_auc, iou_optim, pixel_auc, pixel_optim = analysis(results_dir_list, segmentations_dir_list, augmented_label_file, dataset_root_length, thres_list=thres_list)

        train.report({
            "f1_auc": f1_auc,
            "f1_optim": f1_optim,
            "iou_auc": iou_auc,
            "iou_optim": iou_optim,
            "pixel_auc": pixel_auc,
            "pixel_optim": pixel_optim
        })           

    del prior
    del model
    del processor
    del effnet
    torch.cuda.empty_cache()
    shutil.rmtree(root_dir)

                                      