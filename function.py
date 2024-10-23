import os
import time
import shutil
import nltk
import torch
import json
import numpy as np
import torch.nn.functional as F

from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline, DDIMScheduler, DiffusionPipeline
from typing import Optional, Union, Tuple, List, Callable, Dict
from einops import rearrange
from torchvision import transforms
from ray import train
from PIL import Image
from sklearn.metrics import f1_score, roc_curve, auc
from visualization import create_palette, show_cross_attention, show_cam_on_image
from attention import AttentionStore, aggregate_all_attention


VOC_label_map = {
    1:'aeroplane',
    2:'bicycle',
    3:'bird',
    4:'boat',
    5:'bottle',
    6:'bus',
    7:'car',
    8:'cat',
    9:'chair',
    10:'cow',
    11:'diningtable',
    12:'dog',
    13:'horse',
    14:'motorbike',
    15:'person',
    16:'pottedplant',
    17:'sheep',
    18:'sofa',
    19:'train',
    20:'tvmonitor'
}

Cityscape_label_map = {
    1: 'road', # flat
    2: 'person', # human
    3: 'building', # construction
    4: 'traffic light', # object
    5: 'vegetation', # nature
    6: 'car', # vehicle
    7: 'bus', # vehicle
    8: 'train', # vehicle
    9: 'motorcycle', # vehicle
    10: 'bicycle', #vehicle
}

Vaihingen_label_map = {
    1: 'building'
}

Kvasir_label_map = {
    1: 'tumor'
}

GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


# encode images for SD and SDXL
def encode_imgs(imgs, vae):
    # imgs: [B, 3, H, W]
    imgs = 2 * imgs - 1
    posterior = vae.encode(imgs).latent_dist.mean
    latents = posterior * 0.18215
    return latents


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True 


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.config.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            x = hidden_states
            context = encoder_hidden_states
            mask = attention_mask
            batch_size = len(x)
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            ## controller foward function saving the attention map in self.step_store
            attn = controller(attn, is_cross, place_in_unet)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


@torch.no_grad()
def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False, height=None, width=None):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        context = torch.cat(context)
        added_cond_kwargs = {}
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context, added_cond_kwargs=added_cond_kwargs)["sample"]

    return latents


## text to image custom pipeline
@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
    noise_sample_num=1,
    height: int = None,
    width: int = None
):
    register_attention_control(model, controller)
    height = 512 if height is None else height
    width = 512 if width is None else width
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    model.unet.to(model.device)
    if hasattr(model, 'text_encoder'):
        text_input_ids = text_input.input_ids
        prompt_embeds = model.text_encoder(text_input_ids.to(model.device), attention_mask=None)
        text_embeddings = prompt_embeds[0]

        uncond_input = model.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_prompt_embeds = model.text_encoder(
            uncond_input.input_ids.to(model.device),
            attention_mask=None,
        )
        uncond_embeddings = negative_prompt_embeds[0]
    else:
        text_input = model.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
        uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    latents = latent.squeeze(1).to(model.device)

    ## Sets the discrete timesteps used for the diffusion chain (to be run before inference).
    model.scheduler.set_timesteps(num_inference_steps)
    latents = diffusion_step(model, controller, latents, context, num_inference_steps, guidance_scale, low_resource, height, width)

    return None, None


def generate_att(t, ldm_stable, input_latent, noise, prompts, controller, pos_positions, device,
                 is_self=True, is_multi_self=False, is_cross_norm=True, weight=[0.3,0.5,0.1,0.1],
                 height=None, width=None, verbose=False, alpha=8, beta=0.4, neg_positions=[], neg_weight=1.0, cls_name='', all_masks=False):

    ## pos: position of the target class word int he prompt
    controller.reset()
    g_cpu = torch.Generator(4307)
    t = int(t)
    latents_noisy = ldm_stable.scheduler.add_noise(input_latent, noise, torch.tensor(t, device=device))
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latents_noisy, num_inference_steps=t, guidance_scale=GUIDANCE_SCALE, generator=g_cpu, low_resource=False, height=height, width=width)
    layers = ("mid", "up", "down")
    cross_attention_maps = aggregate_all_attention(prompts, controller, layers, True, 0)
    self_attention_maps = aggregate_all_attention(prompts, controller, ("up", "mid", "down"), False, 0)
    imgs = []
    ## res: resolution
    for idx,res in enumerate([8, 16, 32, 64]):
        out_att = cross_attention_maps[idx].permute(2,0,1).float()
        if is_cross_norm:
            att_max = torch.amax(out_att,dim=(1,2),keepdim=True)
            att_min = torch.amin(out_att,dim=(1,2),keepdim=True)
            out_att = (out_att-att_min)/(att_max-att_min)
        if is_multi_self:
            self_att = self_attention_maps[idx].view(res*res,res*res).float()
            self_att = self_att/self_att.max()
            out_att = torch.matmul(self_att.unsqueeze(0),out_att.view(-1,res*res,1)).view(-1,res,res)
        if res != 64:
            out_att = F.interpolate(out_att.unsqueeze(0), size=(64,64), mode='bilinear', align_corners=False).squeeze()
        ## 8*8: 0.3, 16*16: 0.5, 32*32: 0.1, 64*64: 0.1
        imgs.append(out_att * weight[idx])

    # aggregated cross attention map
    if not all_masks:
        cross_att_map = torch.stack(imgs).sum(0)[pos_positions[0]].mean(0).view(64*64, 1)
        for pos in pos_positions[1:]:
            cross_att_map += torch.stack(imgs).sum(0)[pos].mean(0).view(64*64, 1)
        if len(pos_positions) > 1:
            cross_att_map /= len(pos_positions)

        if len(neg_positions) > 0:
            cross_att_map_neg = torch.zeros_like(cross_att_map)
            for pos in neg_positions:
                cross_att_map_neg += torch.stack(imgs).sum(0)[pos].mean(0).view(64*64, 1)
            cross_att_map_neg /= len(neg_positions)
            cross_att_map -= neg_weight * cross_att_map_neg

        # refine cross attention map with self attention map
        if is_self and not is_multi_self:
            self_att = self_attention_maps[3].view(64*64,64*64).float()
            self_att = self_att/self_att.max()
            for i in range(1):
                cross_att_map = torch.matmul(self_att, cross_att_map)
        # res here is the highest resulution iterated in previous for loop, 64
        att_map = cross_att_map.view(res,res)
        att_map = F.interpolate(att_map.unsqueeze(0).unsqueeze(0), size=(512,512), mode='bilinear', align_corners=False).squeeze().squeeze()
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())
        att_map = F.sigmoid(alpha * (att_map - beta))
        att_map = (att_map-att_map.min()) / (att_map.max()-att_map.min())
        att_map = [att_map]

        del cross_att_map
        torch.cuda.empty_cache()
    else:
        assert len(imgs) == 1
        cross_att_maps = imgs[0].reshape(-1, 64*64)
        att_map = []
        tokens = ldm_stable.tokenizer.encode(prompts[0])
        self_att = self_attention_maps[3].view(64*64, 64*64).float()
        self_att = self_att / self_att.max()
        for i in range(len(tokens)):
            cross_att_map = cross_att_maps[i]
            cross_att_map = torch.matmul(self_att, cross_att_map)
            att_map_ = cross_att_map.view(64, 64)
            att_map_ = F.interpolate(att_map_.unsqueeze(0).unsqueeze(0),
                                size=(512, 512),
                                mode='bilinear',
                                align_corners=False).squeeze().squeeze()
            att_map_ = (att_map_ - att_map_.min()) / (att_map_.max() - att_map_.min())
            att_map_ = F.sigmoid(alpha * (att_map_- beta))
            att_map_ = (att_map_ - att_map_.min()) / (att_map_.max() - att_map_.min())
            att_map.append(att_map_)

    if verbose:
        # att_map_map = Image.fromarray((att_map[0].cpu().detach().numpy()*255).astype(np.uint8), mode="L")
        palette = create_palette('viridis')
        tokenizer = ldm_stable.tokenizer
        print("8x8 cross att map")
        show_cross_attention(prompts, tokenizer, controller, palette, res=8, from_where=layers, cls_name=cls_name)
        print("16x16 cross att map")
        show_cross_attention(prompts, tokenizer, controller, palette, res=16, from_where=layers, cls_name=cls_name)
        print("32x32 cross att map")
        show_cross_attention(prompts, tokenizer, controller, palette, res=32, from_where=layers, cls_name=cls_name)
        print("64x64 cross att map")
        show_cross_attention(prompts, tokenizer, controller, palette, res=64, from_where=layers, cls_name=cls_name)

    return att_map


def stable_diffusion_inference(img_path, cls_name, device, blip_device, processor, model, ldm_stable, verbose=False, weight=[0.3,0.5,0.1,0.1], t=100, alpha=8, beta=0.4, seed=3407, negative_token=False, all_masks=False):
  ## img_path: path to the target image
  ## cls name: taget class in the prompt
  ## device: device of stable diffusion model
  ## blip device: device of BLIP model
  ## processor: BLIP processor
  ## model: BLIP model
  ## vae: vae of the stable diffusion model
    with torch.no_grad():
        same_seeds(seed)

        input_img = Image.open(img_path).convert("RGB")

        trans = []
        trans.append(transforms.ToTensor())
        trans.append(transforms.CenterCrop(400))
        trans = transforms.Compose(trans)

        img_tensor = (trans(input_img).unsqueeze(0)).to(device)

        rgb_512 = F.interpolate(img_tensor, (512, 512), mode='bilinear', align_corners=False).bfloat16()
        # rgb_512 = F.interpolate(img_tensor, (512, 512), mode='bilinear', align_corners=False).float()

        if hasattr(ldm_stable, 'vae'):
            vae = ldm_stable.vae
        else:
            vae = ldm_stable.vqvae
        input_latent = encode_imgs(rgb_512, vae)
        noise = torch.randn_like(input_latent).to(device)
        raw_image = input_img

        text = f"a photograph of {cls_name}"
        inputs = processor(raw_image, text, return_tensors="pt").to(blip_device) # processor: Blip processor

        # use blip and "++" emphasizing semantic information of target categories
        out = model.generate(**inputs)
        texts = processor.decode(out[0], skip_special_tokens=True)
        texts = text +"++"+ texts[len(text):] # ", highly realistic, artsy, trending, colorful"

        # weight is the weight of different layer's cross attn
        # pos is the position of target class word in the sentence, in "a photograph of plane" (plane)'s position is 4
        # t is the denoising step, usually set between 50 to 150
        prompts = [texts]
        # print("**** blip_prompt: "+texts+"****")
        token_ids = ldm_stable.tokenizer.encode(texts)
        tokens = [ldm_stable.tokenizer.decode(int(_)) for _ in token_ids]
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
                pos_positions.append([i])
        if negative_token:
            for i, (word, tag) in enumerate(tagged_tokens[neg_start_pos:-1]):
                if tag.startswith('N'):
                    if word != cls_name:
                        # print(word)
                        neg_positions.append([i + neg_start_pos])

        # print(pos_positions)
        # print(neg_positions)
        controller = AttentionStore()

        height = 512
        width = 512
        # pos_positions = [[4]]
        # print(pos_positions)
        mask = generate_att(t, ldm_stable, input_latent, noise, prompts, controller, pos_positions, device,
                        is_self=True, is_multi_self=False, is_cross_norm=True, weight=weight, height=height, width=width,
                        verbose=verbose, alpha=alpha, beta=beta, neg_positions=neg_positions, cls_name=cls_name, all_masks=all_masks)
        for i, m in enumerate(mask):
            mask[i] = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(raw_image.size[1],raw_image.size[0]), mode='bilinear', align_corners=False).squeeze().squeeze()
    
        if verbose:
            left = (raw_image.size[0] - 400)/2
            top = (raw_image.size[1] - 400)/2
            right = (raw_image.size[0] + 400)/2
            bottom = (raw_image.size[1] + 400)/2

            # Crop the center of the image
            raw_image = raw_image.crop((left, top, right, bottom))
            cam = show_cam_on_image(raw_image, mask[0])
            print("visual_cam")
            pil_img = Image.fromarray(cam[:,:,::-1])
            # display(pil_img)
            pil_img.save('cam.pdf')
        del img_tensor
        del noise
        del inputs
        torch.cuda.empty_cache()
    return mask


def domain_test(processor, model, ldm_stable, blip_device, device, images_dir, result_dir, label_map, augmented_label_file, dataset_root_length,
        augmented_label=False, thres_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], weight=[0.3,0.5,0.1,0.1], t = 100, alpha=8, beta=0.4, 
        seed=3407, negative_token=False, single_image=False, all_masks=False):

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
        img = Image.open(img_path)
        
        size += 1
        seg_classes = label_data[img_path[dataset_root_length:]]

        for cls_name in seg_classes.keys():
            mask = stable_diffusion_inference(img_path, cls_name, device, blip_device, processor, model, ldm_stable, verbose=verbose, weight=weight, t=t, alpha=alpha, beta=beta, seed=seed, negative_token=negative_token, all_masks=all_masks)
            for mask_idx, m in enumerate(mask):
                with open(os.path.join(result_dir, 'mask', '{}_{}_{}.npy'.format(img_file.split('.')[0], cls_name, mask_idx)), 'wb') as f:
                    np.save(f, m)
                for mask_threshold in thres_list:
                    mask_binary = np.where(m > mask_threshold, 255, 0)
                    mask_binary_img = Image.fromarray(mask_binary.astype(np.uint8))
                    mask_binary_img.save(os.path.join(result_dir, '{}'.format(mask_threshold), '{}_{}_{}.png'.format(img_file.split('.')[0], cls_name, mask_idx)))

            if augmented_label:
                for aug_cls_name in seg_classes[cls_name]:
                    mask = stable_diffusion_inference(img_path, aug_cls_name, device, blip_device, processor, model, ldm_stable, verbose=verbose, weight=weight, t=t, alpha=alpha, beta=beta, seed=seed, negative_token=negative_token, all_masks=all_masks)
                    for mask_idx, m in enumerate(mask):
                        with open(os.path.join(result_dir, 'mask', '{}_{}_{}.npy'.format(img_file.split('.')[0], aug_cls_name, mask_idx)), 'wb') as f:
                            np.save(f, m)
                        for mask_threshold in thres_list:
                            mask_binary = np.where(m > mask_threshold, 255, 0)
                            mask_binary_img = Image.fromarray(mask_binary.astype(np.uint8))
                            mask_binary_img.save(os.path.join(result_dir, '{}'.format(mask_threshold), '{}_{}_{}.png'.format(img_file.split('.')[0], aug_cls_name, mask_idx)))
    # gt_path = os.path.join(images_dir.replace('images', 'segmentations'), img_file)
    # gt = Image.open(gt_path)
    # display(gt)
    ds_name = images_dir.split('/')[-2]
    print(">>>>>>>>>> dataset: {}, size: {}, test time: {:.2f}s".format(ds_name, size, time.time() - start))


def analysis(results_dir_list, segmentations_dir_list, augmented_label_file, dataset_root_length, thres_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    start = time.time()

    iou_res_ = {}
    pixel_acc_res_ = {}
    f1_res_ = {}
    for thres in thres_list:
        iou_res_[thres] = []
        pixel_acc_res_[thres] = []
        f1_res_[thres] = []

    for results_dir, segmentations_dir in zip(results_dir_list, segmentations_dir_list):
        cls_arr_dir = segmentations_dir.replace("segmentations", "class_array")
        images_dir = segmentations_dir.replace("segmentations", "images")
        augmented_label_path = segmentations_dir.replace("segmentations", augmented_label_file)
        print('>>>> result_dir:', results_dir)
        print('>>>> segmentations_dir:', segmentations_dir)
        with open(augmented_label_path, 'r') as f:
            label_data = json.load(f)

        iou_domain = []
        pixel_acc_domain = []
        f1_domain = []

        for thres in thres_list:
            predict_root_dir = os.path.join(results_dir, '{}'.format(thres))
            if not os.path.isdir(predict_root_dir):
                print("{} does not exist".format(predict_root_dir))
                continue
            iou_res = []
            pixel_acc_res = []
            f1_res = []

            for seg_file in os.listdir(segmentations_dir):
                if not(seg_file.endswith('.png') or seg_file.endswith('.tif') or seg_file.endswith('.jpg')):
                    continue
                img_path = os.path.join(images_dir, seg_file)
                seg_classes = label_data[img_path[dataset_root_length:]]

                for cls_name in seg_classes.keys():
                    all_classes = [cls_name] + seg_classes[cls_name]
                    seg_cls_arr = np.load(os.path.join(cls_arr_dir, '{}_{}.npy'.format(seg_file.split('.')[0], cls_name)))
                    iou = -1
                    pixel_acc = -1
                    f1 = -1
                    for cls in all_classes:
                        for mask_idx in range(77):
                            predict_path = os.path.join(predict_root_dir, '{}_{}_{}.png'.format(seg_file.split('.')[0], cls, mask_idx))
                            if os.path.isfile(predict_path):
                                predict_img = Image.open(predict_path)
                                predict_cls_arr = np.asarray(predict_img) / 255
                            else:
                                break

                            if predict_cls_arr.shape != seg_cls_arr.shape:
                                print('>>>invalid prediction', predict_path, seg_cls_arr.shape, predict_cls_arr.shape)
                                continue
                            intersection = np.sum(predict_cls_arr * seg_cls_arr).astype(np.float32)
                            union = np.sum(np.logical_or(predict_cls_arr, seg_cls_arr)).astype(np.float32)
                            correct = np.sum(predict_cls_arr == seg_cls_arr).astype(np.float32)

                            iou_ = intersection / union
                            pixel_acc_ = correct / (seg_cls_arr.shape[0] * seg_cls_arr.shape[1])
                            f1_ = f1_score(seg_cls_arr.flatten(), predict_cls_arr.flatten())

                            if f1_ > f1:
                                f1 = f1_
                                pixel_acc = pixel_acc_
                                iou = iou_

                    iou_res.append(iou)
                    pixel_acc_res.append(pixel_acc)
                    f1_res.append(f1)

            iou_res_[thres] += iou_res
            pixel_acc_res_[thres] += pixel_acc_res
            f1_res_[thres] += f1_res

            f1_mean = np.array(f1_res).mean()
            iou_mean = np.array(iou_res).mean()
            pixel_acc_mean = np.array(pixel_acc_res).mean()
            iou_domain.append(iou_mean)
            pixel_acc_domain.append(pixel_acc_mean)
            f1_domain.append(f1_mean)
            print('>>>> thres: {}, dice: {:.4f}, iou: {:.4f}, pixel_acc: {:.4f}'.format(thres, f1_mean, iou_mean, pixel_acc_mean))

        iou_domain_auc = auc(np.asarray(thres_list), np.asarray(iou_domain))
        pixel_acc_domain_auc = auc(np.asarray(thres_list), np.asarray(pixel_acc_domain))
        f1_domain_auc = auc(np.asarray(thres_list), np.asarray(f1_domain))
        print('>>> dice AUC: {:.4f}, iou AUC: {:.4f}, pixel_acc AUC: {:.4f}'.format(f1_domain_auc, iou_domain_auc, pixel_acc_domain_auc))

    iou_res_summary = []
    pixel_res_summary = []
    f1_res_summary = []
    for thres in thres_list:
        iou_res_summary.append(np.array(iou_res_[thres]).mean())
        pixel_res_summary.append(np.array(pixel_acc_res_[thres]).mean())
        f1_res_summary.append(np.array(f1_res_[thres]).mean())

    iou_res_summary = np.asarray(iou_res_summary)
    iou_auc = auc(np.asarray(thres_list), iou_res_summary)
    iou_optim = iou_res_summary.max()
    iou_auc_over_optim = iou_auc / iou_optim

    pixel_res_summary = np.asarray(pixel_res_summary)
    pixel_auc = auc(np.asarray(thres_list), pixel_res_summary)
    pixel_optim = pixel_res_summary.max()
    pixel_auc_over_optim = pixel_auc / pixel_optim

    f1_res_summary = np.asarray(f1_res_summary)
    f1_auc = auc(np.asarray(thres_list), f1_res_summary)
    f1_optim = f1_res_summary.max()
    f1_auc_over_optim = f1_auc / f1_optim

    print('>>> dice AUC: {:.4f}, dic optimum: {:.4f}, dice AUC/optim: {:.4f}'.format(f1_auc, f1_optim, f1_auc_over_optim))
    print('>>> iou AUC: {:.4f}, iou optimum: {:.4f}, iou AUC/optim: {:.4f}'.format(iou_auc, iou_optim, iou_auc_over_optim))
    print('>>> pixel_acc AUC: {:.4f}, pixel_acc optimum: {:.4f}, pixel_acc AUC/optim: {:.4f}'.format(pixel_auc, pixel_optim, pixel_auc_over_optim))
    print('>>> analysis time: {:.2f}s'.format(time.time() - start))
    return f1_auc, f1_optim, iou_auc, iou_optim, pixel_auc, pixel_optim


def stable_diffusion_func(config, args=None):
    model_key = args.diffusion_model
    device = torch.device('cuda:0')
    if model_key.endswith("LDM"):
        ldm_stable = DiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.bfloat16).to(device)
    else:
        ldm_stable = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.bfloat16).to(device)

    ldm_stable.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler",
                                                        beta_start=0.00085,beta_end=0.012,
                                                        steps_offset=1)
    processor = BlipProcessor.from_pretrained(args.BLIP)
    model = BlipForConditionalGeneration.from_pretrained(args.BLIP).to(device)
    dataset_names = ["VOC2012", "Cityscape", "Vaihingen", "Kvasir-SEG"]
    datasets = [os.path.join(args.dataset_root, ds) for ds in dataset_names] if not args.single_image else [os.path.join(args.dataset_root, "VOC2012")]
    label_maps = [VOC_label_map, Cityscape_label_map, Vaihingen_label_map, Kvasir_label_map] if not args.single_image else [VOC_label_map]
    map_weight_sum = config["map_weight1"] + config["map_weight2"] + config["map_weight3"] + config["map_weight4"]
    map_weight1_ = config["map_weight1"] / map_weight_sum
    map_weight2_ = config["map_weight2"] / map_weight_sum
    map_weight3_ = config["map_weight3"] / map_weight_sum
    map_weight4_ = config["map_weight4"] / map_weight_sum
    weight = [map_weight1_, map_weight2_, map_weight3_, map_weight4_]

    res_dir = "results_0.9_{}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{}".format(int(config["t"]), map_weight1_, map_weight2_, map_weight3_, map_weight4_, config["alpha"], config["beta"], args.negative_token)
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
        # print(">>>")
        # print(images_dir)
        # print(">>>")
        # print(result_dir)
        domain_test(processor, model, ldm_stable, device, device, images_dir, result_dir, label_map, augmented_label_file, dataset_root_length,
            augmented_label=True, thres_list=thres_list, weight=weight, t=int(config["t"]), alpha=config["alpha"], beta=config["beta"], seed=args.seed, 
            negative_token=args.negative_token, single_image=args.single_image, all_masks=args.all_masks)
        
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

    del ldm_stable
    del model
    del processor
    torch.cuda.empty_cache()
    shutil.rmtree(root_dir)

    
