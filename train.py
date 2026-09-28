"""
Stage 2: EvFRA denoiser.

Trains the event ControlNet and the anchor latent tokenizer on top of a frozen
SD2 UNet. The model denoises the latent residual z_target - z_anchor under the
EDM parameterization; the diffusion start point is either Gaussian noise or
the ER-VAE event prior (`--use_event_prior`) plus noise.

--task vfp: one branch, anchor t -> target t+1 (Eq. 2).
--task vfi: two branches sharing the model, t1 -> t with the events of (t1, t]
            and t2 -> t with the events of (t, t2] played backward. Both residuals
            are supervised, and the pixel losses use the distance-weighted blend
            of the two predictions (Eq. 3, 4).
"""
import argparse
import logging
import math
import os
import shutil

import numpy as np
import pyiqa
import torch
import torch.nn.functional as F
import transformers
import diffusers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from evfra.data import (CROP_HEIGHT, CROP_WIDTH, UPSAMPLE_SCALE, BSERGBInterpTrainDataset, BSERGBTrainDataset,
                        list_frames)
from evfra.events import DATASETS, build_event_stack, er_vae_input
from evfra.models import ControlNetSD2Model, LatentTokenizer, load_er_vae, load_latent_tokenizer
from evfra.pipeline import CLIP_MODEL, EvFRAPipeline, edm_denoise
from evfra.utils import calculate_lpips, calculate_psnr, calculate_ssim, resize_with_antialiasing

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="EvFRA stage 2: denoiser training")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--train_data_path", type=str, required=True, help="BS-ERGB train split")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--er_vae_path", type=str, default=None, help="Stage-1 ER-VAE checkpoint")
    parser.add_argument("--use_event_prior", action="store_true",
                        help="Start the diffusion from the ER-VAE event prior instead of Gaussian noise")
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None,
                        help="Warm start (default: ControlNet initialized from the SD2 UNet encoder)")
    parser.add_argument("--latent_tokenizer_path", type=str, default=None, help="Warm start for the latent tokenizer")
    parser.add_argument("--task", type=str, default="vfp", choices=["vfp", "vfi"])
    parser.add_argument("--skip_frame", type=int, default=1, help="VFP: frame gap between anchor and target")
    parser.add_argument("--max_interval", type=int, default=2,
                        help="VFI: anchors are 2..max_interval frames apart, target in between")

    parser.add_argument("--lambda_lpips", type=float, default=1.0)
    parser.add_argument("--lambda_pixel_l1", type=float, default=1.0)
    parser.add_argument("--lambda_event_l1", type=float, default=0.0, help="Event-activity weighted pixel L1")
    parser.add_argument("--sigma_min", type=float, default=0.05)
    parser.add_argument("--sigma_max", type=float, default=3.0)
    parser.add_argument("--loss_weight_max", type=float, default=500.0, help="Clamp for the EDM loss weight")
    parser.add_argument("--conditioning_dropout_prob", type=float, default=0.1)

    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_train_steps", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Sequence folder used for validation, e.g. bs_ergb/train/horse_04")
    parser.add_argument("--val_frame_idx", type=int, default=23, help="Validation target frame index")
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help='Checkpoint path or "latest"')
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, default="EvFRA")
    args = parser.parse_args()

    if args.use_event_prior and not args.er_vae_path:
        raise ValueError("--use_event_prior requires --er_vae_path")
    return args


def rand_log_normal(shape, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
    """Log-normal samples (from k-diffusion)."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def vae_encode(vae, images):
    return vae.encode(images).latent_dist.mode() * vae.config.scaling_factor


@torch.no_grad()
def encode_event_prior(er_vae, event_values):
    """ER-VAE latent residual estimate from the most accumulated event stack."""
    event_rgb = event_values[:, 0:1].repeat(1, 3, 1, 1).to(device=er_vae.device, dtype=er_vae.dtype)
    return er_vae.encode(event_rgb).latent_dist.mean * er_vae.config.scaling_factor


def load_validation_sample(val_data_path, target_idx, task, skip_frame=1):
    """
    Center 512x320 crop (after 2x upsampling) of one validation target.

    Returns (target, branches) with one (anchor, event_stack, er_vae_input, weight)
    per branch: t - skip_frame -> t for VFP, t - 1 -> t and t + 1 -> t for VFI.
    """
    cfg = DATASETS["bs_ergb"]
    image_folder = os.path.join(val_data_path, cfg["image_dir"])
    event_folder = os.path.join(val_data_path, cfg["event_dir"])
    frames = list_frames(image_folder)

    def load(idx):
        img = Image.open(os.path.join(image_folder, frames[idx])).convert("RGB")
        return img.resize((img.width * UPSAMPLE_SCALE, img.height * UPSAMPLE_SCALE), Image.BICUBIC), img.size

    target, (w, h) = load(target_idx)
    x = (target.width - CROP_WIDTH) // 2
    y = (target.height - CROP_HEIGHT) // 2
    box = (x, y, x + CROP_WIDTH, y + CROP_HEIGHT)

    if task == "vfp":
        spans = [(target_idx - skip_frame, target_idx, target_idx - skip_frame, False, 1.0)]
    else:
        spans = [(target_idx - 1, target_idx, target_idx - 1, False, 0.5),
                 (target_idx, target_idx + 1, target_idx + 1, True, 0.5)]

    branches = []
    for start, end, anchor_idx, reverse, weight in spans:
        stack, is_empty = build_event_stack(event_folder, start, end, w, h, UPSAMPLE_SCALE, cfg, reverse=reverse)
        stack = stack[:, y:y + CROP_HEIGHT, x:x + CROP_WIDTH].contiguous()
        branches.append((load(anchor_idx)[0].crop(box), stack, er_vae_input(stack, is_empty), weight))
    return target.crop(box), branches


def blend_images(images, weights):
    """Weighted average of PIL images (Eq. 4 for interpolation)."""
    blended = sum(w * np.array(img, dtype=np.float32) for img, w in zip(images, weights)) / sum(weights)
    return Image.fromarray(np.clip(np.round(blended), 0, 255).astype(np.uint8))


def prune_checkpoints(output_dir, limit):
    checkpoints = sorted(
        (d for d in os.listdir(output_dir) if d.startswith("checkpoint-") and d != "checkpoint-best"),
        key=lambda d: int(d.split("-")[1]),
    )
    for d in checkpoints[:max(0, len(checkpoints) - limit + 1)]:
        shutil.rmtree(os.path.join(output_dir, d))


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir,
                                            logging_dir=os.path.join(args.output_dir, "logs")),
    )
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(args.seed)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(accelerator.mixed_precision, torch.float32)

    # Frozen modules
    feature_extractor = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", variant="fp16")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                                low_cpu_mem_usage=True, variant="fp16")
    for module in (image_encoder, vae, unet):
        module.requires_grad_(False)
        module.to(accelerator.device, dtype=weight_dtype)

    er_vae = None
    if args.er_vae_path:
        er_vae = load_er_vae(args.pretrained_model_name_or_path, args.er_vae_path, variant="fp16")
        er_vae.requires_grad_(False)
        er_vae.to(accelerator.device, dtype=torch.float32 if args.use_event_prior else weight_dtype).eval()

    # Trainable modules
    if args.controlnet_model_name_or_path:
        controlnet = ControlNetSD2Model.from_pretrained(args.controlnet_model_name_or_path)
    else:
        controlnet = ControlNetSD2Model.from_unet(unet)
    latent_tokenizer = (load_latent_tokenizer(args.latent_tokenizer_path)
                        if args.latent_tokenizer_path else LatentTokenizer())
    controlnet.train()
    latent_tokenizer.train()

    ema_controlnet = None
    if args.use_ema:
        ema_controlnet = EMAModel(controlnet.parameters(), model_cls=ControlNetSD2Model, model_config=controlnet.config)

    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    def save_model_hook(models, weights, output_dir):
        if ema_controlnet is not None:
            ema_controlnet.save_pretrained(os.path.join(output_dir, "controlnet_ema"))
        for model in models:
            if isinstance(model, ControlNetSD2Model):
                model.save_pretrained(os.path.join(output_dir, "controlnet"))
            elif isinstance(model, LatentTokenizer):
                torch.save(model.state_dict(), os.path.join(output_dir, "latent_tokenizer.pth"))
            weights.pop()

    def load_model_hook(models, input_dir):
        if ema_controlnet is not None:
            ema = EMAModel.from_pretrained(os.path.join(input_dir, "controlnet_ema"), ControlNetSD2Model)
            ema_controlnet.load_state_dict(ema.state_dict())
            ema_controlnet.to(accelerator.device)
        while models:
            model = models.pop()
            if isinstance(model, ControlNetSD2Model):
                loaded = ControlNetSD2Model.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**loaded.config)
                model.load_state_dict(loaded.state_dict())
            elif isinstance(model, LatentTokenizer):
                model.load_state_dict(torch.load(os.path.join(input_dir, "latent_tokenizer.pth"), map_location="cpu"))

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    trainable_params = list(controlnet.parameters()) + list(latent_tokenizer.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    if args.task == "vfp":
        train_dataset = BSERGBTrainDataset(args.train_data_path, skip_frame=args.skip_frame)
    else:
        train_dataset = BSERGBInterpTrainDataset(args.train_data_path, max_interval=args.max_interval)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.per_gpu_batch_size, shuffle=True, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0, pin_memory=True,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    controlnet, latent_tokenizer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, latent_tokenizer, optimizer, train_dataloader, lr_scheduler
    )
    if ema_controlnet is not None:
        ema_controlnet.to(accelerator.device)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=vars(args))

    lpips_loss_fn = pyiqa.create_metric("lpips", device=accelerator.device, as_loss=True)
    if accelerator.is_main_process:
        ssim_metric = pyiqa.create_metric("ssim", device=accelerator.device, as_loss=False)
        lpips_metric = pyiqa.create_metric("lpips", device=accelerator.device, as_loss=False)

    def encode_anchor(pixel_values, anchor_latents):
        """Cross-attention tokens: [CLIP image embedding, 4x4 anchor latent tokens]."""
        pixel_values = resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0
        pixel_values = feature_extractor(images=pixel_values, do_normalize=True, do_center_crop=False,
                                         do_resize=False, do_rescale=False, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=accelerator.device, dtype=weight_dtype)
        clip_tokens = image_encoder(pixel_values).image_embeds.unsqueeze(1)
        latent_tokens = latent_tokenizer(anchor_latents.to(weight_dtype))
        return torch.cat([clip_tokens, latent_tokens], dim=1)

    global_step, first_epoch, resume_step = 0, 0, 0
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if path == "latest":
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-") and d != "checkpoint-best"]
            path = max(dirs, key=lambda d: int(d.split("-")[1])) if dirs else None
        if path is not None:
            accelerator.print(f"Resuming from {path}")
            accelerator.load_state(os.path.join(args.output_dir, os.path.basename(path)))
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = (global_step * args.gradient_accumulation_steps) % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    best_lpips = float("inf")
    val_sample = None
    if accelerator.is_main_process and args.val_data_path:
        val_sample = load_validation_sample(args.val_data_path, args.val_frame_idx, args.task, args.skip_frame)

    # Log-normal training sigma distribution, narrowed for larger frame gaps.
    sigma_loc = 0.7 - 0.1 * (args.skip_frame - 1)
    sigma_scale = 1.6 - 0.1 * (args.skip_frame - 1)

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    for epoch in range(first_epoch, num_train_epochs):
        controlnet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(controlnet):
                # The branches (one for VFP, forward + backward for VFI) are stacked along the
                # batch dimension and share the target, the noise level and the dropout mask.
                keys = [("anchor_values", "event_values")]
                if args.task == "vfi":
                    keys.append(("anchor_values_bwd", "event_values_bwd"))
                num_branches = len(keys)
                anchor_images = torch.cat([batch[a] for a, _ in keys]).to(accelerator.device, dtype=weight_dtype)
                event_values = torch.cat([batch[e] for _, e in keys]).to(accelerator.device, dtype=weight_dtype)
                target_images = batch["target_values"].to(accelerator.device, dtype=weight_dtype)

                target_latents = vae_encode(vae, target_images)
                anchor_latents = vae_encode(vae, anchor_images)
                residual_gt = target_latents.repeat(num_branches, 1, 1, 1) - anchor_latents
                bsz = target_latents.shape[0]

                # Noise-augmented anchor for the CLIP branch.
                cond_sigmas = rand_log_normal([bsz], loc=-3.0, scale=0.5).to(target_latents).repeat(num_branches)
                anchor_pixel_values = torch.randn_like(anchor_images) * cond_sigmas[:, None, None, None] + anchor_images

                if args.use_event_prior:
                    init_latents = encode_event_prior(er_vae, event_values).to(residual_gt.dtype)
                else:
                    init_latents = torch.zeros_like(residual_gt)

                noise = torch.randn_like(init_latents)
                sigmas = rand_log_normal([bsz], loc=sigma_loc, scale=sigma_scale).to(target_latents.device)
                sigmas = torch.clamp(sigmas, min=args.sigma_min, max=args.sigma_max).repeat(num_branches)
                sigmas = sigmas[:, None, None, None]
                noisy_latents = init_latents + noise * sigmas
                timesteps = 0.25 * torch.log(torch.clamp(sigmas, min=1e-6))
                timesteps = timesteps.view(-1).to(device=accelerator.device, dtype=weight_dtype)
                model_input = noisy_latents / ((sigmas ** 2 + 1) ** 0.5)

                encoder_hidden_states = encode_anchor(anchor_pixel_values.float(), anchor_latents)
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=target_latents.device, generator=generator)
                    drop = (random_p < 2 * args.conditioning_dropout_prob).repeat(num_branches).reshape(-1, 1, 1)
                    encoder_hidden_states = torch.where(drop, torch.zeros_like(encoder_hidden_states),
                                                        encoder_hidden_states)

                down_res, mid_res = controlnet(
                    sample=model_input.to(weight_dtype), timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states.to(weight_dtype),
                    controlnet_cond=event_values, return_dict=False,
                )
                model_pred = unet(
                    sample=model_input.to(weight_dtype), timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states.to(weight_dtype),
                    down_block_additional_residuals=[s.to(dtype=weight_dtype) for s in down_res],
                    mid_block_additional_residual=mid_res.to(dtype=weight_dtype),
                ).sample
                pred_residual = edm_denoise(model_pred, noisy_latents, sigmas)

                # EDM-weighted MSE, summed over branches (Eq. 3).
                safe_sigmas = torch.clamp(sigmas, min=1e-6)
                loss_weight = torch.clamp((1 + safe_sigmas ** 2) * safe_sigmas ** -2.0, max=args.loss_weight_max)
                loss_mse = num_branches * torch.mean(
                    loss_weight.float() * (pred_residual.float() - residual_gt.float()) ** 2)

                with torch.amp.autocast("cuda", dtype=weight_dtype):
                    decoded = vae.decode((anchor_latents + pred_residual) / vae.config.scaling_factor).sample
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    target_rgb = (target_images / 2 + 0.5).clamp(0, 1)

                    if args.task == "vfi":
                        # Eq. 4: the prediction from the closer anchor gets the larger weight.
                        d_fwd = batch["d_fwd"].to(decoded.device, dtype=decoded.dtype).view(-1, 1, 1, 1)
                        d_bwd = batch["d_bwd"].to(decoded.device, dtype=decoded.dtype).view(-1, 1, 1, 1)
                        decoded_fwd, decoded_bwd = decoded.chunk(2)
                        decoded = (d_bwd * decoded_fwd + d_fwd * decoded_bwd) / (d_fwd + d_bwd)

                    loss_lpips = lpips_loss_fn(decoded, target_rgb).mean()
                    loss_pixel_l1 = F.l1_loss(decoded, target_rgb)

                    # L1 weighted by per-pixel event activity, normalized per sample.
                    abs_error = torch.abs(decoded - target_rgb)
                    activity = event_values.detach().float().abs().mean(dim=1, keepdim=True)
                    activity = activity.view(num_branches, bsz, *activity.shape[1:]).amax(dim=0)
                    activity = F.interpolate(activity, size=decoded.shape[-2:], mode="bilinear", align_corners=False)
                    event_mask = (activity / activity.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)).clamp(0.0, 1.0)
                    event_weight = event_mask.to(abs_error.dtype).expand_as(abs_error)
                    loss_event_l1 = (abs_error * event_weight).sum() / event_weight.sum().clamp_min(1.0)

                loss = (loss_mse + args.lambda_lpips * loss_lpips + args.lambda_pixel_l1 * loss_pixel_l1
                        + args.lambda_event_l1 * loss_event_l1)

                avg_loss = accelerator.gather(loss.unsqueeze(0)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            if ema_controlnet is not None:
                ema_controlnet.step(controlnet.parameters())
            progress_bar.update(1)
            global_step += 1
            accelerator.log({
                "train/loss": train_loss, "train/mse": loss_mse.item(), "train/lpips": loss_lpips.item(),
                "train/pixel_l1": loss_pixel_l1.item(), "train/event_l1": loss_event_l1.item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }, step=global_step)
            train_loss = 0.0

            if accelerator.is_main_process:
                if val_sample is not None and global_step % args.validation_steps == 0:
                    best_lpips = validate(args, accelerator, global_step, val_sample, best_lpips,
                                          vae, er_vae, image_encoder, feature_extractor, unet, controlnet,
                                          latent_tokenizer, ema_controlnet, ssim_metric, lpips_metric)
                if global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        prune_checkpoints(args.output_dir, args.checkpoints_total_limit)
                    accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))

            progress_bar.set_postfix(loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    final_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    if accelerator.is_main_process and not os.path.exists(final_dir):
        if args.checkpoints_total_limit is not None:
            prune_checkpoints(args.output_dir, args.checkpoints_total_limit)
        accelerator.save_state(final_dir)
    accelerator.end_training()


@torch.no_grad()
def validate(args, accelerator, global_step, val_sample, best_lpips, vae, er_vae, image_encoder, feature_extractor,
             unet, controlnet, latent_tokenizer, ema_controlnet, ssim_metric, lpips_metric):
    """Predicts the validation frame with EMA weights; keeps checkpoint-best by LPIPS."""
    target, branches = val_sample
    controlnet = accelerator.unwrap_model(controlnet)
    controlnet.eval()
    if ema_controlnet is not None:
        ema_controlnet.store(controlnet.parameters())
        ema_controlnet.copy_to(controlnet.parameters())

    pipeline = EvFRAPipeline(vae, er_vae, image_encoder, feature_extractor, unet, controlnet,
                             accelerator.unwrap_model(latent_tokenizer)).to(accelerator.device)
    predictions = []
    for anchor, event_stack, event_frame, _ in branches:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        with torch.autocast(accelerator.device.type, enabled=accelerator.mixed_precision == "fp16"):
            predictions.append(pipeline(anchor, event_stack, event_frame, use_event_prior=args.use_event_prior,
                                        generator=generator))
    weights = [b[3] for b in branches]
    prediction = blend_images(predictions, weights)

    if ema_controlnet is not None:
        ema_controlnet.restore(controlnet.parameters())
    controlnet.train()

    pred, gt = np.array(prediction), np.array(target)
    psnr = calculate_psnr(pred, gt)
    ssim = calculate_ssim(pred, gt, ssim_metric)
    lpips = calculate_lpips(pred, gt, lpips_metric)
    # A model that ignores the events can still look good by copying (blending) the
    # anchors; only accept checkpoints that beat that baseline.
    anchor_copy = blend_images([b[0] for b in branches], weights)
    anchor_copy_lpips = calculate_lpips(np.array(anchor_copy), gt, lpips_metric)
    accelerator.log({"val/psnr": psnr, "val/ssim": ssim, "val/lpips": lpips,
                     "val/anchor_copy_lpips": anchor_copy_lpips}, step=global_step)
    logger.info(f"step {global_step}: val PSNR {psnr:.2f} SSIM {ssim:.4f} LPIPS {lpips:.4f} "
                f"(anchor copy {anchor_copy_lpips:.4f})")

    val_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)
    prediction.save(os.path.join(val_dir, f"step_{global_step}.png"))

    if lpips < anchor_copy_lpips and lpips < best_lpips:
        best_lpips = lpips
        best_dir = os.path.join(args.output_dir, "checkpoint-best")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        accelerator.save_state(best_dir)
        with open(os.path.join(args.output_dir, "best_checkpoint_info.txt"), "w") as f:
            f.write(f"Step: {global_step}\nLPIPS: {lpips:.6f}\nPSNR: {psnr:.4f}\nSSIM: {ssim:.4f}\n"
                    f"AnchorCopyLPIPS: {anchor_copy_lpips:.6f}\n")
    return best_lpips


if __name__ == "__main__":
    main()
