"""
Stage 1: ER-VAE.

Fine-tunes the SD2 VAE encoder so that encoding the accumulated event frame
between two consecutive frames gives the latent residual z_target - z_anchor.
Loss = L1(residual) + gamma * MSE(decode(z_anchor + residual), target), with
gamma following a cosine ramp from gamma_min to gamma_max.
"""
import argparse
import glob
import math
import os
import re

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm import tqdm

from evfra.data import ERVAEDataset


def cosine_gamma(gamma_min, gamma_max, progress):
    progress = min(max(progress, 0.0), 1.0)
    return gamma_min + 0.5 * (gamma_max - gamma_min) * (1 - math.cos(math.pi * progress))


@torch.no_grad()
def encode_latent(vae, image):
    return vae.encode(image).latent_dist.mode() * vae.config.scaling_factor


def compute_losses(vae_event, vae_gt, batch, device, dtype, use_amp):
    anchor = batch["anchor_image"].to(device, dtype=dtype)
    target = batch["target_image"].to(device, dtype=dtype)
    event = batch["event_image"].to(device, dtype=dtype)

    with torch.amp.autocast("cuda", enabled=use_amp):
        anchor_latent = encode_latent(vae_gt, anchor.float()).float()
        target_latent = encode_latent(vae_gt, target.float()).float()
        residual_gt = target_latent - anchor_latent

        residual_pred = vae_event.encode(event.float()).latent_dist.mean * vae_event.config.scaling_factor
        latent_loss = F.l1_loss(residual_pred.float(), residual_gt.float())

        decoded = vae_event.decode((anchor_latent + residual_pred) / vae_event.config.scaling_factor).sample.float()
        img_loss = F.mse_loss(decoded.float(), target.float())
    return latent_loss, img_loss


@torch.no_grad()
def evaluate(vae_event, vae_gt, loader, device, dtype, use_amp, args):
    vae_event.eval()
    total = latent_total = img_total = 0.0
    for count, batch in enumerate(loader):
        latent_loss, img_loss = compute_losses(vae_event, vae_gt, batch, device, dtype, use_amp)
        gamma = cosine_gamma(args.gamma_min, args.gamma_max, count / max(1, len(loader)))
        total += (latent_loss + gamma * img_loss).item()
        latent_total += latent_loss.item()
        img_total += img_loss.item()
    n = max(len(loader), 1)
    return total / n, latent_total / n, img_total / n


def save_checkpoint(path, epoch, vae_event, optimizer, scheduler, scaler, best_val_loss):
    torch.save({
        "epoch": epoch,
        "vae_event_state": vae_event.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)


def latest_checkpoint(save_dir):
    ckpts = glob.glob(os.path.join(save_dir, "checkpoint_epoch_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(re.search(r"checkpoint_epoch_(\d+)\.pt", p).group(1)))


def parse_args():
    parser = argparse.ArgumentParser(description="EvFRA stage 1: ER-VAE training")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--train_data_path", type=str, required=True, help="BS-ERGB train split")
    parser.add_argument("--val_data_path", type=str, required=True, help="BS-ERGB valid split")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gamma_min", type=float, default=0.3)
    parser.add_argument("--gamma_max", type=float, default=0.7)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--save_interval", type=int, default=2)
    parser.add_argument("--val_interval", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from (default: latest in output_dir)")
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.precision == "fp16" and device.type == "cuda"
    dtype = torch.float16 if use_amp else torch.float32

    wandb = None
    if args.report_to == "wandb":
        import wandb
        wandb.init(project="EvFRA-ER-VAE", config=vars(args))

    vae_gt = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device).eval()
    vae_gt.requires_grad_(False)

    # Only the encoder (and quant_conv) is trained; the decoder stays the SD2 decoder.
    vae_event = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    vae_event.requires_grad_(False)
    trainable_params = list(vae_event.encoder.parameters()) + list(vae_event.quant_conv.parameters())
    for p in trainable_params:
        p.requires_grad = True
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    train_loader = DataLoader(ERVAEDataset(args.train_data_path, args.width, args.height, random_crop=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(ERVAEDataset(args.val_data_path, args.width, args.height, random_crop=False),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_val_loss = 0, float("inf")
    resume = args.resume or latest_checkpoint(args.output_dir)
    if resume:
        state = torch.load(resume, map_location=device)
        vae_event.load_state_dict(state["vae_event_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        best_val_loss = state["best_val_loss"]
        start_epoch = state["epoch"] + 1
        print(f"Resumed from {resume} (epoch {start_epoch})")

    num_batches = max(1, len(train_loader))
    total_steps = args.epochs * num_batches

    for epoch in range(start_epoch, args.epochs):
        vae_event.train()
        sums = [0.0, 0.0, 0.0]
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress):
            gamma = cosine_gamma(args.gamma_min, args.gamma_max, (epoch * num_batches + step) / max(1, total_steps))
            optimizer.zero_grad(set_to_none=True)
            latent_loss, img_loss = compute_losses(vae_event, vae_gt, batch, device, dtype, use_amp)
            loss = latent_loss + gamma * img_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            for i, v in enumerate((loss, latent_loss, img_loss)):
                sums[i] += v.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", latent=f"{latent_loss.item():.4f}",
                                 img=f"{img_loss.item():.4f}", gamma=f"{gamma:.3f}")
            if wandb:
                wandb.log({"train/total_loss": loss.item(), "train/latent_loss": latent_loss.item(),
                           "train/img_loss": img_loss.item(), "train/lr": optimizer.param_groups[0]["lr"]})
        scheduler.step()
        print(f"Epoch {epoch + 1}: loss={sums[0] / num_batches:.4f}, latent={sums[1] / num_batches:.4f}, "
              f"img={sums[2] / num_batches:.4f}")

        if (epoch + 1) % args.val_interval == 0:
            val_loss, val_latent, val_img = evaluate(vae_event, vae_gt, val_loader, device, dtype, use_amp, args)
            print(f"  val: loss={val_loss:.4f}, latent={val_latent:.4f}, img={val_img:.4f}")
            if wandb:
                wandb.log({"val/loss": val_loss, "val/latent_loss": val_latent, "val/img_loss": val_img,
                           "epoch": epoch + 1})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(os.path.join(args.output_dir, "best.pt"), epoch, vae_event, optimizer,
                                scheduler, scaler, best_val_loss)

        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt"), epoch, vae_event,
                            optimizer, scheduler, scaler, best_val_loss)

    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
