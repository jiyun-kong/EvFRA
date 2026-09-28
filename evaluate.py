"""
Video frame prediction / interpolation on BS-ERGB / HS-ERGB / GoPro.

Frames t and t+K+1 are keyframes, and the K frames in between are synthesized;
the next keyframe pair starts at t+K+1.
  vfp: each target is predicted from frame t and the events since t.
  vfi: each target is also predicted backward from frame t+K+1 with the
       time-reversed events, and the two predictions are blended with weights
       proportional to the distance to the other keyframe (Eq. 4).
Frames are upsampled 2x, predicted in overlapping 512x320 tiles, merged, and
resized back to the original resolution for evaluation. Reported metrics are
averaged over all synthesized frames (keyframes are not counted).

Anchors are split round-robin with --shard_id/--num_shards; finished frames are
skipped, so an interrupted run can be relaunched with the same command.
"""
import argparse
import os

import numpy as np
import pyiqa
import torch
from accelerate.utils import set_seed
from PIL import Image
from tqdm.auto import tqdm

from evfra import EvFRAPipeline
from evfra.data import UPSAMPLE_SCALE, list_frames, list_sequences
from evfra.events import DATASETS, build_event_stack, er_vae_input
from evfra.utils import calculate_lpips, calculate_psnr, calculate_ssim, get_views, merge_views


def parse_args():
    parser = argparse.ArgumentParser(description="EvFRA evaluation")
    parser.add_argument("--task", type=str, default="vfp", choices=["vfp", "vfi"])
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS))
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_frames", type=int, required=True, help="Frames synthesized per keyframe (K)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--controlnet_path", type=str, default=None)
    parser.add_argument("--latent_tokenizer_path", type=str, default=None,
                        help="Default: latent_tokenizer.pth next to the ControlNet folder")
    parser.add_argument("--er_vae_path", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--overlap_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sequences", type=str, nargs="*", default=None, help="Restrict to these sequences")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--summarize_only", action="store_true", help="Only aggregate existing metrics")
    args = parser.parse_args()
    if not args.summarize_only and not (args.controlnet_path and args.er_vae_path):
        parser.error("--controlnet_path and --er_vae_path are required")
    return args


def build_anchor_list(args):
    """(seq_name, image_folder, event_folder, frames, first_target_idx) for every anchor."""
    K = args.num_frames
    # Events needed: (t, t+K] for VFP, (t, t+K+1] for VFI.
    num_event_files = K if args.task == "vfp" else K + 1
    anchors = []
    for name, image_folder, event_folder in list_sequences(args.dataset, args.data_root):
        if args.sequences and name not in args.sequences and os.path.basename(name) not in args.sequences:
            continue
        frames = list_frames(image_folder)
        available = set(f for f in os.listdir(event_folder) if f.endswith(".npz"))
        skipped = 0
        for idx in range(1, len(frames), K + 1):
            if idx + K >= len(frames):
                continue
            # Some HS-ERGB event streams end before the frames do.
            if any(f"{i:06d}.npz" not in available for i in range(idx - 1, idx - 1 + num_event_files)):
                skipped += 1
                continue
            anchors.append((name, image_folder, event_folder, frames, idx))
        if skipped:
            print(f"{name}: skipped {skipped} anchors without events")
    return anchors


@torch.no_grad()
def predict_frame(pipeline, anchor_full, event_folder, start_idx, end_idx, orig_size, cfg, args, reverse=False):
    """
    Tiled prediction at 2x resolution from the events between frames start_idx and
    end_idx (played backward if `reverse`). Returns a uint8 (2H, 2W, 3) array.
    """
    w, h = orig_size
    stack, is_empty = build_event_stack(event_folder, start_idx, end_idx, w, h, UPSAMPLE_SCALE, cfg,
                                        reverse=reverse)
    views = get_views(anchor_full.height, anchor_full.width, 320, 512, args.overlap_ratio)
    tiles = []
    for x0, y0, x1, y1 in views:
        tile_stack = stack[:, y0:y1, x0:x1].contiguous()
        with torch.autocast(pipeline.device.type, enabled=pipeline.device.type == "cuda"):
            tiles.append(pipeline(
                anchor_full.crop((x0, y0, x1, y1)), tile_stack, er_vae_input(tile_stack, is_empty),
                num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
            ))
    return merge_views(views, tiles, anchor_full.height, anchor_full.width)


def summarize(output_dir):
    """Mean over all predicted frames, plus per-sequence means."""
    metrics_root = os.path.join(output_dir, "metrics")
    per_seq = {}
    for cur, _, files in os.walk(metrics_root):
        for f in files:
            values = dict(line.split(":", 1) for line in open(os.path.join(cur, f)) if ":" in line)
            per_seq.setdefault(os.path.relpath(cur, metrics_root), []).append(
                [float(values[k]) for k in ("PSNR", "SSIM", "LPIPS")])
    if not per_seq:
        print("No metrics found.")
        return
    all_frames = np.array([m for ms in per_seq.values() for m in ms])
    print(f"{output_dir}: {len(all_frames)} frames, {len(per_seq)} sequences")
    for name, ms in sorted(per_seq.items()):
        ms = np.mean(ms, axis=0)
        print(f"  {name:40s} PSNR {ms[0]:6.2f}  SSIM {ms[1]:.3f}  LPIPS {ms[2]:.3f}")
    mean = all_frames.mean(axis=0)
    print(f"  {'mean':40s} PSNR {mean[0]:6.2f}  SSIM {mean[1]:.3f}  LPIPS {mean[2]:.3f}")


def main():
    args = parse_args()
    if args.summarize_only:
        summarize(args.output_dir)
        return

    cfg = DATASETS[args.dataset]
    K = args.num_frames
    anchors = build_anchor_list(args)[args.shard_id::args.num_shards]
    print(f"{len(anchors)} anchors, {len(anchors) * K} frames -> {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = EvFRAPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args.controlnet_path, args.er_vae_path,
        latent_tokenizer_path=args.latent_tokenizer_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    ssim_metric = pyiqa.create_metric("ssim", device=device, as_loss=False)
    lpips_metric = pyiqa.create_metric("lpips", device=device, as_loss=False)

    for name, image_folder, event_folder, frames, idx in tqdm(anchors):
        anchor_idx = idx - 1
        pred_dir = os.path.join(args.output_dir, "prediction", name)
        metrics_dir = os.path.join(args.output_dir, "metrics", name)
        if os.path.exists(os.path.join(metrics_dir, f"{idx + K - 1:06d}.txt")):
            continue
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        set_seed(args.seed + anchor_idx)
        next_idx = idx + K
        anchor = Image.open(os.path.join(image_folder, frames[anchor_idx])).convert("RGB")
        w, h = anchor.size
        size_2x = (w * UPSAMPLE_SCALE, h * UPSAMPLE_SCALE)
        anchor_full = anchor.resize(size_2x, Image.LANCZOS)
        if args.task == "vfi":
            next_full = Image.open(os.path.join(image_folder, frames[next_idx])).convert("RGB").resize(
                size_2x, Image.LANCZOS)

        for target_idx in range(idx, idx + K):
            pred = predict_frame(pipeline, anchor_full, event_folder, anchor_idx, target_idx, (w, h), cfg, args)
            if args.task == "vfi":
                pred_bwd = predict_frame(pipeline, next_full, event_folder, target_idx, next_idx, (w, h), cfg, args,
                                         reverse=True)
                d_fwd, d_bwd = target_idx - anchor_idx, next_idx - target_idx
                pred = (d_bwd * pred.astype(np.float32) + d_fwd * pred_bwd.astype(np.float32)) / (d_fwd + d_bwd)
                pred = np.clip(np.round(pred), 0, 255).astype(np.uint8)
            pred = Image.fromarray(pred).resize((w, h), Image.LANCZOS)
            gt = np.array(Image.open(os.path.join(image_folder, frames[target_idx])).convert("RGB"))
            pred_np = np.array(pred)
            psnr = calculate_psnr(pred_np, gt)
            ssim = calculate_ssim(pred_np, gt, ssim_metric)
            lpips = calculate_lpips(pred_np, gt, lpips_metric)

            pred.save(os.path.join(pred_dir, f"{target_idx:06d}.png"))
            with open(os.path.join(metrics_dir, f"{target_idx:06d}.txt"), "w") as f:
                f.write(f"Anchor: {anchor_idx}\nPSNR: {psnr:.4f}\nSSIM: {ssim:.4f}\nLPIPS: {lpips:.4f}\n")

    summarize(args.output_dir)


if __name__ == "__main__":
    main()
