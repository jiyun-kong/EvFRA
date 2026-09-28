"""
Packs a training checkpoint into the release layout used by evaluate.py:

    <output_dir>/controlnet/           EMA ControlNet weights
    <output_dir>/latent_tokenizer.pth
    <output_dir>/er_vae.pt             ER-VAE weights only (no optimizer state)
"""
import argparse
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evfra.models import ControlNetSD2Model  # noqa: E402

EMA_CONFIG_KEYS = ["decay", "inv_gamma", "min_decay", "optimization_step", "power",
                   "update_after_step", "use_ema_warmup"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="e.g. experiments/phase3/checkpoint-best")
    parser.add_argument("--er_vae_path", type=str, required=True, help="Stage-1 checkpoint, e.g. experiments/er_vae/best.pt")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    controlnet = ControlNetSD2Model.from_pretrained(os.path.join(args.checkpoint_dir, "controlnet_ema"))
    controlnet.save_pretrained(os.path.join(args.output_dir, "controlnet"))
    # Drop the EMA bookkeeping entries that EMAModel stores in the config.
    config_path = os.path.join(args.output_dir, "controlnet", "config.json")
    with open(config_path) as f:
        config = json.load(f)
    for key in EMA_CONFIG_KEYS + ["_name_or_path"]:
        config.pop(key, None)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    shutil.copy(os.path.join(args.checkpoint_dir, "latent_tokenizer.pth"),
                os.path.join(args.output_dir, "latent_tokenizer.pth"))

    state = torch.load(args.er_vae_path, map_location="cpu", weights_only=True)
    torch.save({"vae_event_state": state.get("vae_event_state", state)}, os.path.join(args.output_dir, "er_vae.pt"))
    print(f"Exported to {args.output_dir}")


if __name__ == "__main__":
    main()
