import torch

from .controlnet import ControlNetSD2Model


class LatentTokenizer(torch.nn.Module):
    """Pools the anchor latent to a grid and projects each cell to a cross-attention token."""

    def __init__(self, latent_channels=4, grid_size=4, hidden_dim=1024):
        super().__init__()
        self.grid_size = grid_size
        self.pool = torch.nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.proj = torch.nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)

    def forward(self, latents):
        return self.proj(self.pool(latents)).flatten(2).transpose(1, 2)


def load_latent_tokenizer(path, map_location="cpu"):
    tokenizer = LatentTokenizer()
    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    tokenizer.load_state_dict({k.removeprefix("module."): v for k, v in state_dict.items()})
    return tokenizer


def load_er_vae(pretrained_model_name_or_path, checkpoint_path, **kwargs):
    """ER-VAE: an SD2 VAE whose encoder was fine-tuned to map event stacks to latent residuals."""
    from diffusers import AutoencoderKL

    er_vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", **kwargs)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    er_vae.load_state_dict(state.get("vae_event_state", state))
    return er_vae


__all__ = ["ControlNetSD2Model", "LatentTokenizer", "load_latent_tokenizer", "load_er_vae"]
