"""EvFRA inference: EDM residual denoising initialized from the ER-VAE event prior."""
import os
from typing import Optional

import numpy as np
import PIL.Image
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor

from .models import ControlNetSD2Model, load_er_vae, load_latent_tokenizer
from .utils import resize_with_antialiasing

CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
SIGMA_MAX = 3.0
SIGMA_MIN = 0.05


def edm_sigmas(num_steps, device, sigma_max=SIGMA_MAX):
    """Log-uniform sigma schedule from sigma_max to SIGMA_MIN."""
    return torch.exp(torch.linspace(np.log(sigma_max), np.log(SIGMA_MIN), num_steps, device=device, dtype=torch.float32))


def edm_denoise(model_pred, noisy, sigma):
    """EDM output parameterization: D(x) = c_skip * x + c_out * F(x)."""
    c_out = -sigma / ((sigma ** 2 + 1) ** 0.5)
    c_skip = 1 / (sigma ** 2 + 1)
    return model_pred * c_out + c_skip * noisy


class EvFRAPipeline:
    """
    Predicts frame t+k from the anchor frame t and the events in (t, t+k].

    The model denoises the latent residual z_{t+k} - z_t. The trajectory starts
    from the ER-VAE encoding of the events (`prior_alpha=1`) or from Gaussian
    noise (`use_event_prior=False`), and the anchor is conditioned through CLIP
    image tokens plus latent tokens.
    """

    def __init__(self, vae, er_vae, image_encoder, feature_extractor, unet, controlnet, latent_tokenizer):
        self.vae = vae
        self.er_vae = er_vae
        self.image_encoder = image_encoder
        self.feature_extractor = feature_extractor
        self.unet = unet
        self.controlnet = controlnet
        self.latent_tokenizer = latent_tokenizer
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, controlnet_path, er_vae_path,
                        latent_tokenizer_path=None, torch_dtype=torch.float16):
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        if latent_tokenizer_path is None:
            latent_tokenizer_path = os.path.join(os.path.dirname(controlnet_path.rstrip("/")), "latent_tokenizer.pth")

        pipe = cls(
            vae=AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", variant="fp16"),
            er_vae=load_er_vae(pretrained_model_name_or_path, er_vae_path, variant="fp16"),
            image_encoder=CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL),
            feature_extractor=CLIPImageProcessor.from_pretrained(CLIP_MODEL),
            unet=UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", variant="fp16"),
            controlnet=ControlNetSD2Model.from_pretrained(controlnet_path),
            latent_tokenizer=load_latent_tokenizer(latent_tokenizer_path),
        )
        for module in pipe.modules():
            module.requires_grad_(False)
            module.eval()
            module.to(dtype=torch_dtype)
        return pipe

    def modules(self):
        return [self.vae, self.er_vae, self.image_encoder, self.unet, self.controlnet, self.latent_tokenizer]

    def to(self, device):
        self.device = torch.device(device)
        for module in self.modules():
            if module is not None:
                module.to(self.device)
        return self

    def encode_anchor(self, image, anchor_latents, do_classifier_free_guidance):
        """Cross-attention tokens: [CLIP image embedding, 4x4 anchor latent tokens]."""
        dtype = next(self.image_encoder.parameters()).dtype
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0
        pixel_values = self.feature_extractor(
            images=image, do_normalize=True, do_center_crop=False, do_resize=False, do_rescale=False,
            return_tensors="pt",
        ).pixel_values.to(device=self.device, dtype=dtype)

        clip_tokens = self.image_encoder(pixel_values).image_embeds.unsqueeze(1)
        latent_tokens = self.latent_tokenizer(anchor_latents.to(dtype))
        embeddings = torch.cat([clip_tokens, latent_tokens], dim=1)

        if do_classifier_free_guidance:
            embeddings = torch.cat([torch.zeros_like(embeddings), embeddings])
        return embeddings

    @torch.no_grad()
    def encode_event_prior(self, event_frame, dtype):
        """ER-VAE posterior mean of the event frame, i.e. the predicted latent residual."""
        event_input = event_frame.to(device=self.device, dtype=next(self.er_vae.parameters()).dtype)
        needs_upcasting = self.er_vae.dtype == torch.float16 and self.er_vae.config.force_upcast
        if needs_upcasting:
            self.er_vae.to(dtype=torch.float32)
            event_input = event_input.float()
        prior = self.er_vae.encode(event_input).latent_dist.mean * self.er_vae.config.scaling_factor
        if needs_upcasting:
            self.er_vae.to(dtype=torch.float16)
        return prior.to(dtype)

    @torch.no_grad()
    def __call__(
        self,
        image: PIL.Image.Image,
        event_stack: torch.Tensor,
        event_frame: Optional[torch.Tensor] = None,
        num_inference_steps: int = 5,
        guidance_scale: float = 1.0,
        noise_aug_strength: float = 0.02,
        use_event_prior: bool = True,
        prior_alpha: float = 1.0,
        sigma_start: Optional[float] = 0.1,
        generator: Optional[torch.Generator] = None,
    ) -> PIL.Image.Image:
        """
        Args:
            image: anchor frame (H, W divisible by 8).
            event_stack: (NUM_STACKS, H, W) multi-scale event stack in [-1, 1].
            event_frame: (1, 3, H, W) ER-VAE input, see `evfra.events.er_vae_input`.
            prior_alpha: blend between the event prior (1.0) and noise at sigma_max (0.0).
            sigma_start: the event prior is perturbed with noise of this level and denoised from
                sigma_start down to SIGMA_MIN. None starts from the prior as is with the full schedule
                from SIGMA_MAX. With num_inference_steps=0 the prior is decoded directly.
        """
        height, width = event_stack.shape[-2:]
        do_cfg = guidance_scale > 1

        image = self.image_processor.preprocess(image, height=height, width=width).to(self.device)
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
        image = image.to(dtype=self.vae.dtype)
        anchor_latents = self.vae.encode(image).latent_dist.mode() * self.vae.config.scaling_factor

        image_for_clip = image
        if noise_aug_strength > 0:
            strength = torch.tensor([noise_aug_strength], device=self.device, dtype=image.dtype)[:, None, None, None]
            image_for_clip = torch.randn_like(image) * strength + image
        embeddings = self.encode_anchor(image_for_clip, anchor_latents, do_cfg)
        anchor_latents = anchor_latents.to(embeddings.dtype)
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        sigmas = edm_sigmas(num_inference_steps, self.device, sigma_start or SIGMA_MAX)
        timesteps = 0.25 * torch.log(sigmas)

        if use_event_prior:
            prior = self.encode_event_prior(event_frame, embeddings.dtype)
            noise = torch.randn(prior.shape, generator=generator, device=self.device, dtype=prior.dtype) \
                if generator is not None else torch.randn_like(prior)
            if sigma_start is not None:
                latents = prior + noise * sigma_start
            else:
                latents = prior * prior_alpha + noise * SIGMA_MAX * (1.0 - prior_alpha)
            if num_inference_steps == 0:
                latents = prior
        else:
            shape = (1, 4, height // self.vae_scale_factor, width // self.vae_scale_factor)
            latents = randn_tensor(shape, generator=generator, device=self.device, dtype=embeddings.dtype) * sigmas[0]

        cond = event_stack.unsqueeze(0) if event_stack.ndim == 3 else event_stack
        cond = cond.to(device=self.device, dtype=latents.dtype)
        if do_cfg:
            cond = torch.cat([cond] * 2)

        for i, t in enumerate(timesteps):
            sigma = sigmas[i].view(-1, 1, 1, 1)
            model_input = torch.cat([latents] * 2) if do_cfg else latents
            model_input = model_input / ((sigma ** 2 + 1) ** 0.5)
            t_b = t.expand(model_input.shape[0]).to(dtype=embeddings.dtype)

            down_res, mid_res = self.controlnet(
                sample=model_input, timestep=t_b, encoder_hidden_states=embeddings,
                controlnet_cond=cond, return_dict=False,
            )
            model_pred = self.unet(
                sample=model_input, timestep=t_b, encoder_hidden_states=embeddings,
                down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res,
                return_dict=False,
            )[0]
            if do_cfg:
                pred_uncond, pred_cond = model_pred.chunk(2)
                model_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            denoised = edm_denoise(model_pred, latents, sigma)
            if i < len(sigmas) - 1:
                d = (latents - denoised) / sigma
                latents = latents + (sigmas[i + 1].view(-1, 1, 1, 1) - sigma) * d
            else:
                latents = denoised

        frame_latents = anchor_latents + latents
        decoded = self.vae.decode(frame_latents / self.vae.config.scaling_factor).sample.float()
        decoded = (decoded.cpu().permute(0, 2, 3, 1).numpy() / 2 + 0.5).clip(0, 1)
        return self.image_processor.numpy_to_pil(decoded)[0]
