# Adapted from diffusers' ControlNetModel.
#
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn, get_down_block
from diffusers.utils import BaseOutput, logging

from ..events import NUM_STACKS

logger = logging.get_logger(__name__)


@dataclass
class ControlNetOutput(BaseOutput):
    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNetConditioningEmbedding(nn.Module):
    """Encodes the event stack (B, NUM_STACKS, H, W) into (B, C, H/8, W/8) features."""

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = NUM_STACKS,
        block_out_channels: Tuple[int, ...] = (32, 64, 128, 256),
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = F.silu(self.conv_in(conditioning))
        for block in self.blocks:
            embedding = F.silu(block(embedding))
        return self.conv_out(embedding)


class ControlNetSD2Model(ModelMixin, ConfigMixin):
    """
    Event ControlNet for the Stable Diffusion 2.x UNet.

    Takes the noisy latent, the timestep, the anchor tokens and the event stack
    and returns the residuals added to the UNet down/mid blocks.
    `up_block_types`, `out_channels` and `attention_head_dim` are kept for
    config compatibility with released checkpoints.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        attention_head_dim: Union[int, Tuple[int]] = (40, 80, 160, 160),
        conditioning_channels: int = NUM_STACKS,
        conditioning_embedding_out_channels: Tuple[int, ...] = (32, 64, 128, 256),
    ):
        super().__init__()
        self.sample_size = sample_size

        if len(down_block_types) != len(up_block_types):
            raise ValueError("`down_block_types` and `up_block_types` must have the same length.")
        if len(block_out_channels) != len(down_block_types):
            raise ValueError("`block_out_channels` and `down_block_types` must have the same length.")
        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError("`attention_head_dim` and `down_block_types` must have the same length.")
        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError("`cross_attention_dim` and `down_block_types` must have the same length.")
        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError("`layers_per_block` and `down_block_types` must have the same length.")

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        self.controlnet_down_blocks.append(zero_module(nn.Conv2d(output_channel, output_channel, kernel_size=1)))

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            self.down_blocks.append(
                get_down_block(
                    down_block_type,
                    num_layers=layers_per_block[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    downsample_padding=1,
                    resnet_eps=1e-5,
                    cross_attention_dim=cross_attention_dim[i],
                    num_attention_heads=8,
                    attention_head_dim=8,  # unused by these blocks; set to silence the diffusers default warning
                    resnet_act_fn="silu",
                    resnet_groups=32,
                )
            )

            # One zero-conv per residual: one per layer, plus one for the downsampler.
            for _ in range(layers_per_block[i]):
                self.controlnet_down_blocks.append(
                    zero_module(nn.Conv2d(output_channel, output_channel, kernel_size=1))
                )
            if not is_final_block:
                self.controlnet_down_blocks.append(
                    zero_module(nn.Conv2d(output_channel, output_channel, kernel_size=1))
                )

        mid_block_channel = block_out_channels[-1]
        self.controlnet_mid_block = zero_module(nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1))

        mid_cross_attention_dim = cross_attention_dim[-1]
        if isinstance(mid_cross_attention_dim, (tuple, list)):
            mid_cross_attention_dim = mid_cross_attention_dim[0]
        mid_transformer_layers = transformer_layers_per_block[-1]
        if isinstance(mid_transformer_layers, (tuple, list)):
            mid_transformer_layers = mid_transformer_layers[0]

        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=mid_block_channel,
            temb_channels=time_embed_dim,
            num_layers=mid_transformer_layers,
            resnet_eps=1e-5,
            cross_attention_dim=mid_cross_attention_dim,
            num_attention_heads=8,
            resnet_act_fn="silu",
            resnet_groups=32,
        )

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (32, 64, 128, 256),
        load_weights_from_unet: bool = True,
        conditioning_channels: int = NUM_STACKS,
    ):
        """Builds the ControlNet from a UNet config and copies its encoder weights."""
        up_block_types = tuple(
            "CrossAttnUpBlock2D" if "CrossAttn" in t else "UpBlock2D"
            for t in reversed(unet.config.down_block_types)
        )
        controlnet = cls(
            in_channels=unet.config.in_channels,
            down_block_types=unet.config.down_block_types,
            up_block_types=up_block_types,
            block_out_channels=unet.config.block_out_channels,
            transformer_layers_per_block=getattr(unet.config, "transformer_layers_per_block", 1),
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=getattr(unet.config, "attention_head_dim", (64, 64, 128, 128)),
            sample_size=getattr(unet.config, "sample_size", None),
            layers_per_block=getattr(unet.config, "layers_per_block", 2),
            conditioning_channels=conditioning_channels,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
        )

        if load_weights_from_unet:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
            # The SD2 UNet uses linear attention projections (proj_in/proj_out) while
            # these blocks use 1x1 convs; load_state_dict copies every matching tensor
            # before raising on those shape mismatches, which stay randomly initialized.
            for name in ("down_blocks", "mid_block"):
                try:
                    getattr(controlnet, name).load_state_dict(getattr(unet, name).state_dict(), strict=False)
                except RuntimeError as e:
                    logger.info(f"{name}: kept random init for mismatched tensors ({str(e).count('size mismatch')})")

        return controlnet

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.FloatTensor = None,
        return_dict: bool = True,
        conditioning_scale: float = 1.0,
    ) -> Union[ControlNetOutput, Tuple]:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        # Timesteps always returns fp32; cast for mixed precision.
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)
        if controlnet_cond is not None:
            sample = sample + self.controlnet_cond_embedding(controlnet_cond)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if getattr(downsample_block, "has_cross_attention", False):
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        down_block_res_samples = tuple(
            block(res) * conditioning_scale
            for res, block in zip(down_block_res_samples, self.controlnet_down_blocks)
        )
        mid_block_res_sample = self.controlnet_mid_block(sample) * conditioning_scale

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)
        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_gradient_checkpointing(self):
        self._set_gradient_checkpointing(self, True)

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Any] = None):
        from diffusers.models.attention_processor import XFormersAttnProcessor

        self.set_attn_processor(XFormersAttnProcessor(attention_op=attention_op))

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]], _remove_lora=False):
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                "Cannot call `set_default_attn_processor` when attention processors are of type "
                f"{next(iter(self.attn_processors.values()))}"
            )
        self.set_attn_processor(processor, _remove_lora=True)

    def set_attention_slice(self, slice_size: Union[str, int, List[int]]) -> None:
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)
            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)
        if slice_size == "auto":
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            slice_size = num_sliceable_layers * [1]
        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers."
            )
        for size, dim in zip(slice_size, sliceable_head_dims):
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())
            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)
