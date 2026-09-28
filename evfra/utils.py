from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------

def get_views(height, width, window_height=320, window_width=512, overlap_ratio=0.1):
    """Window boxes (x0, y0, x1, y1) covering a (height, width) image with the given overlap."""
    stride_x = int(window_width * (1 - overlap_ratio))
    stride_y = int(window_height * (1 - overlap_ratio))

    num_h = (height - window_height) // stride_y + 1
    if (height - window_height) % stride_y != 0:
        num_h += 1
    num_w = (width - window_width) // stride_x + 1
    if (width - window_width) % stride_x != 0:
        num_w += 1

    views = []
    for i in range(num_h * num_w):
        y0 = int((i // num_w) * stride_y)
        y1 = y0 + window_height
        if y1 > height:
            y1 = height
            y0 = y1 - window_height

        x0 = int((i % num_w) * stride_x)
        x1 = x0 + window_width
        if x1 > width:
            x1 = width
            x0 = x1 - window_width

        views.append((x0, y0, x1, y1))
    return views


def merge_views(views, view_images, height, width):
    """Averages overlapping tiles back into a (height, width, 3) uint8 image."""
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)
    for (x0, y0, x1, y1), view in zip(views, view_images):
        canvas[y0:y1, x0:x1] += np.array(view, dtype=np.float32)
        weight[y0:y1, x0:x1] += 1.0
    return (canvas / np.maximum(weight, 1e-8)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Antialiased resize for the CLIP image encoder (from diffusers' SVD pipeline)
# ---------------------------------------------------------------------------

def resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])
    sigmas = (max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) / 2.0, 0.001))

    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
    if ks[0] % 2 == 0:
        ks = ks[0] + 1, ks[1]
    if ks[1] % 2 == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)
    return F.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)


def _compute_padding(kernel_size):
    computed = [k - 1 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]
    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        pad_front = computed_tmp // 2
        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = computed_tmp - pad_front
    return out_padding


def _filter2d(input, kernel):
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]

    padding_shape: List[int] = _compute_padding([height, width])
    input = F.pad(input, padding_shape, mode="reflect")

    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    return output.view(b, c, h, w)


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])
    batch_size = sigma.shape[0]
    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    return _filter2d(out_x, kernel_y[..., None])


# ---------------------------------------------------------------------------
# Metrics (uint8 HWC arrays)
# ---------------------------------------------------------------------------

_to_tensor = transforms.ToTensor()


def calculate_psnr(pred, gt):
    mse = F.mse_loss(_to_tensor(pred), _to_tensor(gt))
    if mse == 0:
        return 100.0
    return float(10 * torch.log10(1 / mse))


def calculate_ssim(pred, gt, ssim_metric):
    return float(ssim_metric(_to_tensor(pred).unsqueeze(0), _to_tensor(gt).unsqueeze(0)).mean())


def calculate_lpips(pred, gt, lpips_metric):
    return float(lpips_metric(_to_tensor(pred).unsqueeze(0), _to_tensor(gt).unsqueeze(0)).mean())
