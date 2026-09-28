"""Multi-scale event stack representation (Sec. 3.2)."""
import os

import numpy as np
import torch
from PIL import Image

NUM_STACKS = 10

# Per-dataset layout of an event sequence folder.
#   keys      : npz field names for (timestamp, x, y, polarity)
#   coord_div : raw coordinates are divided by this before rounding
DATASETS = {
    "bs_ergb": dict(image_dir="images", event_dir="events",
                    keys=("timestamp", "x", "y", "polarity"), coord_div=32),
    "hs_ergb": dict(image_dir="images_corrected", event_dir="events_aligned",
                    keys=("t", "x", "y", "p"), coord_div=1),
    "gopro": dict(image_dir="images", event_dir="events",
                  keys=("timestamp", "x", "y", "polarity"), coord_div=1),
}


def events_to_voxel_grid(events, num_bins, width, height):
    """Voxel grid with bilinear interpolation in time. `events` is an [N, 4] array of (t, x, y, p)."""
    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    if len(events) < 5:
        return voxel_grid.reshape(num_bins, height, width)

    events = events[np.argsort(events[:, 0])]
    first_stamp, last_stamp = events[0, 0], events[-1, 0]
    delta_t = last_stamp - first_stamp
    if delta_t == 0:
        delta_t = 1.0

    ts = events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / delta_t
    xs = events[:, 1].astype(np.int64)
    ys = events[:, 2].astype(np.int64)
    pols = events[:, 3]
    pols[pols == 0] = -1

    tis = ts.astype(np.int64)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid = tis < num_bins
    np.add.at(voxel_grid, xs[valid] + ys[valid] * width + tis[valid] * width * height, vals_left[valid])
    valid = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid] + ys[valid] * width + (tis[valid] + 1) * width * height, vals_right[valid])

    return voxel_grid.reshape(num_bins, height, width)


def voxel_norm(voxel):
    """Clip positive/negative values to their 2-98th percentiles and scale each to [0, 1] / [-1, 0]."""
    voxel_pos = voxel[voxel > 0]
    voxel_neg = voxel[voxel < 0]
    if len(voxel_pos) == 0 or len(voxel_neg) == 0:
        return voxel

    pos_2, pos_98 = np.percentile(voxel_pos, 2), np.percentile(voxel_pos, 98)
    neg_2, neg_98 = np.percentile(voxel_neg, 2), np.percentile(voxel_neg, 98)

    voxel[voxel > 0] = np.clip(voxel[voxel > 0], pos_2, pos_98)
    voxel[voxel < 0] = np.clip(voxel[voxel < 0], neg_2, neg_98)

    if pos_98 == pos_2:
        pos_98, pos_2 = np.max(voxel_pos), np.min(voxel_pos)
    if neg_98 == neg_2:
        neg_98, neg_2 = np.max(voxel_neg), np.min(voxel_neg)

    if pos_98 == pos_2:
        voxel[voxel > 0] = np.where(voxel[voxel > 0] > 0, 1, 0)
    else:
        voxel[voxel > 0] = (voxel[voxel > 0] - pos_2) / (pos_98 - pos_2)

    if neg_98 == neg_2:
        voxel[voxel < 0] = np.where(voxel[voxel < 0] < 0, -1, 0)
    else:
        voxel[voxel < 0] = -1 * (voxel[voxel < 0] - neg_2) / (neg_98 - neg_2)

    return voxel


def get_event_stacks(events, num_stacks):
    """
    Split events into `num_stacks` stacks that all end at the target time;
    stack i keeps the most recent N / 2^i events.
    """
    events[:, 0] = np.max(events[:, 0]) - events[:, 0]
    events_reversed = events[np.argsort(events[:, 0])]
    total_events = len(events_reversed)

    stacks = []
    for i in range(num_stacks):
        stacks.append(events_reversed[:total_events // (2 ** i) - 1, ...])
    return stacks


def load_events(event_folder, start_idx, end_idx, width, height, cfg):
    """Concatenates event files [start_idx, end_idx) into an [N, 4] (t, x, y, p) array, or None."""
    k_t, k_x, k_y, k_p = cfg["keys"]
    chunks = []
    for i in range(start_idx, end_idx):
        path = os.path.join(event_folder, f"{i:06d}.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path)
        ts = data[k_t].astype(np.float64)
        if len(ts) == 0:
            continue
        x = np.round(data[k_x].astype(np.float32) / cfg["coord_div"]).astype(np.int32)
        y = np.round(data[k_y].astype(np.float32) / cfg["coord_div"]).astype(np.int32)
        p = data[k_p].astype(np.int32)
        if np.min(p) == 0:
            p = np.where(p == 0, -1, p)
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        chunks.append(np.stack((ts, x, y, p), axis=1))

    if not chunks:
        return None
    events = np.concatenate(chunks, axis=0)
    return events[np.argsort(events[:, 0])]


def reverse_events(events):
    """Plays the stream backward in time: reversed timestamps and flipped polarity."""
    events = events.copy()
    events[:, 0] = np.max(events[:, 0]) - events[:, 0]
    events[:, 3] = -events[:, 3]
    return events[np.argsort(events[:, 0])]


def build_event_stack(event_folder, start_idx, end_idx, width, height, scale, cfg, num_stacks=NUM_STACKS,
                      reverse=False):
    """
    Multi-scale event stack for the events between frames `start_idx` and
    `end_idx`, rendered at the original resolution, quantized to 8 bit and
    LANCZOS-resized by `scale`. `num_stacks` < NUM_STACKS renders only the
    first (most accumulated) stacks.

    By default the stacks end at frame `end_idx` (prediction from `start_idx`).
    With `reverse=True` the stream is time-reversed, so the stacks end at frame
    `start_idx` (prediction backward from `end_idx`).

    Returns:
        stack: (num_stacks, H * scale, W * scale) tensor in [-1, 1]
        is_empty: True when there are no events (the ER-VAE input is then all zeros)
    """
    out_w, out_h = int(width * scale), int(height * scale)
    events = load_events(event_folder, start_idx, end_idx, width, height, cfg)
    if events is None:
        return torch.zeros((num_stacks, out_h, out_w), dtype=torch.float32), True
    if reverse:
        events = reverse_events(events)

    event_stacks = get_event_stacks(events, NUM_STACKS)
    stack = torch.zeros((num_stacks, out_h, out_w), dtype=torch.float32)
    for i in range(num_stacks):
        voxel = torch.zeros((height, width), dtype=torch.float32)
        if len(event_stacks[i]) > 0:
            v = events_to_voxel_grid(event_stacks[i], 1, width, height)
            voxel = torch.from_numpy(voxel_norm(v).squeeze()).float()

        # 8-bit quantization, then resize in PIL.
        v = torch.clamp(voxel, -1.0, 1.0)
        v = ((v + 1.0) / 2.0 * 255.0).byte().float() / 255.0 * 2.0 - 1.0
        img = Image.fromarray(((v + 1.0) / 2.0 * 255.0).byte().numpy())
        img = img.resize((out_w, out_h), Image.LANCZOS)
        stack[i] = torch.from_numpy(np.array(img).astype(np.float32) / 255.0 * 2.0 - 1.0)

    return stack, len(event_stacks[0]) == 0


def er_vae_input(stack, is_empty):
    """ER-VAE input: the most accumulated stack repeated to 3 channels, (1, 3, H, W)."""
    if is_empty:
        return torch.zeros((1, 3) + tuple(stack.shape[-2:]), dtype=torch.float32)
    return stack[0:1].repeat(3, 1, 1).unsqueeze(0)
