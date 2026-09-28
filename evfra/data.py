import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .events import DATASETS, build_event_stack, er_vae_input

UPSAMPLE_SCALE = 2
CROP_WIDTH = 512
CROP_HEIGHT = 320


def list_frames(image_folder):
    return sorted(f for f in os.listdir(image_folder) if f.endswith(".png"))


def list_sequences(dataset, root):
    """
    Test sequences as (name, image_folder, event_folder).
      bs_ergb: <root>/test/<seq>
      hs_ergb: <root>/{close,far}/test/<seq>   (named close/<seq>, far/<seq>)
      gopro  : <root>/test_converted/<seq>
    """
    cfg = DATASETS[dataset]
    if dataset == "bs_ergb":
        splits = [("", os.path.join(root, "test"))]
    elif dataset == "hs_ergb":
        splits = [(f"{c}/", os.path.join(root, c, "test")) for c in ("close", "far")]
    else:
        splits = [("", os.path.join(root, "test_converted"))]

    sequences = []
    for prefix, split_dir in splits:
        for seq in sorted(os.listdir(split_dir)):
            seq_dir = os.path.join(split_dir, seq)
            if os.path.isdir(seq_dir):
                sequences.append((prefix + seq,
                                  os.path.join(seq_dir, cfg["image_dir"]),
                                  os.path.join(seq_dir, cfg["event_dir"])))
    return sequences


def load_scaled_pair(image_folder, frames, anchor_idx, target_idx, resample):
    anchor = Image.open(os.path.join(image_folder, frames[anchor_idx])).convert("RGB")
    target = Image.open(os.path.join(image_folder, frames[target_idx])).convert("RGB")
    w, h = anchor.size
    size = (int(w * UPSAMPLE_SCALE), int(h * UPSAMPLE_SCALE))
    return anchor.resize(size, resample), target.resize(size, resample), (w, h)


def to_tensor(image):
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0


class BSERGBTrainDataset(Dataset):
    """
    Random (anchor, anchor + skip_frame) pairs from BS-ERGB sequences
    (<root>/<seq>/{images,events}), upsampled 2x and randomly cropped to 512x320.
    Samples are drawn at random, so the length only sets the epoch size.
    """

    def __init__(self, root, skip_frame=1, length=8447):
        self.root = root
        self.skip_frame = skip_frame
        self.length = length
        self.cfg = DATASETS["bs_ergb"]
        self.sequences = []
        for seq in sorted(os.listdir(root)):
            image_folder = os.path.join(root, seq, self.cfg["image_dir"])
            if os.path.isdir(image_folder) and len(list_frames(image_folder)) >= skip_frame + 2:
                self.sequences.append(seq)
        if not self.sequences:
            raise ValueError(f"No sequences with >= {skip_frame + 2} frames under {root}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seq_dir = os.path.join(self.root, random.choice(self.sequences))
        image_folder = os.path.join(seq_dir, self.cfg["image_dir"])
        event_folder = os.path.join(seq_dir, self.cfg["event_dir"])
        frames = list_frames(image_folder)

        anchor_idx = random.randint(0, len(frames) - self.skip_frame - 1)
        target_idx = anchor_idx + self.skip_frame
        anchor, target, (w, h) = load_scaled_pair(image_folder, frames, anchor_idx, target_idx, Image.LANCZOS)

        x = random.randint(0, anchor.size[0] - CROP_WIDTH)
        y = random.randint(0, anchor.size[1] - CROP_HEIGHT)
        box = (x, y, x + CROP_WIDTH, y + CROP_HEIGHT)

        with Image.open(os.path.join(image_folder, frames[0])) as first:
            ev_w, ev_h = first.size
        events, _ = build_event_stack(event_folder, anchor_idx, target_idx, ev_w, ev_h, UPSAMPLE_SCALE, self.cfg)

        return {
            "anchor_values": to_tensor(anchor.crop(box)),
            "target_values": to_tensor(target.crop(box)),
            "event_values": events[:, y:y + CROP_HEIGHT, x:x + CROP_WIDTH].contiguous(),
        }


class BSERGBInterpTrainDataset(BSERGBTrainDataset):
    """
    Interpolation triplets (t1, t, t2) from BS-ERGB: the anchors are `interval`
    frames apart with `interval` drawn from [2, max_interval], and the target
    lies strictly between them. The forward branch predicts t from t1 with the
    events of (t1, t]; the backward branch predicts t from t2 with the events of
    (t, t2] played backward.
    """

    def __init__(self, root, max_interval=2, length=8447):
        super().__init__(root, skip_frame=max_interval - 1, length=length)
        self.max_interval = max_interval

    def __getitem__(self, idx):
        seq_dir = os.path.join(self.root, random.choice(self.sequences))
        image_folder = os.path.join(seq_dir, self.cfg["image_dir"])
        event_folder = os.path.join(seq_dir, self.cfg["event_dir"])
        frames = list_frames(image_folder)

        interval = random.randint(2, min(self.max_interval, len(frames) - 1))
        t1 = random.randint(0, len(frames) - interval - 1)
        t2 = t1 + interval
        t = t1 + random.randint(1, interval - 1)

        anchor_fwd, target, (w, h) = load_scaled_pair(image_folder, frames, t1, t, Image.LANCZOS)
        anchor_bwd = Image.open(os.path.join(image_folder, frames[t2])).convert("RGB").resize(anchor_fwd.size, Image.LANCZOS)

        x = random.randint(0, anchor_fwd.size[0] - CROP_WIDTH)
        y = random.randint(0, anchor_fwd.size[1] - CROP_HEIGHT)
        box = (x, y, x + CROP_WIDTH, y + CROP_HEIGHT)

        events_fwd, _ = build_event_stack(event_folder, t1, t, w, h, UPSAMPLE_SCALE, self.cfg)
        events_bwd, _ = build_event_stack(event_folder, t, t2, w, h, UPSAMPLE_SCALE, self.cfg, reverse=True)
        crop = (slice(None), slice(y, y + CROP_HEIGHT), slice(x, x + CROP_WIDTH))

        return {
            "anchor_values": to_tensor(anchor_fwd.crop(box)),
            "anchor_values_bwd": to_tensor(anchor_bwd.crop(box)),
            "target_values": to_tensor(target.crop(box)),
            "event_values": events_fwd[crop].contiguous(),
            "event_values_bwd": events_bwd[crop].contiguous(),
            "d_fwd": float(t - t1),
            "d_bwd": float(t2 - t),
        }


class ERVAEDataset(Dataset):
    """
    Consecutive (anchor, target) pairs with the most accumulated event stack,
    for training the ER-VAE encoder. Random crops for training, center crops otherwise.
    """

    def __init__(self, root, width=CROP_WIDTH, height=CROP_HEIGHT, sequences=None, random_crop=True):
        self.width = width
        self.height = height
        self.random_crop = random_crop
        self.cfg = DATASETS["bs_ergb"]
        self.samples = []
        self.frames = {}
        for seq in sorted(os.listdir(root)):
            if sequences is not None and seq not in sequences:
                continue
            seq_dir = os.path.join(root, seq)
            image_folder = os.path.join(seq_dir, self.cfg["image_dir"])
            if not os.path.isdir(image_folder):
                continue
            frames = list_frames(image_folder)
            if len(frames) < 2:
                continue
            self.frames[seq_dir] = frames
            self.samples += [(seq_dir, i) for i in range(len(frames) - 1)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        seq_dir, anchor_idx = self.samples[index]
        frames = self.frames[seq_dir]
        image_folder = os.path.join(seq_dir, self.cfg["image_dir"])
        target_idx = anchor_idx + 1
        anchor, target, (w, h) = load_scaled_pair(image_folder, frames, anchor_idx, target_idx, Image.BICUBIC)

        W, H = anchor.size
        if self.random_crop:
            x = random.randint(0, max(0, W - self.width))
            y = random.randint(0, max(0, H - self.height))
        else:
            x = int((W - self.width) / 2)
            y = int((H - self.height) / 2)
        box = (x, y, x + self.width, y + self.height)

        stack, is_empty = build_event_stack(os.path.join(seq_dir, self.cfg["event_dir"]),
                                            anchor_idx, target_idx, w, h, UPSAMPLE_SCALE, self.cfg, num_stacks=1)
        event = er_vae_input(stack[:, y:y + self.height, x:x + self.width], is_empty)[0]

        return {
            "anchor_image": to_tensor(anchor.crop(box)),
            "target_image": to_tensor(target.crop(box)),
            "event_image": event,
        }
