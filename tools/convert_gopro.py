"""
Builds the 240 fps GoPro test set used for evaluation from
  - GOPRO_Large_all  : all 240 fps sharp frames  (<frames_root>/<seq>/000001.png, ...)
  - EFNet GoPro h5   : events simulated over the whole sequence (<events_root>/<seq>.h5)

EFNet timestamps put 240 fps frame k at k/240 s (in ns): sharp image i is stamped as
the center of its 11-frame blur window, frame 5 + 11 i. The events of the 240 fps
interval (k, k+1) are therefore those with k/240 <= t < (k+1)/240. (In three test
sequences GoPro picked a different frame of the window as the sharp image, so the
pixel check below reports a mismatch there; the event timing is unaffected.)

Output (same layout as BS-ERGB):
  <output_root>/<seq>/images/000000.png ...
  <output_root>/<seq>/events/000000.npz ...   x, y, timestamp (ns), polarity
"""
import argparse
import os
import shutil

import h5py
import numpy as np
from PIL import Image

FRAME_NS = 1e9 / 240
WINDOW = 11


def check_alignment(f, frames, frames_dir, num_checks=3):
    """Max abs difference between EFNet sharp images and the center frames of their blur windows."""
    keys = sorted(f["sharp_images"].keys())
    assert len(frames) == WINDOW * len(keys), (len(frames), len(keys))
    worst = 0.0
    for i in np.linspace(0, len(keys) - 1, num_checks).astype(int):
        ds = f["sharp_images"][keys[i]]
        k = int(round(ds.attrs["timestamp"] / FRAME_NS))
        assert k == WINDOW * i + WINDOW // 2, (i, k)
        sharp = np.array(ds)
        if ds.attrs.get("type", "") == "color_bgr":
            sharp = sharp[..., ::-1]
        frame = np.array(Image.open(os.path.join(frames_dir, frames[k])).convert("RGB"))
        worst = max(worst, float(np.abs(sharp.astype(np.int16) - frame.astype(np.int16)).max()))
    return worst


def convert_sequence(seq, frames_root, events_root, output_root):
    frames_dir = os.path.join(frames_root, seq)
    frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    out_images = os.path.join(output_root, seq, "images")
    out_events = os.path.join(output_root, seq, "events")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_events, exist_ok=True)

    with h5py.File(os.path.join(events_root, f"{seq}.h5"), "r") as f:
        max_diff = check_alignment(f, frames, frames_dir)
        ts = f["events/ts"][:]
        xs = f["events/xs"][:]
        ys = f["events/ys"][:]
        ps = f["events/ps"][:]

    order = np.argsort(ts, kind="stable")
    ts, xs, ys, ps = ts[order], xs[order], ys[order], ps[order]
    bounds = np.searchsorted(ts, np.arange(len(frames)) * FRAME_NS, side="left")

    for k, name in enumerate(frames):
        shutil.copyfile(os.path.join(frames_dir, name), os.path.join(out_images, f"{k:06d}.png"))
        if k + 1 < len(frames):
            s, e = bounds[k], bounds[k + 1]
            np.savez(os.path.join(out_events, f"{k:06d}.npz"),
                     x=xs[s:e].astype(np.int16), y=ys[s:e].astype(np.int16),
                     timestamp=ts[s:e].astype(np.int64), polarity=(ps[s:e] > 0).astype(np.uint8))
    return len(frames), len(ts), max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_root", required=True, help="GOPRO_Large_all/test")
    parser.add_argument("--events_root", required=True, help="EFNet GoPro h5 test folder")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--sequences", nargs="*", default=None)
    args = parser.parse_args()

    sequences = args.sequences or sorted(os.listdir(args.frames_root))
    for seq in sequences:
        n_frames, n_events, max_diff = convert_sequence(seq, args.frames_root, args.events_root, args.output_root)
        print(f"{seq}: {n_frames} frames, {n_events} events, max |sharp - center frame| = {max_diff:.0f}", flush=True)


if __name__ == "__main__":
    main()
