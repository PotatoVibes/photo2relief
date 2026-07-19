"""DA3 inference worker — runs inside the isolated da3worker venv, one image per call.

Protocol (consumed by app.depth.DA3Backend):
    python worker.py <in_image> <out_npy> [--device cuda|cpu] [--model <hf_repo>]

Reads the image, runs DA3 monocular depth, resizes the native-resolution output back to
the input image size (mirroring the DA2 adapter), and writes RAW float32 depth (H, W)
to <out_npy>. No convention normalization here — the main app owns that logic so it
stays in one unit-tested place. Errors go to stderr with a nonzero exit code.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_image", help="Path to the input image (any PIL-readable format)")
    parser.add_argument("out_npy", help="Path to write raw float32 depth as .npy")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", default="depth-anything/DA3MONO-LARGE")
    args = parser.parse_args()

    # depth_anything_3 logs its pipeline steps at INFO on every call; keep stdout clean.
    logging.disable(logging.INFO)

    import numpy as np
    import torch
    from PIL import Image

    from depth_anything_3.api import DepthAnything3

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("warning: cuda requested but unavailable in worker; using cpu", file=sys.stderr)
        device = "cpu"

    with Image.open(args.in_image) as img:
        img = img.convert("RGB")
        width, height = img.width, img.height
        arr = np.asarray(img)

    model = DepthAnything3.from_pretrained(args.model).to(device)
    with torch.inference_mode():
        pred = model.inference([arr])

    depth = np.asarray(pred.depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        print(f"error: unexpected prediction shape {depth.shape}", file=sys.stderr)
        return 2

    # Resize native-resolution depth back to the input image size.
    t = torch.from_numpy(np.ascontiguousarray(depth))[None, None]
    t = torch.nn.functional.interpolate(
        t, size=(height, width), mode="bicubic", align_corners=False
    )
    np.save(args.out_npy, t.squeeze().numpy().astype(np.float32))
    return 0


if __name__ == "__main__":
    sys.exit(main())
