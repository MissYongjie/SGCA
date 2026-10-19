"""Loading of a bi-temporal multimodal image pair with its scribble and reference map."""
import os
from dataclasses import dataclass

import imageio.v2 as imageio
import numpy as np


@dataclass
class ChangePair:
    name: str
    t1: np.ndarray        # [H, W, C] float32 in [0, 1]
    t2: np.ndarray        # [H, W, C] float32 in [0, 1]
    gt: np.ndarray        # [H, W] uint8 {0, 1}
    scribble: np.ndarray  # [H, W] uint8 {0, 1}


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img[..., :3].astype(np.float32) / 255.0


def load_pair(root: str, name: str, scribble_file: str = "scribble.png") -> ChangePair:
    """Read ``{root}/{name}/{t1,t2,gt}.png`` and the scribble map."""
    d = os.path.join(root, name)
    read = lambda f: imageio.imread(os.path.join(d, f))
    t1, t2 = _to_rgb(read("t1.png")), _to_rgb(read("t2.png"))
    if t1.shape != t2.shape:
        raise ValueError(f"{name}: t1 {t1.shape} and t2 {t2.shape} must share the same size")
    gt = (read("gt.png") > 0).astype(np.uint8)
    scribble = (read(scribble_file) > 0).astype(np.uint8)
    gt = gt[..., 0] if gt.ndim == 3 else gt
    scribble = scribble[..., 0] if scribble.ndim == 3 else scribble
    return ChangePair(name, t1, t2, gt, scribble)
