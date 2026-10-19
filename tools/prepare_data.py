"""Convert the original ``.mat`` benchmark files into the layout expected by SGCA.

Output layout (one folder per scene)::

    data/hetdata1/{t1.png, t2.png, gt.png, scribble.png}

The scribble used for the paper experiments is either the ``scribble`` field stored in the
dataset ``.mat`` or one entry of the companion ``*_scribbles.mat`` collection; the mapping
below records exactly which one was used for every scene.

Example:
    python tools/prepare_data.py --src /path/to/MCD --dst data
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np
import scipy.io as sio

# paper name -> (source .mat, scribble source); scribble source is None for the field stored in
# the dataset .mat, or the index into ``<stem>_scribbles.mat['all_data']['scribble']``.
DATASETS = {
    "hetdata1": ("hetdata/dataset#2.mat", None),
    "hetdata2": ("hetdata/dataset#5.mat", 2),
    "hetdata3": ("hetdata/dataset#6.mat", None),
    "hetdata4": ("hetdata/dataset#1.mat", None),
    "hetdata5": ("hetdata/dataset#3.mat", 2),
    "hetdata6": ("hetdata/dataset#4.mat", None),
}


def to_uint8(img: np.ndarray) -> np.ndarray:
    return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)


def convert(src_root: str, dst_root: str, name: str):
    mat_file, scribble_idx = DATASETS[name]
    data = sio.loadmat(os.path.join(src_root, mat_file))
    if scribble_idx is None:
        scribble = data["scribble"]
    else:
        stem = os.path.splitext(os.path.basename(mat_file))[0]
        bank = sio.loadmat(os.path.join(src_root, "scribblefeatures", f"{stem}_scribbles.mat"))
        scribble = bank["all_data"]["scribble"][0][scribble_idx]

    out = os.path.join(dst_root, name)
    os.makedirs(out, exist_ok=True)
    imageio.imwrite(os.path.join(out, "t1.png"), to_uint8(data["image_t1"]))
    imageio.imwrite(os.path.join(out, "t2.png"), to_uint8(data["image_t2"]))
    imageio.imwrite(os.path.join(out, "gt.png"), (np.squeeze(data["Ref_gt"]) > 0).astype(np.uint8) * 255)
    imageio.imwrite(os.path.join(out, "scribble.png"), (np.squeeze(scribble) > 0).astype(np.uint8) * 255)
    gt = np.squeeze(data["Ref_gt"]) > 0
    print(f"{name:9s} <- {mat_file:22s} size={gt.shape} changed={gt.mean():.2%} "
          f"scribble={(np.squeeze(scribble) > 0).mean():.3%}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", required=True, help="folder containing hetdata/ and scribblefeatures/")
    p.add_argument("--dst", default="data")
    p.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = p.parse_args()
    for name in args.datasets:
        convert(args.src, args.dst, name)


if __name__ == "__main__":
    main()
