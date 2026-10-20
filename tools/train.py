"""Train SGCA on one bi-temporal image pair and evaluate the resulting binary change map.

SGCA is optimised per scene (no pre-training): the network is fitted to the image
pair under scribble supervision while the scribbles are progressively expanded with
high-confidence pixels of the feature-distance map. The evolved pseudo change map
S^(E) is the final change mask; an Otsu-thresholded distance map is saved as well.

Example:
    python tools/train.py --config configs/hetdata1.yaml
    python tools/train.py --config configs/hetdata1.yaml --seed 1 --opts optim.epochs=200
"""
import argparse
import json
import os
import sys
import time

import imageio.v2 as imageio
import numpy as np
import torch
from skimage.filters import threshold_otsu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sgca import SGCA, SGCALoss, DynamicScribbleUpdater, error_map, evaluate, load_pair, scribble_weight_map  # noqa: E402
from sgca.utils import count_parameters, load_config, set_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="SGCA training / evaluation")
    p.add_argument("--config", required=True, help="path to a YAML config in configs/")
    p.add_argument("--data-root", default=None, help="override data.root")
    p.add_argument("--work-dir", default=None, help="output directory (default: work_dirs/<dataset>)")
    p.add_argument("--seed", type=int, default=None, help="override the config seed")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-intermediate", action="store_true",
                   help="save distance / pseudo-label / error maps at each scribble update")
    p.add_argument("--opts", nargs="*", default=[], help="config overrides, e.g. optim.lr=1e-4")
    return p.parse_args()


def to_tensor(img: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(img)
    t = t.permute(2, 0, 1) if t.ndim == 3 else t[None]
    return t[None].float().to(device)


def normalise(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.opts)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])
    device = torch.device(args.device)

    pair = load_pair(cfg["data"]["root"], cfg["data"]["name"], cfg["data"].get("scribble", "scribble.png"))
    work_dir = args.work_dir or os.path.join("work_dirs", pair.name)
    os.makedirs(work_dir, exist_ok=True)
    if args.save_intermediate:
        os.makedirs(os.path.join(work_dir, "train_vis"), exist_ok=True)

    m, o, s, lc = cfg["model"], cfg["optim"], cfg["scribble"], cfg["loss"]
    x1, x2 = to_tensor(pair.t1, device), to_tensor(pair.t2, device)
    model = SGCA(x1.shape[1], m["feature_dim"], m["heads"], m["downsample_ratio"],
                 m["gauss_kernel"], m["gauss_sigma"]).to(device)
    criterion = SGCALoss(lc["margin"], lc["w_con"], lc["w_sep"], lc["w_rec"], lc["w_cross"])
    optimizer = torch.optim.Adam([
        {"params": model.branch_t1.parameters()},
        {"params": model.branch_t2.parameters()},
        {"params": [model.smoothing.log_sigma], "lr": o["sigma_lr"]},
    ], lr=o["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=o["plateau_factor"], patience=o["plateau_patience"])
    updater = DynamicScribbleUpdater(
        pair.scribble, aoi_init_kernel=s["aoi_init_kernel"], aoi_step=s["aoi_step"],
        aoi_max_radius=s["aoi_max_radius"], target_ratio=s["target_ratio"],
        candidate_kernel=s["candidate_kernel"], candidate_dilation_iters=s["candidate_dilation_iters"],
        top_percentage=s["top_percentage"], background_ignore_radius=s["background_ignore_radius"])

    print(f"[SGCA] dataset={pair.name} size={pair.gt.shape} params={count_parameters(model):,} "
          f"init scribble coverage={updater.coverage:.4%} device={device}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    history = []
    for epoch in range(1, o["epochs"] + 1):
        model.train()
        scribble = updater.scribble
        weight_map = scribble_weight_map(scribble, s["weight_map_distance"]).to(device)
        scr_t = torch.from_numpy(scribble).float()[None, None].to(device)

        f1, f2, r1, r2, prior = model(x1, x2, scr_t, weight_map)
        loss, logs = criterion(f1, f2, r1, r2, x1, x2, prior)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=o["grad_clip"])
        optimizer.step()
        scheduler.step(loss.item())

        if epoch % s["update_interval"] == 0:
            dist = SGCA.distance(f1, f2).squeeze(0).detach().cpu().numpy()
            new_scribble = updater.step(dist)
            pm = evaluate(new_scribble, pair.gt)
            history.append({"epoch": epoch, "coverage": updater.coverage,
                            **{k: pm[k] for k in ("OA", "Kappa", "F1")}})
            print(f"  epoch {epoch:4d} | loss {logs['total']:.4f} (con {logs['con']:.4f} sep {logs['sep']:.4f} "
                  f"rec {logs['rec']:.4f}) | sigma {model.smoothing.sigma.item():.3f} | "
                  f"pseudo coverage {updater.coverage:.2%} F1 {pm['F1']:.3f}", flush=True)
            if args.save_intermediate:
                vis = os.path.join(work_dir, "train_vis")
                imageio.imwrite(f"{vis}/epoch{epoch:04d}_distance.png", (normalise(dist) * 255).astype(np.uint8))
                imageio.imwrite(f"{vis}/epoch{epoch:04d}_pseudo.png", new_scribble * 255)
                imageio.imwrite(f"{vis}/epoch{epoch:04d}_pseudo_error.png", error_map(new_scribble, pair.gt))

    # Output (paper protocol): the evolved pseudo change map S^(E) is the final change mask.
    # The Otsu-thresholded feature-distance map is saved as an auxiliary result.
    model.eval()
    with torch.no_grad():
        scribble = updater.scribble
        weight_map = scribble_weight_map(scribble, s["weight_map_distance"]).to(device)
        scr_t = torch.from_numpy(scribble).float()[None, None].to(device)
        f1, f2, *_ = model(x1, x2, scr_t, weight_map)
        dist = normalise(SGCA.distance(f1, f2).squeeze(0).cpu().numpy())
    pseudo = updater.scribble.astype(np.uint8)
    otsu = (dist > threshold_otsu(dist)).astype(np.uint8)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    metrics = evaluate(pseudo, pair.gt)
    metrics.update(dataset=pair.name, seed=cfg["seed"], time_s=round(elapsed, 2),
                   otsu={k: v for k, v in evaluate(otsu, pair.gt).items()}, history=history)
    print(f"[SGCA] {pair.name}: OA {metrics['OA']:.3f} | Kappa {metrics['Kappa']:.3f} | "
          f"F1 {metrics['F1']:.3f} | time {elapsed:.2f}s  (Otsu on distance map: F1 {metrics['otsu']['F1']:.3f})")

    imageio.imwrite(os.path.join(work_dir, "change_map.png"), pseudo * 255)
    imageio.imwrite(os.path.join(work_dir, "error_map.png"), error_map(pseudo, pair.gt))
    imageio.imwrite(os.path.join(work_dir, "distance_map.png"), (dist * 255).astype(np.uint8))
    imageio.imwrite(os.path.join(work_dir, "otsu_change_map.png"), otsu * 255)
    with open(os.path.join(work_dir, "metrics.json"), "w") as f:
        json.dump({**metrics, "config": cfg}, f, indent=2)


if __name__ == "__main__":
    main()
