<h1 align="center">SGCA: A Scribble-Guided Cross-Attention Framework<br>for Multimodal Remote Sensing Change Detection</h1>

<p align="center">
Yongjie Zheng, Hao Liu, Houcai Guo, Sicong Liu<sup>*</sup>, Lorenzo Bruzzone<sup>*</sup>
<br>
University of Trento &nbsp;·&nbsp; Tongji University
<br>
<em>ISPRS Journal of Photogrammetry and Remote Sensing</em>, vol. 235, pp. 416–427, 2026
</p>

<p align="center">
<a href="https://doi.org/10.1016/j.isprsjprs.2026.03.015"><img src="https://img.shields.io/badge/Paper-ISPRS%20JPRS-blue" alt="Paper"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
<img src="https://img.shields.io/badge/PyTorch-%E2%89%A51.12-orange" alt="PyTorch">
</p>

Official PyTorch implementation of **SGCA**, a weakly supervised framework for change detection between
heterogeneous sensors (optical, SAR, NDVI, …) that needs only a few freehand **scribbles** on changed
objects. This repository contains the code, data, scribbles and configurations needed to reproduce
the main heterogeneous-dataset results of the paper (Table 3 and Fig. 4).

<p align="center"><img src="assets/framework.png" width="95%" alt="SGCA framework"></p>

## Highlights

- **Scribble-guided learning.** A learnable Gaussian turns 1-pixel scribbles into a soft spatial prior
  that weights the losses and modulates the attention of a dual-branch encoder–decoder.
- **Composite objective.** Feature consistency on (likely) unchanged pixels, margin-based separation
  on (likely) changed pixels, and SSIM-based reconstruction for structure preservation.
- **Dynamic scribble learning.** Scribbles are progressively expanded with the most confident pixels of
  the feature-distance map inside a growing area of interest; the evolved pseudo map is the final
  change mask.
- **Per-scene optimisation.** No pre-training and no dense labels: each image pair is processed in
  seconds to about a minute on a single GPU.

## Installation

```bash
git clone https://github.com/MissYongjie/SGCA.git
cd SGCA
conda create -n sgca python=3.10 -y && conda activate sgca
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # pick the build for your CUDA
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.2 (CUDA 12.1) on an NVIDIA RTX 3090. The code also runs on CPU
(`--device cpu`), roughly 20–50× slower.

## Data

The six heterogeneous benchmarks are included in [`data/`](data/) as lossless PNGs, together with the
scribbles used in the paper (7.6 MB in total):

| Folder | Paper | Sensors | Size (H × W) | Event |
|---|---|---|---|---|
| `hetdata1` | HetData #1 | Spot / NDVI | 990 × 554 | Flooding, Gloucester (UK) |
| `hetdata2` | HetData #2 | Radarsat-2 / Google Earth | 343 × 291 | Embankment change, Yellow River (CN) |
| `hetdata3` | HetData #3 | QuickBird 2 / TerraSAR-X | 1034 × 582 | Flooding, Gloucester (UK) |
| `hetdata4` | HetData #4 | Landsat-5 / Google Earth | 300 × 412 | Lake expansion, Sardinia (IT) |
| `hetdata5` | HetData #5 | Pleiades / WorldView-2 | 500 × 500 | Construction, Toulouse (FR) |
| `hetdata6` | HetData #6 | Radarsat-2 / Google Earth | 593 × 921 | Building construction, Shuguang (CN) |

Each folder contains `t1.png`, `t2.png`, `gt.png` (evaluation only) and `scribble.png`. HetData #3 and #5
are used at the reduced resolution listed above. The image pairs and reference maps are public
benchmarks released by their original authors; please cite them when you use the data. See
[`data/README.md`](data/README.md) for details and for how to add your own image pairs.

## Quick start

Train and evaluate on one dataset:

```bash
python tools/train.py --config configs/hetdata1.yaml
```

```
[SGCA] dataset=hetdata1 size=(990, 554) params=187,655 init scribble coverage=0.1834% device=cuda
  epoch   10 | loss 2.5729 (...) | sigma 2.142 | pseudo coverage 0.26% F1 0.027
  ...
[SGCA] hetdata1: OA 0.980 | Kappa 0.909 | F1 0.921 | time 26.25s
```

Outputs are written to `results/<dataset>/`:

| File | Content |
|---|---|
| `change_map.png` | final binary change map (evolved pseudo map S<sup>(E)</sup>) |
| `error_map.png` | TP white · TN black · FP red · FN green |
| `distance_map.png` | normalised feature-distance map D |
| `otsu_change_map.png` | auxiliary result: Otsu threshold of D |
| `metrics.json` | OA / Kappa / F1, run time, per-update pseudo-map accuracy, full config |

Useful options: `--seed 1`, `--device cpu`, `--save-intermediate` (saves the distance map, pseudo map and
error map at every scribble update, as in Fig. 3 of the paper), and `--opts key=value ...` to override
any config entry, e.g. `--opts optim.epochs=200 scribble.top_percentage=0.1`.

## Reproducing the main results

```bash
bash scripts/run_all.sh            # all six datasets, then prints a comparison table
```

Results of the released configurations on an RTX 3090 (mean ± std over seeds 0, 1, 2), next to Table 3
of the paper:

| Dataset | OA | Kappa | F1 | F1 (seed 0 / 1 / 2) | Time (s) | Paper OA / Kappa / F1 |
|---|---|---|---|---|---|---|
| HetData #1 | 0.980 ± 0.000 | 0.909 ± 0.001 | 0.921 ± 0.001 | 0.920 / 0.922 / 0.919 | 26.4 | 0.980 / 0.910 / 0.921 |
| HetData #2 | 0.991 ± 0.001 | 0.860 ± 0.012 | 0.865 ± 0.011 | 0.879 / 0.851 / 0.864 | 4.0 | 0.989 / 0.840 / 0.845 |
| HetData #3 | 0.985 ± 0.000 | 0.872 ± 0.002 | 0.880 ± 0.002 | 0.881 / 0.878 / 0.882 | 29.6 | 0.985 / 0.878 / 0.885 |
| HetData #4 | 0.969 ± 0.004 | 0.739 ± 0.034 | 0.755 ± 0.032 | 0.777 / 0.779 / 0.711 | 6.1 | 0.979 / 0.807 / 0.818 |
| HetData #5 | 0.914 ± 0.008 | 0.677 ± 0.032 | 0.728 ± 0.027 | 0.746 / 0.749 / 0.690 | 12.1 | 0.931 / 0.737 / 0.778 |
| HetData #6 | 0.986 ± 0.001 | 0.839 ± 0.012 | 0.847 ± 0.011 | 0.855 / 0.855 / 0.831 | 43.4 | 0.987 / 0.843 / 0.850 |

Notes on reproducibility:

- **HetData #1, #2, #3 and #6** match the paper. On **HetData #4 and #5** the mean F1 is 0.063 and 0.050
  below the reported values (0.04 and 0.03 for the best seed). These two scenes are the most sensitive to
  the expansion schedule and to run-to-run variation: HetData #4 has a single short scribble (0.06 % of the
  pixels), and the scribble of HetData #5 crosses unchanged structures that differ strongly across sensors.
- **GPU non-determinism.** Some CUDA kernels used in training (e.g. the backward pass of bilinear
  up-sampling) accumulate gradients with atomic operations, so two runs with the same seed can differ
  slightly on GPU. The effect is negligible on most scenes but reaches about 0.01 F1 on HetData #4. In our
  tests, repeated CPU runs with the same seed gave identical results.
- **Time** is the end-to-end per-scene time (training plus inference) on an otherwise idle RTX 3090; the
  paper reports times on an RTX 3060.

### Configurations

All dataset configs inherit from [`configs/default.yaml`](configs/default.yaml) and override only a few
entries:

| Dataset | Overrides |
|---|---|
| HetData #1 | α = 0.05, τ = 0.12, 150 epochs |
| HetData #2 | α = 0.05, τ = 0.03, 150 epochs |
| HetData #3 | α = 0.05, τ = 0.055, 150 epochs |
| HetData #4 | α = 0.02, τ = 0.06, lr = 1e-4, σ<sub>init</sub> = 1, 200 epochs |
| HetData #5 | α = 0.3, τ = 0.16, lr = 1e-4, local candidate growth, 200 epochs |
| HetData #6 | α = 0.2, τ = 0.044, local candidate growth, 250 epochs |

The two parameters that matter most:

- **α (`scribble.top_percentage`)** is the fraction of candidate pixels added to the scribbles at each
  update (every Δ = 10 epochs). Expansion has to be fast enough to cover the changed objects before the
  consistency loss suppresses their feature distance.
- **τ (`scribble.target_ratio`)** stops the expansion once the pseudo map covers this fraction of the image.
  It acts as a prior on the proportion of changed pixels in the scene; the values above are slightly
  below the changed proportion of each benchmark. On new data, set it from a rough estimate of the
  changed area. `metrics.json` records the pseudo-map accuracy after every update, so you can also run once
  with a large τ and inspect the whole expansion curve.

"Local candidate growth" (`scribble.candidate_kernel: 7`, `candidate_dilation_iters: 1`) restricts new
pixels to the immediate neighbourhood of the current scribbles. It helps when a long scribble crosses
unchanged objects that are more dissimilar across sensors than the changed objects themselves.

## Qualitative results

Binary change maps of all compared methods (paper Fig. 4; TP white, TN black, FP red, FN green). The
individual maps are in [`assets/comparison/`](assets/comparison/), and
[`tools/make_comparison.py`](tools/make_comparison.py) rebuilds this figure, optionally with your own
results appended (`--extra Ours=results/{dataset}/error_map.png`).

<p align="center"><img src="assets/comparison.png" width="100%" alt="Qualitative comparison"></p>

Evolution of the feature-difference map, the pseudo map and its error map during training on HetData #1
(paper Fig. 3):

<p align="center"><img src="assets/training_process.jpg" width="95%" alt="Training process"></p>

## Repository structure

```
SGCA/
├── configs/                 # default.yaml + one config per dataset
├── data/                    # six heterogeneous benchmarks with scribbles (PNG)
├── sgca/
│   ├── models/sgca.py       # adaptive Gaussian smoothing, scribble-guided attention, dual-branch network
│   ├── losses.py            # consistency + separation + SSIM reconstruction
│   ├── scribble.py          # scribble prior and dynamic scribble learning (AOI, accumulated mask)
│   ├── data.py              # image-pair loading
│   ├── metrics.py           # OA, Kappa, F1 and error maps
│   └── utils.py             # config loading, seeding
├── tools/
│   ├── train.py             # train + evaluate one scene
│   ├── summarize.py         # collect results into a table
│   ├── make_comparison.py   # qualitative comparison figure
│   └── prepare_data.py      # rebuild data/ from the original .mat files
├── scripts/run_all.sh       # reproduce all six datasets
└── assets/                  # figures used in this README
```

## Implementation notes

For readers who want to map the code to the paper:

- The change map is the **evolved pseudo map** S<sup>(E)</sup>. The feature distance
  D(p) = ‖F<sup>(1)</sup>(p) − F<sup>(2)</sup>(p)‖<sub>2</sub> is computed on the encoder features, and its
  values rank the candidate pixels at every scribble update.
- The scribble-guided attention operates on an 8× average-pooled grid. The attention weights are
  re-weighted by a distance-decayed scribble weight map, and the attended features feed each branch's
  decoder, so the attention shapes the encoder through the reconstruction objective.
- The smoothed prior M produced by the learnable Gaussian weights L<sub>con</sub> by (1 − M) and L<sub>sep</sub> by M.
- The reconstruction loss combines 1 − SSIM(x̂<sub>t</sub>, x<sub>t</sub>) for each branch with an SSIM term
  between each reconstruction and the other acquisition. The latter is minimised, which keeps the two
  decoders modality-specific.
- The network has 187,655 trainable parameters for 3-channel inputs (single-band images are replicated
  to three channels).

## Citation

If you find this work useful, please cite:

```bibtex
@article{zheng2026sgca,
  title   = {{SGCA}: A Scribble-Guided Cross-Attention Framework for Multimodal Remote Sensing Change Detection},
  author  = {Zheng, Yongjie and Liu, Hao and Guo, Houcai and Liu, Sicong and Bruzzone, Lorenzo},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume  = {235},
  pages   = {416--427},
  year    = {2026},
  doi     = {10.1016/j.isprsjprs.2026.03.015}
}
```

## Acknowledgements

We thank the authors of the public heterogeneous change detection benchmarks used in this work, and the
authors of the compared methods (IRG-McS, NPSG, SCASC, GIR-MRF, SDIR, SRF, SG-SRF and SHIFT) for releasing
their code. This work was supported by the National Natural Science Foundation of China (Grant 42071324),
the Research Project of Tongji Architectural Design (Group) Co., Ltd (2023J-JB11), and the China
Scholarship Council.

## License

This project is released under the [MIT License](LICENSE). The datasets remain subject to the terms of
their original providers.
