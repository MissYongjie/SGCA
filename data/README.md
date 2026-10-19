# Data

Each scene is stored as one folder of lossless PNG files:

```
data/hetdataX/
├── t1.png         # pre-event image  (H x W x 3, uint8)
├── t2.png         # post-event image (H x W x 3, uint8; single-band images are replicated to 3 channels)
├── gt.png         # reference change map (0 = unchanged, 255 = changed), used for evaluation only
└── scribble.png   # sparse change scribbles used for training (0 / 255)
```

| Folder | Paper name | Sensors | Size (H x W) | Event |
|---|---|---|---|---|
| `hetdata1` | HetData #1 | Spot / NDVI | 990 x 554 | Flooding, Gloucester (UK) |
| `hetdata2` | HetData #2 | Radarsat-2 / Google Earth | 343 x 291 | Embankment change, Yellow River (CN) |
| `hetdata3` | HetData #3 | QuickBird 2 / TerraSAR-X | 1034 x 582 | Flooding, Gloucester (UK) |
| `hetdata4` | HetData #4 | Landsat-5 / Google Earth | 300 x 412 | Lake expansion, Sardinia (IT) |
| `hetdata5` | HetData #5 | Pleiades / WorldView-2 | 500 x 500 | Construction, Toulouse (FR) |
| `hetdata6` | HetData #6 | Radarsat-2 / Google Earth | 593 x 921 | Building construction, Shuguang (CN) |

HetData #3 and #5 are used at the reduced resolution listed above (the original scenes are
4135 x 2325 and 2000 x 2000 pixels).

The image pairs and reference maps are public benchmarks released by their original authors;
please cite the corresponding sources when using them. The scribbles are provided by us.

## Using your own data

Put a new folder with the four files above under `data/`, copy one of the configs in `configs/`,
and set `data.name` to the folder name. Scribbles only need to mark *changed* pixels: a few
freehand strokes (about 0.1–2% of the image) drawn inside representative change regions are
enough. Thick strokes can be thinned to 1-pixel skeletons with `skimage.morphology.skeletonize`.

## Re-generating from the original `.mat` files

`tools/prepare_data.py` rebuilds this folder from the original MATLAB files and records which
scribble was used for each scene:

```bash
python tools/prepare_data.py --src /path/to/MCD --dst data
```
