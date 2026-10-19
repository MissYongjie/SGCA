"""Scribble priors and the dynamic scribble learning (pseudo-label expansion) strategy."""
import cv2
import numpy as np
import torch


def _square(radius: int) -> np.ndarray:
    return np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)


def scribble_weight_map(scribble: np.ndarray, max_distance: float = 50.0) -> torch.Tensor:
    """Distance-decayed scribble weight map in [0, 1], returned as a [1, 1, H, W] tensor."""
    dist = cv2.distanceTransform(1 - scribble.astype(np.uint8), cv2.DIST_L2, 3)
    w = np.exp(-(dist ** 2) / (2 * (max_distance / 2) ** 2))
    w = w / w.max()
    return torch.from_numpy(w.astype(np.float32))[None, None]


class DynamicScribbleUpdater:
    """Progressively expands sparse scribbles with high-confidence pixels from the distance map.

    Keeps three masks: the current scribble ``S``, the area of interest (AOI) that
    grows by ``aoi_step`` per update but never beyond ``aoi_max_radius`` around the
    initial scribble, and the accumulated mask ``M_acc`` of every pixel added so far.
    """

    def __init__(self, init_scribble: np.ndarray, aoi_init_kernel: int = 33, aoi_step: int = 30,
                 aoi_max_radius: int = 350, target_ratio: float = 0.2, candidate_kernel: int = 33,
                 candidate_dilation_iters: int = 5, top_percentage: float = 0.01,
                 background_ignore_radius: int = 100):
        self.init_scribble = (init_scribble > 0).astype(np.uint8)
        self.scribble = self.init_scribble.copy()
        self.accumulated = self.init_scribble.copy()
        self.aoi = cv2.dilate(self.init_scribble, np.ones((aoi_init_kernel, aoi_init_kernel), np.uint8))
        self.aoi_limit = cv2.dilate(self.init_scribble, _square(aoi_max_radius))
        self.aoi_step = aoi_step
        self.target_ratio = target_ratio
        self.candidate_kernel = np.ones((candidate_kernel, candidate_kernel), np.uint8)
        self.candidate_dilation_iters = candidate_dilation_iters
        self.top_percentage = top_percentage
        self.background_kernel = _square(background_ignore_radius)

    @property
    def coverage(self) -> float:
        return float(self.scribble.mean())

    def _expand(self, distance: np.ndarray) -> np.ndarray:
        if self.target_ratio - self.coverage <= 0:
            return self.scribble
        near_scribble = cv2.dilate(self.scribble, self.candidate_kernel, iterations=self.candidate_dilation_iters)
        near_annotated = cv2.dilate(self.accumulated, self.background_kernel)
        candidates = (near_scribble > 0) & (self.scribble == 0) & (self.aoi > 0) & (near_annotated > 0)
        idx = np.flatnonzero(candidates)
        if idx.size == 0:
            return self.scribble
        n_new = max(1, int(self.top_percentage * idx.size))
        top = idx[np.argsort(distance.ravel()[idx])[::-1][:n_new]]
        new = self.scribble.copy()
        new.flat[top] = 1
        return new

    def step(self, distance: np.ndarray) -> np.ndarray:
        """One update: expand S inside the AOI, then grow M_acc and the AOI."""
        self.scribble = self._expand(distance)
        self.accumulated = np.logical_or(self.accumulated, self.scribble).astype(np.uint8)
        grown = cv2.dilate(self.scribble, _square(self.aoi_step))
        self.aoi = (np.logical_or(self.aoi, grown) & (self.aoi_limit > 0)).astype(np.uint8)
        return self.scribble
