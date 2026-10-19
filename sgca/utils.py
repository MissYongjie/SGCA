"""Config handling and reproducibility helpers."""
import copy
import os
import random

import numpy as np
import torch
import yaml


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str, overrides=()) -> dict:
    """Load a YAML config, resolving ``_base_`` inheritance and ``a.b=value`` overrides."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = _merge(load_config(os.path.join(os.path.dirname(path), base)), cfg)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_value(value)
    return cfg


def _parse_value(text: str):
    value = yaml.safe_load(text)
    if isinstance(value, str):  # YAML 1.1 reads e.g. "1e-4" as a string
        try:
            return float(value)
        except ValueError:
            pass
    return value


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
