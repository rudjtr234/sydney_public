from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch.nn as nn

from .abmil import ABMILModel
from .acmil import ACMILModel
from .clam_sb import CLAMSBModel
from .dsmil import DSMILModel
from .transmil import TransMILModel


# Thyroid_Mutation_model_v2/src/models/factory.py 와 같은 기본값
MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "abmil": {
        "gate": True,
        "embed_dim": 512,
        "attn_dim": 384,
        "num_fc_layers": 1,
        "dropout": 0.25,
    },
    "clam_sb": {
        "gate": True,
        "embed_dim": 512,
        "attn_dim": 384,
        "num_fc_layers": 1,
        "dropout": 0.25,
    },
    "dsmil": {
        "embed_dim": 512,
        "num_fc_layers": 1,
        "dropout": 0.25,
        "temperature": 1.0,
    },
    "transmil": {
        "embed_dim": 512,
        "n_heads": 8,
        "n_layers": 2,
        "ffn_dim": 1024,
        "dropout": 0.1,
        "max_tokens": 2048,
    },
    "acmil": {
        "embed_dim": 512,
        "num_fc_layers": 1,
        "dropout": 0.25,
        "num_branches": 4,
        "topk_ratio": 0.1,
    },
}

MODEL_CLASSES = {
    "abmil": ABMILModel,
    "clam_sb": CLAMSBModel,
    "dsmil": DSMILModel,
    "transmil": TransMILModel,
    "acmil": ACMILModel,
}


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower().replace("-", "_")
    return {"clam": "clam_sb"}.get(normalized, normalized)


def get_available_models() -> List[str]:
    return sorted(MODEL_DEFAULTS.keys())


def create_mil_model(model_name: str,
                     in_dim: int,
                     num_classes: int = 2,
                     **overrides) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Returns (model, resolved_kwargs). Unknown override keys raise, so a typo in
    --model_kwargs fails loudly instead of being silently ignored.
    """
    name = normalize_model_name(model_name)
    if name not in MODEL_DEFAULTS:
        raise ValueError(f"Unsupported model '{model_name}'. Available models: {', '.join(get_available_models())}")

    kwargs = deepcopy(MODEL_DEFAULTS[name])
    unknown = sorted(set(overrides) - set(kwargs))
    if unknown:
        raise ValueError(f"Unknown kwargs for '{name}': {unknown}. Allowed: {sorted(kwargs)}")
    kwargs.update({k: v for k, v in overrides.items() if v is not None})

    model_kwargs = {"in_dim": in_dim, "num_classes": num_classes, **kwargs}
    return MODEL_CLASSES[name](**model_kwargs), model_kwargs
