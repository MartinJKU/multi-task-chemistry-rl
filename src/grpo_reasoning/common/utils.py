"""Small shared utilities."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer


def set_seed(seed: int) -> None:
    """Set random seeds for supported libraries.

    Args:
        seed: Integer seed used for Python, NumPy, and PyTorch.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML mapping.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tokenizer(model_path: str | Path, **kwargs):
    """Load a tokenizer with compatibility fixes for known tokenizer configs.

    Args:
        model_path: Hugging Face model name or local checkpoint path.
        **kwargs: Extra keyword arguments forwarded to `AutoTokenizer.from_pretrained`.

    Returns:
        Loaded tokenizer.
    """
    try:
        return AutoTokenizer.from_pretrained(
            model_path,
            fix_mistral_regex=True,
            **kwargs,
        )
    except TypeError as exc:
        if "fix_mistral_regex" not in str(exc):
            raise
        return AutoTokenizer.from_pretrained(model_path, **kwargs)


def filter_moleculariq_kwargs(task: str, **kwargs) -> dict:
    """Filter task-specific keyword arguments for MolecularIQ.

    Args:
        task: Registered task name.
        **kwargs: Candidate task-specific keyword arguments.

    Returns:
        Non-`None` kwargs for `moleculariq`; otherwise an empty dictionary.
    """
    if task != "moleculariq":
        return {}
    result = {k: v for k, v in kwargs.items() if v is not None}
    if "properties" in result:
        result["properties"] = list(result["properties"])
    return result
