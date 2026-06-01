from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import torch


@dataclass
class Env:
    """
    Container for a single environment: data, labels, and meta-information.

    Attributes
    ----------
    X : torch.Tensor
        Feature matrix (N, d).
    y : torch.Tensor
        Label vector (N, 1), binary {0,1} (float32).
    y_true : Optional[torch.Tensor]
        Ground-truth labels before label noise was applied.
    meta : Optional[Dict]
        Free-form dict (kind, generative parameters, split, etc.).
    """
    X: torch.Tensor
    y: torch.Tensor
    y_true: Optional[torch.Tensor] = None
    meta: Optional[Dict] = None
