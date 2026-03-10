from __future__ import annotations

import torch

from mlfd.nonlinear import _build_scheduler


def test_build_scheduler_supports_none_and_plateau() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    assert _build_scheduler("none", optimizer, factor=0.5, patience=3, min_lr=1e-5) is None
    scheduler = _build_scheduler("plateau", optimizer, factor=0.5, patience=3, min_lr=1e-5)
    assert scheduler is not None
