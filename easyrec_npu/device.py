from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_name: str) -> torch.device:
    target = device_name.lower()
    if target == "auto":
        try:
            import torch_npu

            npu_runtime = getattr(torch_npu, "npu", None) or getattr(torch, "npu", None)
            if npu_runtime is not None and npu_runtime.is_available():
                npu_runtime.set_device("npu:0")
                return torch.device("npu:0")
        except Exception:
            pass
        return torch.device("cpu")

    if target.startswith("npu"):
        import torch_npu

        npu_runtime = getattr(torch_npu, "npu", None) or getattr(torch, "npu", None)
        if npu_runtime is None:
            raise RuntimeError("torch_npu 已导入，但没有发现 npu runtime 入口")
        npu_runtime.set_device(target)
        return torch.device(target)

    return torch.device(target)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=False)
        else:
            moved[key] = value
    return moved
