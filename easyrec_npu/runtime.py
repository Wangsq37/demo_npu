from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from easyrec_npu.config import apply_path_overrides, load_config, save_config
from easyrec_npu.data import create_loader, ensure_output_dir
from easyrec_npu.device import move_batch_to_device, resolve_device, setup_seed
from easyrec_npu.model import MultiTowerDIN, ScriptWrapper


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--train_input_path")
    parser.add_argument("--eval_input_path")
    parser.add_argument("--model_dir")
    parser.add_argument("--device")
    return parser


def load_runtime_components(args: Any) -> tuple[dict[str, Any], torch.device]:
    config = load_config(args.config_path)
    config = apply_path_overrides(config, args)
    device = resolve_device(config["runtime"]["device"])
    setup_seed(int(config["runtime"]["seed"]))
    return config, device


def build_model_and_optimizers(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[MultiTowerDIN, torch.optim.Optimizer, torch.optim.Optimizer]:
    model = MultiTowerDIN(config).to(device)
    embedding_params = []
    dense_params = []
    for name, param in model.named_parameters():
        if "embeddings" in name:
            embedding_params.append(param)
        else:
            dense_params.append(param)
    sparse_optimizer = torch.optim.Adagrad(embedding_params, lr=float(config["train"]["sparse_lr"]))
    dense_optimizer = torch.optim.Adam(
        dense_params,
        lr=float(config["train"]["dense_lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    return model, sparse_optimizer, dense_optimizer


def latest_checkpoint(model_dir: str | Path) -> Path:
    model_path = Path(model_dir)
    best = model_path / "best.ckpt"
    if best.exists():
        return best
    latest = model_path / "latest.ckpt"
    if latest.exists():
        return latest
    raise FileNotFoundError(f"未找到 checkpoint: {model_path}")


def save_checkpoint(
    checkpoint_path: str | Path,
    model: MultiTowerDIN,
    sparse_optimizer: torch.optim.Optimizer,
    dense_optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_auc: float,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "sparse_optimizer": sparse_optimizer.state_dict(),
        "dense_optimizer": dense_optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_auc": best_auc,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: MultiTowerDIN,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    return payload


def binary_auc(labels: list[float], scores: list[float]) -> float:
    if not labels:
        return float("nan")
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    pos_total = sum(1 for _, label in ordered if label >= 0.5)
    neg_total = len(ordered) - pos_total
    if pos_total == 0 or neg_total == 0:
        return float("nan")

    auc_pairs = 0.0
    neg_seen = 0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        pos_block = 0
        neg_block = 0
        while index < len(ordered) and ordered[index][0] == score:
            if ordered[index][1] >= 0.5:
                pos_block += 1
            else:
                neg_block += 1
            index += 1
        auc_pairs += pos_block * neg_seen + 0.5 * pos_block * neg_block
        neg_seen += neg_block
    return auc_pairs / (pos_total * neg_total)


def evaluate_model(
    model: MultiTowerDIN,
    loader: Any,
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, float]:
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    losses = []
    labels = []
    scores = []
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            if max_steps and step > max_steps:
                break
            reserved = batch.pop("reserved", None)
            batch = move_batch_to_device(batch, device)
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            probs = torch.sigmoid(logits)
            losses.append(float(loss.item()))
            labels.extend(batch["label"].detach().cpu().tolist())
            scores.extend(probs.detach().cpu().tolist())
            if reserved is not None:
                batch["reserved"] = reserved
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "auc": float(binary_auc(labels, scores)),
    }


def train_one_epoch(
    model: MultiTowerDIN,
    loader: Any,
    device: torch.device,
    sparse_optimizer: torch.optim.Optimizer,
    dense_optimizer: torch.optim.Optimizer,
    log_interval: int,
    max_steps: int | None,
) -> tuple[int, float]:
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    total_loss = 0.0
    total_steps = 0
    start_time = time.time()
    for step, batch in enumerate(loader, start=1):
        if max_steps and step > max_steps:
            break
        batch.pop("reserved", None)
        batch = move_batch_to_device(batch, device)
        sparse_optimizer.zero_grad()
        dense_optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch["label"])
        loss.backward()
        sparse_optimizer.step()
        dense_optimizer.step()
        total_loss += float(loss.item())
        total_steps = step
        if step % max(1, log_interval) == 0:
            elapsed = time.time() - start_time
            print(
                f"[train] step={step} loss={loss.item():.6f} "
                f"samples/sec={batch['label'].numel() * log_interval / max(elapsed, 1e-6):.2f}"
            )
            start_time = time.time()
    return total_steps, total_loss / max(1, total_steps)


def export_script_model(
    model: MultiTowerDIN,
    export_dir: str | Path,
    config: dict[str, Any],
) -> None:
    export_path = ensure_output_dir(export_dir)
    wrapper = ScriptWrapper(model.cpu()).eval()
    batch_size = 2
    seq_len = int(config["data"]["sequence_length"])
    sample_inputs = (
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.ones(batch_size, seq_len, dtype=torch.long),
        torch.ones(batch_size, seq_len, dtype=torch.long),
        torch.ones(batch_size, seq_len, dtype=torch.long),
        torch.ones(batch_size, seq_len, dtype=torch.bool),
    )
    try:
        scripted = torch.jit.trace(wrapper, sample_inputs, strict=False)
        scripted.save(str(export_path / "model.pt"))
    except Exception as exc:
        print(f"[export] TorchScript 导出失败，将仅保留 state_dict: {exc}")
    torch.save(model.state_dict(), export_path / "model_state.pt")
    save_config(config, export_path / "pipeline.yaml")


def write_predictions(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    file_name: str = "predict_result.csv",
) -> Path:
    output_path = ensure_output_dir(output_dir) / file_name
    if not rows:
        raise ValueError("没有可写出的预测结果")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path
