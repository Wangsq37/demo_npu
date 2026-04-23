from __future__ import annotations

import argparse
from pathlib import Path

from easyrec_npu.config import save_config
from easyrec_npu.data import create_loader, ensure_output_dir
from easyrec_npu.runtime import (
    add_common_args,
    build_model_and_optimizers,
    evaluate_model,
    load_runtime_components,
    save_checkpoint,
    train_one_epoch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Ascend demo train+eval"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, device = load_runtime_components(args)
    model_dir = ensure_output_dir(config["paths"]["model_dir"])
    save_config(config, model_dir / "pipeline.yaml")

    train_loader = create_loader(config, split="train")
    eval_loader = create_loader(config, split="eval")
    model, sparse_optimizer, dense_optimizer = build_model_and_optimizers(config, device)

    best_auc = float("-inf")
    last_step = 0
    for epoch in range(1, int(config["train"]["num_epochs"]) + 1):
        print(f"[train] epoch={epoch} device={device}")
        last_step, train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            sparse_optimizer=sparse_optimizer,
            dense_optimizer=dense_optimizer,
            log_interval=int(config["runtime"]["log_interval"]),
            max_steps=config["runtime"].get("max_train_steps"),
        )
        metrics = evaluate_model(
            model=model,
            loader=eval_loader,
            device=device,
            max_steps=config["runtime"].get("max_eval_steps"),
        )
        print(
            f"[eval] epoch={epoch} train_loss={train_loss:.6f} "
            f"eval_loss={metrics['loss']:.6f} auc={metrics['auc']:.6f}"
        )
        save_checkpoint(model_dir / "latest.ckpt", model, sparse_optimizer, dense_optimizer, epoch, last_step, best_auc)
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            save_checkpoint(model_dir / "best.ckpt", model, sparse_optimizer, dense_optimizer, epoch, last_step, best_auc)
            print(f"[train] new best auc={best_auc:.6f}")


if __name__ == "__main__":
    main()
