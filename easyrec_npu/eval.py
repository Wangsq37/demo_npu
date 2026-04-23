from __future__ import annotations

import argparse

from easyrec_npu.data import create_loader
from easyrec_npu.runtime import (
    add_common_args,
    build_model_and_optimizers,
    evaluate_model,
    latest_checkpoint,
    load_checkpoint,
    load_runtime_components,
)


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Ascend demo eval"))
    parser.add_argument("--checkpoint_path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, device = load_runtime_components(args)
    checkpoint_path = args.checkpoint_path or str(latest_checkpoint(config["paths"]["model_dir"]))
    loader = create_loader(config, split="eval")
    model, _, _ = build_model_and_optimizers(config, device)
    load_checkpoint(checkpoint_path, model, device)
    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        max_steps=config["runtime"].get("max_eval_steps"),
    )
    print(f"[eval] checkpoint={checkpoint_path} loss={metrics['loss']:.6f} auc={metrics['auc']:.6f}")


if __name__ == "__main__":
    main()
