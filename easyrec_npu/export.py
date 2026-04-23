from __future__ import annotations

import argparse

from easyrec_npu.data import ensure_output_dir
from easyrec_npu.runtime import (
    add_common_args,
    build_model_and_optimizers,
    export_script_model,
    latest_checkpoint,
    load_checkpoint,
    load_runtime_components,
)


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Ascend demo export"))
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--export_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, device = load_runtime_components(args)
    checkpoint_path = args.checkpoint_path or str(latest_checkpoint(config["paths"]["model_dir"]))
    export_dir = ensure_output_dir(args.export_dir)
    model, _, _ = build_model_and_optimizers(config, device)
    load_checkpoint(checkpoint_path, model, device)
    export_script_model(model, export_dir, config)
    print(f"[export] checkpoint={checkpoint_path} export_dir={export_dir}")


if __name__ == "__main__":
    main()
