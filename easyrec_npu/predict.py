from __future__ import annotations

import argparse
from pathlib import Path

import torch

from easyrec_npu.data import create_loader
from easyrec_npu.runtime import (
    add_common_args,
    build_model_and_optimizers,
    latest_checkpoint,
    load_checkpoint,
    load_runtime_components,
    write_predictions,
)


SCRIPTED_INPUT_ORDER = [
    "user_id",
    "cms_segid",
    "cms_group_id",
    "final_gender_code",
    "age_level",
    "pvalue_level",
    "shopping_level",
    "occupation",
    "new_user_class_level",
    "adgroup_id",
    "cate_id",
    "campaign_id",
    "customer",
    "brand",
    "price_bucket",
    "pid_hash",
    "click_50_seq__adgroup_id",
    "click_50_seq__cate_id",
    "click_50_seq__brand",
    "seq_mask",
]


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Ascend demo predict"))
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--scripted_model_path", required=True)
    parser.add_argument("--predict_output_path", required=True)
    parser.add_argument("--reserved_columns", default="user_id,adgroup_id,clk")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, device = load_runtime_components(args)
    loader = create_loader(config, split="eval")
    export_dir = Path(args.scripted_model_path)
    scripted_path = export_dir / "model.pt"
    state_path = export_dir / "model_state.pt"

    scripted_model = None
    raw_model = None
    if scripted_path.exists():
        scripted_model = torch.jit.load(str(scripted_path), map_location="cpu").eval()
    else:
        checkpoint_path = args.checkpoint_path or str(latest_checkpoint(config["paths"]["model_dir"]))
        raw_model, _, _ = build_model_and_optimizers(config, device)
        if state_path.exists():
            raw_model.load_state_dict(torch.load(state_path, map_location=device))
        else:
            load_checkpoint(checkpoint_path, raw_model, device)
        raw_model.eval()

    reserved_columns = [column.strip() for column in args.reserved_columns.split(",") if column.strip()]
    rows = []
    max_steps = config["runtime"].get("max_eval_steps")
    for step, batch in enumerate(loader, start=1):
        if max_steps and step > max_steps:
            break
        reserved = batch.pop("reserved")
        if scripted_model is not None:
            inputs = [batch[name].cpu() for name in SCRIPTED_INPUT_ORDER]
            probs = scripted_model(*inputs)
            score_values = probs.detach().cpu().view(-1).tolist()
        else:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            probs = torch.sigmoid(raw_model(batch))
            score_values = probs.detach().cpu().view(-1).tolist()
        batch_size = len(score_values)
        for row_index in range(batch_size):
            row = {column: reserved[column][row_index] for column in reserved_columns if column in reserved}
            row["score"] = float(score_values[row_index])
            rows.append(row)

    output_path = write_predictions(rows, args.predict_output_path)
    print(f"[predict] output={output_path}")


if __name__ == "__main__":
    main()
