from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ID_FEATURES = [
    {"name": "user_id", "bucket_size": 1141730},
    {"name": "cms_segid", "bucket_size": 98},
    {"name": "cms_group_id", "bucket_size": 14},
    {"name": "final_gender_code", "bucket_size": 3},
    {"name": "age_level", "bucket_size": 8},
    {"name": "pvalue_level", "bucket_size": 5},
    {"name": "shopping_level", "bucket_size": 5},
    {"name": "occupation", "bucket_size": 3},
    {"name": "new_user_class_level", "bucket_size": 6},
    {"name": "adgroup_id", "bucket_size": 846812},
    {"name": "cate_id", "bucket_size": 12961},
    {"name": "campaign_id", "bucket_size": 423438},
    {"name": "customer", "bucket_size": 255877},
    {"name": "brand", "bucket_size": 461498},
]

SEQUENCE_FEATURES = [
    {"name": "click_50_seq__adgroup_id", "bucket_size": 846812},
    {"name": "click_50_seq__cate_id", "bucket_size": 12961},
    {"name": "click_50_seq__brand", "bucket_size": 461498},
]

QUERY_FEATURES = ["adgroup_id", "cate_id", "brand"]
DEEP_FEATURE_ORDER = [
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
]

PRICE_BOUNDARIES = [
    1.1,
    2.2,
    3.6,
    5.2,
    7.39,
    9.5,
    10.5,
    12.9,
    15,
    17.37,
    19,
    20,
    23.8,
    25.8,
    28,
    29.8,
    31.5,
    34,
    36,
    38,
    39,
    40,
    45,
    48,
    49,
    51.6,
    55.2,
    58,
    59,
    63.8,
    68,
    69,
    72,
    78,
    79,
    85,
    88,
    90,
    97.5,
    98,
    99,
    100,
    108,
    115,
    118,
    124,
    128,
    129,
    138,
    139,
    148,
    155,
    158,
    164,
    168,
    171.8,
    179,
    188,
    195,
    198,
    199,
    216,
    228,
    238,
    248,
    258,
    268,
    278,
    288,
    298,
    299,
    316,
    330,
    352,
    368,
    388,
    398,
    399,
    439,
    478,
    499,
    536,
    580,
    599,
    660,
    699,
    780,
    859,
    970,
    1080,
    1280,
    1480,
    1776,
    2188,
    2798,
    3680,
    5160,
    8720,
]


def default_config() -> dict[str, Any]:
    return {
        "paths": {
            "train_input_path": "../TorchEasyRec/data/taobao_data_train/*.parquet",
            "eval_input_path": "../TorchEasyRec/data/taobao_data_eval/*.parquet",
            "model_dir": "./experiments/multi_tower_din_taobao_local_npu",
        },
        "runtime": {
            "device": "auto",
            "seed": 2026,
            "amp": False,
            "num_workers": 4,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "prefetch_batches": 4,
            "log_interval": 20,
            "max_train_steps": 200,
            "max_eval_steps": 50,
        },
        "data": {
            "batch_size": 4096,
            "read_batch_size": 4096,
            "sequence_length": 100,
            "max_train_files": 8,
            "max_eval_files": 4,
            "shuffle_files": True,
            "price_boundaries": PRICE_BOUNDARIES,
        },
        "train": {
            "num_epochs": 1,
            "sparse_lr": 0.001,
            "dense_lr": 0.001,
            "weight_decay": 0.0,
        },
        "model": {
            "embedding_dim": 16,
            "deep_hidden_units": [512, 256, 128],
            "din_hidden_units": [256, 64],
            "final_hidden_units": [64],
        },
        "features": {
            "id_features": copy.deepcopy(ID_FEATURES),
            "sequence_features": copy.deepcopy(SEQUENCE_FEATURES),
            "query_features": copy.deepcopy(QUERY_FEATURES),
            "deep_feature_order": copy.deepcopy(DEEP_FEATURE_ORDER),
            "pid_hash_bucket_size": 20,
            "label_field": "clk",
        },
    }


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path) -> dict[str, Any]:
    config = default_config()
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    deep_update(config, user_config)
    return config


def apply_path_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    if getattr(args, "train_input_path", None):
        config["paths"]["train_input_path"] = args.train_input_path
    if getattr(args, "eval_input_path", None):
        config["paths"]["eval_input_path"] = args.eval_input_path
    if getattr(args, "model_dir", None):
        config["paths"]["model_dir"] = args.model_dir
    if getattr(args, "device", None):
        config["runtime"]["device"] = args.device
    return config


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
