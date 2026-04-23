from __future__ import annotations

import glob
import hashlib
import queue
import random
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def discover_files(pattern: str) -> list[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"没有匹配到 parquet 文件: {pattern}")
    return files


def _normalize_ids(values: np.ndarray, bucket_size: int) -> np.ndarray:
    values = np.nan_to_num(values, nan=0).astype(np.int64, copy=False)
    if bucket_size <= 1:
        return np.zeros_like(values, dtype=np.int64)
    mask = values > 0
    values = values.copy()
    values[mask] = ((values[mask] - 1) % (bucket_size - 1)) + 1
    values[~mask] = 0
    return values


def _hash_string(text: str, bucket_size: int) -> int:
    if not text:
        return 0
    if bucket_size <= 1:
        return 0
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(digest, byteorder="little", signed=False)
    return raw % (bucket_size - 1) + 1


@lru_cache(maxsize=200000)
def _parse_sequence_cached(text: str, max_len: int, bucket_size: int) -> tuple[int, ...]:
    if not text:
        return ()
    values = []
    for token in text.split("|")[-max_len:]:
        token = token.strip()
        if not token:
            continue
        try:
            raw = int(float(token))
        except ValueError:
            raw = 0
        if raw <= 0:
            values.append(0)
        else:
            values.append(((raw - 1) % (bucket_size - 1)) + 1 if bucket_size > 1 else 0)
    return tuple(values)


class PrefetchIterator:
    def __init__(self, iterator: Iterable[Any], maxsize: int):
        self._iterator = iter(iterator)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._sentinel = object()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            for item in self._iterator:
                self._queue.put(item)
        finally:
            self._queue.put(self._sentinel)

    def __iter__(self) -> "PrefetchIterator":
        return self

    def __next__(self) -> Any:
        item = self._queue.get()
        if item is self._sentinel:
            raise StopIteration
        return item


class ParquetBatchDataset(IterableDataset):
    def __init__(self, config: dict[str, Any], split: str):
        self.config = config
        self.split = split
        self.paths = config["paths"]
        self.data_conf = config["data"]
        self.runtime_conf = config["runtime"]
        self.features = config["features"]
        self.id_features = self.features["id_features"]
        self.sequence_features = self.features["sequence_features"]
        self.columns = [feature["name"] for feature in self.id_features]
        self.columns.extend(feature["name"] for feature in self.sequence_features)
        self.columns.extend(["price", "pid", self.features["label_field"]])
        self.columns = list(dict.fromkeys(self.columns))

    def _iter_file_batches(self, files: list[str]) -> Iterator[dict[str, Any]]:
        read_batch_size = int(self.data_conf["read_batch_size"])
        prefetch_batches = max(1, int(self.runtime_conf.get("prefetch_batches", 2)))

        def raw_iterator() -> Iterator[Any]:
            for file_path in files:
                parquet_file = pq.ParquetFile(file_path)
                for batch in parquet_file.iter_batches(
                    batch_size=read_batch_size,
                    columns=self.columns,
                    use_threads=True,
                ):
                    yield batch

        for record_batch in PrefetchIterator(raw_iterator(), prefetch_batches):
            yield self._encode_record_batch(record_batch)

    def _encode_record_batch(self, record_batch: Any) -> dict[str, Any]:
        frame = record_batch.to_pandas()
        batch_size = len(frame)
        encoded: dict[str, Any] = {}
        for feature in self.id_features:
            name = feature["name"]
            values = frame[name].fillna(0).to_numpy(dtype=np.float64, copy=False)
            encoded[name] = torch.from_numpy(_normalize_ids(values, int(feature["bucket_size"])))

        boundaries = np.asarray(self.data_conf["price_boundaries"], dtype=np.float32)
        prices = frame["price"].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        price_bucket = np.searchsorted(boundaries, prices, side="right").astype(np.int64) + 1
        price_bucket[prices <= 0] = 0
        encoded["price_bucket"] = torch.from_numpy(price_bucket)

        pid_values = frame["pid"].fillna("").astype(str).tolist()
        pid_bucket_size = int(self.features["pid_hash_bucket_size"])
        pid_hash = np.asarray([_hash_string(value, pid_bucket_size) for value in pid_values], dtype=np.int64)
        encoded["pid_hash"] = torch.from_numpy(pid_hash)

        max_len = int(self.data_conf["sequence_length"])
        seq_masks = np.zeros((batch_size, max_len), dtype=np.bool_)
        for feature in self.sequence_features:
            name = feature["name"]
            bucket_size = int(feature["bucket_size"])
            seq_tensor = np.zeros((batch_size, max_len), dtype=np.int64)
            values = frame[name].fillna("").astype(str).tolist()
            for row_index, text in enumerate(values):
                parsed = _parse_sequence_cached(text, max_len, bucket_size)
                if not parsed:
                    continue
                length = min(max_len, len(parsed))
                seq_tensor[row_index, :length] = np.asarray(parsed[:length], dtype=np.int64)
                seq_masks[row_index, :length] = True
            encoded[name] = torch.from_numpy(seq_tensor)
        encoded["seq_mask"] = torch.from_numpy(seq_masks)

        label_name = self.features["label_field"]
        labels = frame[label_name].fillna(0).to_numpy(dtype=np.float32, copy=False)
        encoded["label"] = torch.from_numpy(labels)
        encoded["reserved"] = {
            "user_id": frame["user_id"].fillna(0).astype("int64").tolist(),
            "adgroup_id": frame["adgroup_id"].fillna(0).astype("int64").tolist(),
            label_name: frame[label_name].fillna(0).astype("int64").tolist(),
        }
        return encoded

    def __iter__(self) -> Iterator[dict[str, Any]]:
        pattern_key = "train_input_path" if self.split == "train" else "eval_input_path"
        pattern = self.paths[pattern_key]
        files = discover_files(pattern)
        limit_key = "max_train_files" if self.split == "train" else "max_eval_files"
        limit = self.data_conf.get(limit_key)
        if limit:
            files = files[: int(limit)]

        worker = get_worker_info()
        worker_id = 0
        if worker is not None:
            worker_id = worker.id
            files = files[worker.id :: worker.num_workers]

        if self.split == "train" and self.data_conf.get("shuffle_files", True):
            rng = random.Random(self.runtime_conf["seed"] + worker_id)
            rng.shuffle(files)

        yield from self._iter_file_batches(files)


def create_loader(config: dict[str, Any], split: str) -> DataLoader:
    dataset = ParquetBatchDataset(config, split=split)
    num_workers = int(config["runtime"]["num_workers"])
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": None,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(config["runtime"].get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(config["runtime"].get("prefetch_factor", 2))
    return DataLoader(**loader_kwargs)


def ensure_output_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output
