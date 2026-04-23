from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_units: list[int], output_dim: int | None = None):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        if output_dim is not None:
            layers.append(nn.Linear(current_dim, output_dim))
            current_dim = output_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = current_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class DINAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_units: list[int]):
        super().__init__()
        self.attn = MLP(input_dim * 4, hidden_units, output_dim=1)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, embed_dim = keys.shape
        query_expand = query.unsqueeze(1).expand(batch_size, seq_len, embed_dim)
        attn_input = torch.cat(
            [query_expand, keys, query_expand - keys, query_expand * keys],
            dim=-1,
        )
        scores = self.attn(attn_input).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        return torch.sum(weights.unsqueeze(-1) * keys, dim=1)


class MultiTowerDIN(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        model_conf = config["model"]
        features = config["features"]
        embedding_dim = int(model_conf["embedding_dim"])

        self.id_features = features["id_features"]
        self.sequence_features = features["sequence_features"]
        self.query_features = features["query_features"]
        self.deep_feature_order = features["deep_feature_order"]

        self.embeddings = nn.ModuleDict()
        for feature in self.id_features:
            name = feature["name"]
            bucket_size = int(feature["bucket_size"]) + 1
            self.embeddings[name] = nn.Embedding(bucket_size, embedding_dim, padding_idx=0)
        self.embeddings["price_bucket"] = nn.Embedding(len(config["data"]["price_boundaries"]) + 2, embedding_dim, padding_idx=0)
        self.embeddings["pid_hash"] = nn.Embedding(int(features["pid_hash_bucket_size"]) + 1, embedding_dim, padding_idx=0)

        deep_input_dim = len(self.deep_feature_order) * embedding_dim
        self.deep_tower = MLP(deep_input_dim, list(model_conf["deep_hidden_units"]))

        sequence_dim = len(self.sequence_features) * embedding_dim
        self.din_attention = DINAttention(sequence_dim, list(model_conf["din_hidden_units"]))

        final_input_dim = self.deep_tower.output_dim + sequence_dim
        self.final_mlp = MLP(final_input_dim, list(model_conf["final_hidden_units"]), output_dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        deep_embeddings = [self.embeddings[name](batch[name]) for name in self.deep_feature_order]
        deep_input = torch.cat(deep_embeddings, dim=-1)
        deep_output = self.deep_tower(deep_input)

        query_embeddings = [self.embeddings[name](batch[name]) for name in self.query_features]
        query = torch.cat(query_embeddings, dim=-1)

        sequence_embeddings = [self.embeddings[feature["name"].replace("click_50_seq__", "")](batch[feature["name"]]) for feature in self.sequence_features]
        keys = torch.cat(sequence_embeddings, dim=-1)
        mask = batch["seq_mask"].bool()
        context = self.din_attention(query, keys, mask)

        logits = self.final_mlp(torch.cat([deep_output, context], dim=-1)).squeeze(-1)
        return logits


class ScriptWrapper(nn.Module):
    def __init__(self, model: MultiTowerDIN):
        super().__init__()
        self.model = model

    def forward(
        self,
        user_id: torch.Tensor,
        cms_segid: torch.Tensor,
        cms_group_id: torch.Tensor,
        final_gender_code: torch.Tensor,
        age_level: torch.Tensor,
        pvalue_level: torch.Tensor,
        shopping_level: torch.Tensor,
        occupation: torch.Tensor,
        new_user_class_level: torch.Tensor,
        adgroup_id: torch.Tensor,
        cate_id: torch.Tensor,
        campaign_id: torch.Tensor,
        customer: torch.Tensor,
        brand: torch.Tensor,
        price_bucket: torch.Tensor,
        pid_hash: torch.Tensor,
        click_50_seq__adgroup_id: torch.Tensor,
        click_50_seq__cate_id: torch.Tensor,
        click_50_seq__brand: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = {
            "user_id": user_id,
            "cms_segid": cms_segid,
            "cms_group_id": cms_group_id,
            "final_gender_code": final_gender_code,
            "age_level": age_level,
            "pvalue_level": pvalue_level,
            "shopping_level": shopping_level,
            "occupation": occupation,
            "new_user_class_level": new_user_class_level,
            "adgroup_id": adgroup_id,
            "cate_id": cate_id,
            "campaign_id": campaign_id,
            "customer": customer,
            "brand": brand,
            "price_bucket": price_bucket,
            "pid_hash": pid_hash,
            "click_50_seq__adgroup_id": click_50_seq__adgroup_id,
            "click_50_seq__cate_id": click_50_seq__cate_id,
            "click_50_seq__brand": click_50_seq__brand,
            "seq_mask": seq_mask,
        }
        return torch.sigmoid(self.model(batch))
