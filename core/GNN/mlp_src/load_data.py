from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch


# ----------------------------
# 1) Load the email bodies
# ----------------------------
def convert_csv_to_json(csv_path: str) -> List[dict]:
    """
    Reads a CSV and returns a list of dicts where each row is an object.
    Also saves the data as JSON in the same directory as the CSV.
    """
    from pathlib import Path
    import json

    df = pd.read_csv(csv_path)
    df = df.replace({np.nan: None})
    records = df.to_dict(orient="records")

    csv_path_obj = Path(csv_path)
    json_path = csv_path_obj.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return records


def load_bodies_from_csv(
    csv_path: str,
    body_col: str = "body",
    url_col: str = "urls",
    only_with_urls: bool = True,
) -> List[str]:
    """
    Reads a CSV and returns the `body_col` column as a list of strings.
    Set `only_with_urls` to filter rows where `url_col` is non-zero.
    """
    df = pd.read_csv(csv_path)
    if body_col not in df.columns:
        raise KeyError(f"Column '{body_col}' not found. Columns: {list(df.columns)}")
    if only_with_urls:
        if url_col not in df.columns:
            raise KeyError(
                f"Column '{url_col}' not found. Columns: {list(df.columns)}"
            )
        url_series = df[url_col]
        has_url = ~(url_series.isna() | (url_series == 0) | (url_series == "0"))
        bodies = df.loc[has_url, body_col].fillna("").astype(str).tolist()
    else:
        bodies = df[body_col].fillna("").astype(str).tolist()
    return bodies


# ----------------------------
# 2) Load embeddings
# ----------------------------
def load_embeddings_npy(npy_path: str) -> np.ndarray:
    """
    Loads embeddings from a .npy file.
    Expected shape: (N, D)
    """
    E = np.load(npy_path)
    if E.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {E.shape}")
    return E

def load_model(path, model_class, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu")

    raw_config = checkpoint.get("config", {})

    # ✅ keep only args that URLEncoder.__init__ accepts
    allowed = {
        "embed_dim",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "max_len",
    }
    model_config = {k: v for k, v in raw_config.items() if k in allowed}

    model = model_class(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    optimizer = None
    opt_state = checkpoint.get("optimizer_state_dict", None)
    if opt_state is not None:
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.load_state_dict(opt_state)

    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    return model, optimizer, checkpoint.get("epoch", None), raw_config
