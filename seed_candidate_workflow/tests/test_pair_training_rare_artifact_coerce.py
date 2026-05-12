from __future__ import annotations

import numpy as np
import pandas as pd


def test_from_rare_artifact_true_when_rarity_max_present() -> None:
    df = pd.DataFrame(
        [
            {"from_rare_artifact": False, "rare_artifact_rarity_max": 0.5},
            {"from_rare_artifact": False, "rare_artifact_rarity_max": np.nan},
        ]
    )
    rar = pd.to_numeric(df["rare_artifact_rarity_max"], errors="coerce")
    df["from_rare_artifact"] = df["from_rare_artifact"].astype(bool) | rar.notna()
    assert df["from_rare_artifact"].tolist() == [True, False]
