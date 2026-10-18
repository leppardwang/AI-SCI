from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "hour_of_day",
    "log_volume_kb",
    "log_latency_ms",
    "is_trusted_line",
    "dest_san_yi",
)


def build_features(df: pd.DataFrame) -> np.ndarray:
    """每事件 5 维确定性特征（不依赖全局 scaler，保证可复现）。

    - hour_of_day: 时间戳小时 / 24
    - log_volume_kb: log1p(volume_kb) / 12
    - log_latency_ms: log1p(latency_ms) / 8
    - is_trusted_line: 0/1
    - dest_san_yi: 核心三医平台 san_yi 置 1，否则 0
    """
    hour = (
        pd.to_datetime(df["timestamp"]).dt.hour.astype(float) / 24.0
    )
    log_volume = np.log1p(df["volume_kb"].astype(float)) / 12.0
    log_latency = np.log1p(df["latency_ms"].astype(float)) / 8.0
    trusted = df["is_trusted_line"].map(
        lambda v: 1.0
        if (isinstance(v, bool) or str(v).strip().lower() in {"true", "1", "yes"})
        else 0.0
    )
    dest = (df["dest_platform"].astype(str) == "san_yi").astype(float)
    return np.column_stack(
        [hour.to_numpy(), log_volume, log_latency, trusted.to_numpy(), dest]
    ).astype(np.float32)
