"""本地 LSTM 风险推理（torch CPU）。

真实训练产出：scripts/train_lstm.py → results/lstm/model.pt + metrics_lstm.json。
本模块仅加载已存在产物做推理；未训练时明确报错，禁止编造数字（§9 铁律）。
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = os.environ.get(
    "AUDIT_LSTM_RESULTS", "results/lstm"
)
WINDOW = int(os.environ.get("AUDIT_LSTM_WINDOW", "24"))
MAX_WINDOWS_PER_HOSPITAL = int(
    os.environ.get("AUDIT_LSTM_MAX_WINDOWS", "5000")
)
BATCH_SIZE = int(os.environ.get("AUDIT_LSTM_BATCH", "1024"))


class _LazyLSTM:
    def __init__(self) -> None:
        self._state = None
        self._error = None
        self._meta = None

    def load(self) -> "torch.nn.Module":
        from .features import FEATURE_NAMES
        from .lstm_model import RiskLSTM

        dir_path = Path(RESULTS_DIR)
        model_path = dir_path / "model.pt"
        meta_path = dir_path / "model_meta.json"
        if not (model_path.is_file() and meta_path.is_file()):
            self._error = (
                "本地 LSTM 模型未训练：缺少 results/lstm/model.pt 或 model_meta.json。"
                "先运行 python3 scripts/train_lstm.py 生成真实产物后再推理。"
            )
            raise FileNotFoundError(self._error)
        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
        import torch

        model = RiskLSTM(
            input_size=self._meta["input_size"],
            hidden_size=self._meta["hidden_size"],
            num_layers=self._meta["num_layers"],
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        self._state = model
        return model

    def meta(self) -> dict:
        return self._meta

    def error(self) -> str | None:
        return self._error


_loader = _LazyLSTM()


def available() -> bool:
    return (Path(RESULTS_DIR) / "model.pt").is_file() and (
        Path(RESULTS_DIR) / "model_meta.json"
    ).is_file()


def _windows(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """滑窗构建序列（t-W..t-1 → 预测 t 的标签）。"""
    if len(features) <= WINDOW:
        return (
            np.zeros((0, WINDOW, features.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    x = np.stack(
        [features[i : i + WINDOW] for i in range(len(features) - WINDOW)]
    )
    return x, np.zeros((len(x),), dtype=np.int64)


def infer_hospitals(df: pd.DataFrame) -> dict:
    """对每个医院按时间序滑动窗口，用本地 LSTM 推理风险概率并聚合。

    返回真实模型输出；无产物时 raise 并说明运行命令。
    """
    from .features import build_features

    if not available():
        raise RuntimeError(
            "本地 LSTM 风险推理无产物（§9 铁律：无产物数字禁止入文）。"
            "请先运行：python3 scripts/train_lstm.py"
        )
    import torch

    model = _loader.load()
    meta = _loader.meta()
    features = build_features(df)
    labels = (df["risk_label"].astype(int) == 1).to_numpy()
    timestamps = pd.to_datetime(df["timestamp"])

    items = []
    for hospital_id, group in df.groupby("hospital_id", sort=True):
        idx = group.index.to_numpy()
        order = np.argsort(timestamps.iloc[idx].to_numpy())
        idx = idx[order]
        if len(idx) < WINDOW:
            items.append(
                {
                    "hospital_id": str(hospital_id),
                    "events": int(len(idx)),
                    "reason": "事件数不足窗口",
                    "risk_mean": None,
                    "high_risk": None,
                }
            )
            continue
        seqs = features[idx]
        n_windows = len(seqs) - WINDOW + 1
        if n_windows < 1:
            items.append(
                {
                    "hospital_id": str(hospital_id),
                    "events": int(len(idx)),
                    "reason": "事件数不足窗口",
                    "risk_mean": None,
                    "high_risk": None,
                }
            )
            continue
        if n_windows > MAX_WINDOWS_PER_HOSPITAL:
            picks = np.linspace(
                0, n_windows - 1, MAX_WINDOWS_PER_HOSPITAL, dtype=np.int64
            )
        else:
            picks = np.arange(n_windows)
        x = np.stack([seqs[i : i + WINDOW] for i in picks])
        probs = []
        with torch.no_grad():
            for start in range(0, len(x), BATCH_SIZE):
                chunk = torch.from_numpy(x[start : start + BATCH_SIZE])
                probs.append(torch.sigmoid(model(chunk)).numpy().ravel())
        prob = np.concatenate(probs)
        pred = (prob >= 0.5).astype(int)
        true = labels[idx[WINDOW - 1 :]][picks]
        acc = float((pred == true).mean())
        pos = true.sum()
        pred_pos = pred.sum()
        tp = float(((pred == 1) & (true == 1)).sum())
        precision = tp / pred_pos if pred_pos else 0.0
        recall = tp / pos if pos else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        items.append(
            {
                "hospital_id": str(hospital_id),
                "events": int(len(idx)),
                "windows": int(len(x)),
                "windows_total": int(n_windows),
                "risk_mean": float(prob.mean()),
                "pred_positive_rate": float(pred.mean()),
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_anomaly_rate": float(true.mean()),
            }
        )

    weights_sum = _weights_size()
    return {
        "model": "RiskLSTM",
        "status": "ok",
        "window": WINDOW,
        "hidden_size": meta["hidden_size"],
        "num_layers": meta["num_layers"],
        "max_windows_per_hospital": MAX_WINDOWS_PER_HOSPITAL,
        "metrics_artifact": str(Path(RESULTS_DIR) / "metrics_lstm.json"),
        "model_artifact": str(Path(RESULTS_DIR) / "model.pt"),
        "weights_mb": weights_sum,
        "items": items,
    }


def _weights_size() -> float:
    if not available():
        return 0.0
    path = Path(RESULTS_DIR) / "model.pt"
    return round(path.stat().st_size / (1024 * 1024), 3)
