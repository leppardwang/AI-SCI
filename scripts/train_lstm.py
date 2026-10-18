"""训练本地 LSTM 风险推理模型，产出真实可复现四件套。

产出（同一目录 results/lstm/）：
- model.pt            模型权重
- model_meta.json     模型结构元数据
- metrics_lstm.json   测试集真实指标（accuracy/precision/recall/f1/auc）
- params_train.json   训练超参与数据路径（可复现过程记录）

用法：
    python3 scripts/train_lstm.py --csv <路径> [--window 24] [--max-train-seqs 120000]
                                  [--epochs 3] [--batch-size 256] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backend.features import FEATURE_NAMES, build_features
from backend.lstm_model import RiskLSTM

# AGENTS §9.4：仓内语料缺失即报错，不静默回落到另一份语料。
# 恢复命令须指向引入本文件的 f0397ed（7965070 里只有旧名语料，取不到本文件）。
_IN_REPO_CSV = ROOT / "data" / "lecheng_security_logs_2024H1_pipeline10.csv"
DEFAULT_CSV = os.environ.get("AUDIT_CSV") or (
    str(_IN_REPO_CSV) if _IN_REPO_CSV.is_file()
    else f"仓内语料缺失：{_IN_REPO_CSV}（AGENTS §9.4 禁止回退；"
         f"请 git show f0397ed:edge-audit-system/data/"
         f"lecheng_security_logs_2024H1_pipeline10.csv 恢复）"
)
OUT_DIR = Path(os.environ.get("AUDIT_LSTM_RESULTS", ROOT / "results" / "lstm"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--max-train-seqs", type=int, default=120000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--pos-weight", type=float, default=0.0,
                   help="正类权重；0 表示自动 neg/pos")
    return p.parse_args()


def make_sequences(
    features: np.ndarray, labels: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """滑窗含当前事件：窗口 [t-window+1 .. t] → 预测 t 的风险标签。

    特征不含标签本身，属于合法的流式风险分类（当前事件 + 前文上下文）。
    """
    n = len(features)
    if n < window:
        return (
            np.zeros((0, window, features.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    x = np.stack([features[i : i + window] for i in range(n - window + 1)])
    y = labels[window - 1 :]
    return x.astype(np.float32), y.astype(np.int64)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not Path(args.csv).is_file():
        raise FileNotFoundError(f"数据不可获取：{args.csv}，请提供真实语料路径。")

    t0 = time.time()
    df = pd.read_csv(args.csv)
    labels = (df["risk_label"].astype(int) == 1).to_numpy()
    features = build_features(df)
    n = len(df)

    split = int(n * (1 - args.test_ratio))
    tr_f, tr_l = make_sequences(features[:split], labels[:split], args.window)
    te_f, te_l = make_sequences(features[split:], labels[split:], args.window)
    if len(tr_f) > args.max_train_seqs:
        rng = np.random.RandomState(args.seed)
        keep = rng.choice(len(tr_f), size=args.max_train_seqs, replace=False)
        tr_f, tr_l = tr_f[keep], tr_l[keep]

    train_ds = TensorDataset(
        torch.from_numpy(tr_f), torch.from_numpy(tr_l)
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    model = RiskLSTM(
        input_size=features.shape[1],
        hidden_size=args.hidden,
        num_layers=args.num_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = args.pos_weight
    if pos_weight <= 0:
        pos_total = int(tr_l.sum())
        neg_total = int(len(tr_l) - pos_total)
        pos_weight = neg_total / pos_total if pos_total else 1.0
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
    )

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        n_batch = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb.float())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batch += 1
        print(
            f"epoch {epoch}/{args.epochs} loss={total_loss / max(n_batch, 1):.6f}",
            flush=True,
        )

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(te_f))).numpy().ravel()
    pred = (probs >= 0.5).astype(int)
    tp = float(((pred == 1) & (te_l == 1)).sum())
    tn = float(((pred == 0) & (te_l == 0)).sum())
    fp = float(((pred == 1) & (te_l == 0)).sum())
    fn = float(((pred == 0) & (te_l == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / len(te_l) if len(te_l) else 0.0

    pos = te_l.sum()
    neg = len(te_l) - pos
    from sklearn.metrics import roc_auc_score

    auc = (
        float(roc_auc_score(te_l, probs))
        if pos and neg
        else 0.0
    )

    metrics = {
        "model": "RiskLSTM",
        "train_sequences": int(len(tr_f)),
        "test_windows": int(len(te_f)),
        "positive_test": int(pos),
        "negative_test": int(neg),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "warning": (
            "所有数字均由本脚本真实训练+评估产生（torch CPU, seed=42）。"
            "若文档引用的 LSTM 指标与此 JSON 不一致，以本 JSON 为准。"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "model.pt")
    (OUT_DIR / "model_meta.json").write_text(
        json.dumps(
            {
                "model": "RiskLSTM",
                "input_size": features.shape[1],
                "hidden_size": args.hidden,
                "num_layers": args.num_layers,
                "feature_names": list(FEATURE_NAMES),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "metrics_lstm.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "params_train.json").write_text(
        json.dumps(
            {
                "csv_path": str(Path(args.csv)),
                "window": args.window,
                "hidden": args.hidden,
                "num_layers": args.num_layers,
                "max_train_seqs": args.max_train_seqs,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "test_ratio": args.test_ratio,
                "pos_weight": pos_weight,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": "1.9.x",
                "train_seconds": round(time.time() - t0, 2),
                "command": "python3 scripts/train_lstm.py "
                f"--csv {args.csv} --window {args.window} --hidden {args.hidden} "
                f"--num-layers {args.num_layers} --max-train-seqs {args.max_train_seqs} "
                f"--epochs {args.epochs} --batch-size {args.batch_size} --seed {args.seed} "
                f"--test-ratio {args.test_ratio}",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"产物目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
