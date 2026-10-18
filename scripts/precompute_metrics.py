#!/usr/bin/env python3
"""预计算审计指标，输出 JSON，供文档引用与验收核对。

四件套之一：可复现过程产物。运行：

    python3 scripts/precompute_metrics.py --csv <CSV路径> --out <输出.json>

输出结构：
- summary: 全局口径（事件/异常/高风险/p99/规则命中/唯一命中事件/R1院集中度）
- rules: 四条规则逐项命中
- hospitals: 12 院事件/异常/高风险/规则命中
- chain: 凭证链完整性
- meta: 运行环境、命令、时间
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import hashlib

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.chain import build_chain, collect_hits, verify_chain
from backend.engine import AuditEngine, keyring_fingerprint
from backend.rules import ruleset_info


def sha256_file(path: str) -> str:
    """语料指纹。只记 csv_path 不足以证明口径——同一路径可指向任意内容。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="80 万行源 CSV 路径")
    parser.add_argument("--out", required=True, help="输出 JSON 路径")
    args = parser.parse_args()

    engine = AuditEngine()
    t0 = time.time()
    engine.load(args.csv)
    elapsed = time.time() - t0

    status = dict(engine.chain_status)
    status["head"] = engine.credentials[0] if engine.credentials else None
    status["tail"] = engine.credentials[-1] if engine.credentials else None

    payload = {
        "summary": engine.summary,
        "rules": engine.rule_stats,
        "hospitals": engine.hospitals,
        "nodes": engine.node_chains,
        "signatures": {
            "algorithm": "SM2",
            "private_key_path": str(
                Path(engine.csv_path).parent / "keys"
                / "edge_demo_sm2_private.hex"),
            "distinct_node_keys": len({s.private_key for s in engine.node_signers.values()}),
            "public_key_fingerprint": keyring_fingerprint(engine.keyring),
            "node_keyring_fingerprints": {k: v["fingerprint"] for k, v in engine.keyring.items()},
            "signatures_checked": engine.chain_status.get("signatures_checked", 0),
            "signatures_verified": engine.chain_status.get("signatures_verified", 0),
        },
        "chain": status,
        "meta": {
            "csv_path": engine.csv_path,
            "corpus_sha256": sha256_file(engine.csv_path),
            "ruleset": ruleset_info(),
            "events": int(len(engine.df)),
            "load_seconds": round(elapsed, 3),
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "seed": 42,
            "note": "合成语料方法学演示，非真实医院审计实证。",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {out}")
    print(f"rule_hits={payload['summary']['rule_hits']} "
          f"chain_valid={payload['summary']['chain_valid']} "
          f"load_seconds={elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
