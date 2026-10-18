from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .chain import verify_chain
from .engine import (
    DEFAULT_CSV,
    DEFAULT_SM2_KEY,
    SM2_KEYS_DIR,
    engine,
    keyring_fingerprint,
)
from . import model_inference


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine.load(os.environ.get("AUDIT_CSV", DEFAULT_CSV))
    yield


app = FastAPI(title="乐城边缘审计（合成语料现算）", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_ready() -> None:
    if not engine.ready:
        raise HTTPException(status_code=503, detail=engine.error or "审计引擎未加载")


@app.get("/api/health")
def health():
    return {
        "ok": engine.ready,
        "csv_path": engine.csv_path,
        "error": engine.error,
    }


@app.get("/api/summary")
def summary():
    _require_ready()
    return engine.summary


@app.get("/api/rules")
def rules():
    _require_ready()
    return {
        "p99_volume_kb": engine.p99,
        "items": engine.rule_stats,
        "definitions": [
            {
                "rule_id": "R1",
                "logic": "risk_label=0 且 volume_kb > 正常样本 p99 且 action 不是 bulk_export/nightly_backup",
            },
            {
                "rule_id": "R2",
                "logic": "dest_platform 不属于核心三医平台 {san_yi} 且 is_trusted_line=False",
            },
            {
                "rule_id": "R3",
                "logic": "action=bulk_export 或 volume_kb>5000 或 risk_type=data_leak，且 is_trusted_line=False",
            },
            {
                "rule_id": "R4",
                "logic": "RandomState(42) 无放回抽取 0.2% 事件",
            },
        ],
    }


@app.get("/api/hospitals")
def hospitals():
    _require_ready()
    return {"items": engine.hospitals}


@app.get("/api/credentials")
def credentials(
    rule_id: str | None = None,
    hospital_id: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    _require_ready()
    return engine.query_credentials(rule_id, hospital_id, offset, limit)


@app.get("/api/chain")
def chain():
    _require_ready()
    status = dict(engine.chain_status)
    status["head"] = engine.credentials[0] if engine.credentials else None
    status["tail"] = engine.credentials[-1] if engine.credentials else None
    return status


@app.get("/api/nodes")
def nodes():
    """12 条按院 Edge 凭证链（P2 数据不出域、P4 链式可追溯）。"""
    _require_ready()
    return {
        "total": len(engine.node_chains),
        "items": engine.node_chains,
    }


@app.get("/api/signatures")
def signatures():
    _require_ready()
    # G4 改造：不再有单一全局密钥，改为每节点独立公钥 + 指纹清单。
    # 该清单本身可被哈希公示后作为外部锚定的信任锚材料。
    return {
        "algorithm": "SM2",
        "keys_dir": SM2_KEYS_DIR,
        "distinct_node_keys": len({s.private_key for s in engine.node_signers.values()}),
        "node_public_keys": {
            nid: {"public_key": v["public_key"], "fingerprint": v["fingerprint"]}
            for nid, v in engine.keyring.items()
        },
        "keyring_fingerprint": keyring_fingerprint(engine.keyring),
        "chain_signed": engine.chain_status.get("signatures_checked", 0) > 0,
        "signatures_checked": engine.chain_status.get("signatures_checked", 0),
        "signatures_verified": engine.chain_status.get("signatures_verified", 0),
        "unknown_node_count": engine.chain_status.get("unknown_node_count", 0),
    }


@app.get("/api/verify")
def verify():
    """独立复验入口：对全局凭证链做结构校验 + 哈希连续性 + SM2 抽样验签。

    G4 改造：按每条凭证的 node_id 选择对应节点的公钥验签，验签方无需持有私钥；
    公钥清单中不存在的节点按 fail-closed 处理（计入未通过）。
    """
    _require_ready()
    result = verify_chain(engine.credentials, verifiers=engine.verifiers)
    anchored, why = engine.anchor_log.verify("main", engine.credentials) \
        if engine.anchor_log else (None, "未启用链尾锚定")
    result["anchor_checked"] = anchored is not None
    if anchored is not None:
        result["anchor_ok"] = anchored
        result["anchor_reason"] = why
    return result


@app.get("/api/model/infer")
def model_infer():
    """本地 LSTM 风险推理（torch CPU，按院聚合真实模型输出）。"""
    _require_ready()
    try:
        return model_inference.infer_hospitals(engine.df)
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("AUDIT_PORT", "3001")),
        factory=False,
    )
