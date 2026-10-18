from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .anchor import ChainAnchorLog
from .chain import build_chain, collect_hits, verify_chain
from .crypto_signer import (
    export_public_keyring,
    load_node_signers,
    verifiers_from_keyring,
)
from .rules import HIGH_RISK_TYPES, RULES, apply_rules

# 仓内自带语料副本（2026-10-03 纳入 git，63,884,514 B / 63.9 MB）。优先使用
# 仓内路径，容器部署时可用 AUDIT_CSV 覆盖指向挂载卷。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_REPO_CSV = os.path.join(ROOT, "data",
                           "lecheng_security_logs_2024H1_pipeline10.csv")
_LEGACY_CSV = (
    "<external-archive>/datasets/lecheng_security_logs_2024H1.csv"
)
if "AUDIT_CSV" in os.environ:
    # 容器部署显式指定（挂载卷），不算静默回落
    DEFAULT_CSV = os.environ["AUDIT_CSV"]
elif os.path.isfile(IN_REPO_CSV):
    DEFAULT_CSV = IN_REPO_CSV
else:
    # AGENTS §9.4：数据不可获取时报错，禁止回退到另一份语料。
    # 恢复命令必须指向真正含该文件的提交：f0397ed 是引入本文件的提交
    # （7965070 里只有旧名 lecheng_security_logs_2024H1.csv，取不到本文件）。
    raise FileNotFoundError(
        f"仓内语料缺失：{IN_REPO_CSV}\n"
        f"  恢复：git show f0397ed:edge-audit-system/data/"
        f"lecheng_security_logs_2024H1_pipeline10.csv > {IN_REPO_CSV}\n"
        f"  或重新生成：python data/generator/gen_lecheng_logs.py "
        f"--out {IN_REPO_CSV} --no-clinical "
        f"--config data/generator/clinical_config.json\n"
        f"  容器部署请显式设置 AUDIT_CSV 指向挂载卷。"
    )

# 历史遗留的挂载路径，仅作说明保留，不再用于回落：
#   <external-archive>/datasets/lecheng_security_logs_2024H1.csv
#   —— 指向 git 历史版语料（sha c594baf5…），非当前口径，不得使用。
DEFAULT_SM2_KEY = os.environ.get(
    "AUDIT_SM2_KEY",
    "data/keys/edge_demo_sm2_private.hex",
)

# G4 改造（2026-09-30）：节点密钥目录。每节点一把独立 SM2 私钥，存于该目录下的
# node_<node_id>.hex；主链按凭证所属 hospital_id 选择签名者，中心方无法代签。
# 该目录下的公钥指纹清单是链尾外部锚定（G3）的信任锚材料。
SM2_KEYS_DIR = os.environ.get(
    "AUDIT_SM2_KEYS_DIR", os.path.dirname(DEFAULT_SM2_KEY)
)

# G3：链尾锚定公示日志路径。锚定不依赖外部存证服务，仅需本地追加写入。
ANCHOR_LOG_PATH = os.environ.get(
    "AUDIT_ANCHOR_LOG",
    os.path.join(os.path.dirname(SM2_KEYS_DIR), "anchors", "chain_anchors.jsonl"),
)


def keyring_fingerprint(keyring: dict) -> str:
    """节点公钥清单的聚合指纹（对 node_id 排序后拼接全部公钥再取 SM3）。

    单值口径便于与既有消费方对接；逐节点指纹见 keyring 本身。
    """
    from .crypto_signer import aggregate_keyring_fingerprint

    # 委托给 crypto_signer 的单一实现，避免与 chain.verify_chain 的清单校验
    # 出现两份口径而各自漂移（G5 改造）
    return aggregate_keyring_fingerprint(
        {nid: v["public_key"] for nid, v in keyring.items()})

HOSPITAL_NAMES = {
    "ruijin_hn": "Hospital 01",
    "shulan": "Hospital 02",
    "super_hn": "Hospital 03",
    "huaxi": "Hospital 04",
    "prov_people": "Hospital 05",
    "yiling": "Hospital 06",
    "hainan_med": "Hospital 07",
    "evergrande": "Hospital 08",
    "boai": "Hospital 09",
    "ciming": "Hospital 10",
    "jixin": "Hospital 11",
    "wenlan": "Hospital 12",
}



def union_of(rules, masks):
    """对给定规则集的掩码求并集（任一规则命中即算命中）。

    `rules` 可以是规则码序列（"R1"）或完整规则元组序列
    （("R1", "minimum_necessary", "最小必要违规")）——RULES 常量是后者，
    直接迭代会拿到元组，故此处统一取首元素。

    用于「唯一命中事件」这类需要跨规则去重的统计。注意 collect_hits 不去重，
    命中总次数 ≠ 唯一命中事件，见 REPRODUCIBILITY.md §4.3。
    """
    out = None
    for item in rules:
        rid = item[0] if isinstance(item, (tuple, list)) else item
        m = masks[rid]
        out = m if out is None else (out | m)
    return out.astype(bool) if out is not None else None

class AuditEngine:
    def __init__(self) -> None:
        self.ready = False
        self.error = None
        self.csv_path = None
        self.df = None
        self.masks = None
        self.p99 = None
        self.hits = None
        self.signer = None
        self.credentials = []
        self.chain_status = None
        self.node_chains = []
        self.node_signers = {}
        self.keyring = {}
        self.verifiers = {}
        self.anchor_log = None
        self.anchor_record = None
        self.summary = None
        self.hospitals = []
        self.rule_stats = []

    def load(self, csv_path: str | None = None) -> None:
        path = Path(csv_path or DEFAULT_CSV)
        self.csv_path = str(path)
        if not path.is_file():
            self.ready = False
            self.error = f"数据不可获取：找不到 CSV {path}。请提供真实语料路径（环境变量 AUDIT_CSV）。"
            raise FileNotFoundError(self.error)

        df = pd.read_csv(path)
        required = {
            "timestamp",
            "hospital_id",
            "volume_kb",
            "latency_ms",
            "risk_label",
            "risk_type",
            "action",
            "response_status",
            "is_trusted_line",
            "dest_platform",
        }
        missing = required - set(df.columns)
        if missing:
            self.ready = False
            self.error = f"CSV 缺字段：{sorted(missing)}"
            raise ValueError(self.error)

        # G4：每个节点一把独立私钥；主链按 hospital_id 选签名者
        result = apply_rules(df)
        masks = {rule_id: result[rule_id] for rule_id, _, _ in RULES}
        hits = collect_hits(df, masks)
        node_ids = sorted(hits["hospital_id"].astype(str).unique())
        node_signers = load_node_signers(SM2_KEYS_DIR, node_ids)
        keyring = export_public_keyring(node_signers)
        verifiers = verifiers_from_keyring(keyring)
        creds, chain_status = build_chain(hits, signer=None, signers_by_node=node_signers)
        chain_status.update(verify_chain(creds, verifiers=verifiers))
        node_chains = self._node_chains(hits, node_signers, verifiers)

        self.df = df
        self.masks = masks
        self.p99 = result["p99"]
        self.hits = hits
        self.signer = None
        self.node_signers = node_signers
        self.keyring = keyring
        self.verifiers = verifiers

        # G3：链尾锚定。核验通过后把链长与 Merkle 根公示到链式日志，
        # 后续任何对链尾的截断、尾部重写、追加或重放都会导致核验失败。
        try:
            self.anchor_log = ChainAnchorLog(ANCHOR_LOG_PATH)
            self.anchor_record = self.anchor_log.publish("main", creds)
        except OSError:
            self.anchor_log = None
            self.anchor_record = None
        self.credentials = creds
        self.chain_status = chain_status
        self.node_chains = node_chains
        self.rule_stats = self._rule_stats(masks, creds)
        self.hospitals = self._hospital_stats(df, masks)
        self.summary = self._summary(df, masks, creds)
        self.ready = True
        self.error = None

    def _node_chains(self, hits: pd.DataFrame, node_signers: dict, verifiers: dict) -> list[dict]:
        """按院拆分为 12 条 Edge 凭证链。

        对应论文 P2（数据不出域，凭证摘要上传）与 P4（链式可追溯）：
        每个医院节点各自维护本地凭证链，中心管理层仅聚合各节点摘要。
        """
        nodes = []
        for hospital_id, group in hits.groupby("hospital_id", sort=True):
            g = group.sort_values(["timestamp", "source_index"], kind="mergesort")
            nid = str(hospital_id)
            creds, status = build_chain(g, signer=node_signers[nid], node_id=nid)
            status.update(verify_chain(creds, verifiers=verifiers, sm2_sample=8))
            nodes.append(
                {
                    "node_id": str(hospital_id),
                    "hospital_name": HOSPITAL_NAMES.get(str(hospital_id), str(hospital_id)),
                    "events": int(len(g)),
                    "chain_length": status["length"],
                    "tip": status["tip"],
                    "valid": status["valid"],
                    "signatures_verified": status["signatures_verified"],
                    "signatures_checked": status["signatures_checked"],
                }
            )
        nodes.sort(key=lambda x: x["events"], reverse=True)
        return nodes

    def _rule_stats(self, masks: dict, creds: list[dict]) -> list[dict]:
        total_hits = len(creds)
        rows = []
        for rule_id, rule_key, rule_label in RULES:
            count = int(masks[rule_id].sum())
            rows.append(
                {
                    "rule_id": rule_id,
                    "rule_key": rule_key,
                    "rule_label": rule_label,
                    "hits": count,
                    "share": (count / total_hits) if total_hits else 0.0,
                }
            )
        return rows

    def _hospital_stats(self, df: pd.DataFrame, masks: dict) -> list[dict]:
        # 命中数 = **检出规则**之和。基线（BASELINES）不计入，
        # 否则 v2 下「每院命中数」会包含随机抽样行，指标失去意义。
        hit_count = sum(
            masks[rule_id].astype(int) for rule_id, _, _ in RULES
        )
        anomaly = df["risk_label"].astype(int) == 1
        high_risk = anomaly & df["risk_type"].astype(str).isin(HIGH_RISK_TYPES)
        tmp = pd.DataFrame(
            {
                "hospital_id": df["hospital_id"].astype(str),
                "events": 1,
                "anomaly": anomaly.astype(int),
                "high_risk": high_risk.astype(int),
                "rule_hits": hit_count,
            }
        )
        grouped = tmp.groupby("hospital_id", as_index=False).sum(numeric_only=True)
        n = int(len(df))
        rows = []
        for rec in grouped.to_dict("records"):
            hid = rec["hospital_id"]
            events = int(rec["events"])
            rows.append(
                {
                    "hospital_id": hid,
                    "hospital_name": HOSPITAL_NAMES.get(hid, hid),
                    "events": events,
                    "event_share": events / n if n else 0.0,
                    "anomaly": int(rec["anomaly"]),
                    "anomaly_rate": rec["anomaly"] / events if events else 0.0,
                    "high_risk": int(rec["high_risk"]),
                    "high_risk_rate": rec["high_risk"] / events if events else 0.0,
                    "rule_hits": int(rec["rule_hits"]),
                }
            )
        rows.sort(key=lambda x: x["events"], reverse=True)
        return rows

    def _summary(self, df: pd.DataFrame, masks: dict, creds: list[dict]) -> dict:
        n = int(len(df))
        anomaly = int((df["risk_label"].astype(int) == 1).sum())
        high_risk = int(
            (
                (df["risk_label"].astype(int) == 1)
                & df["risk_type"].astype(str).isin(HIGH_RISK_TYPES)
            ).sum()
        )
        r1_by_h = (
            df.loc[masks["R1"], "hospital_id"].astype(str).value_counts().to_dict()
            if masks["R1"].any()
            else {}
        )
        r1_top = max(r1_by_h.items(), key=lambda kv: kv[1]) if r1_by_h else ("", 0)
        r1_hits = int(masks["R1"].sum())
        sm2_checked = int(self.chain_status.get("signatures_checked", 0))
        sm2_verified = int(self.chain_status.get("signatures_verified", 0))
        return {
            "corpus": "lecheng_security_logs_2024H1_pipeline10.csv",
            "disclaimer": "合成语料方法学演示。数字由 80 万行 CSV 按附录 C 四条规则现场计算，不能当作真实医院审计实证。",
            "csv_path": self.csv_path,
            "events": n,
            "anomaly": anomaly,
            "anomaly_rate": anomaly / n if n else 0.0,
            "high_risk": high_risk,
            "high_risk_rate": high_risk / n if n else 0.0,
            "p99_volume_kb": self.p99,
            "rule_hits": len(creds),
            "unique_hit_events": int(
                union_of(RULES, masks).sum()
            ),
            "r1_top_hospital": r1_top[0],
            "r1_top_hospital_hits": int(r1_top[1]),
            "r1_top_hospital_share": (r1_top[1] / r1_hits) if r1_hits else 0.0,
            "chain_valid": bool(self.chain_status.get("valid", False)),
            "chain_hash_ok": bool(self.chain_status.get("hash_ok", False)),
            "signatures_checked": sm2_checked,
            "signatures_verified": sm2_verified,
            "sign_algorithm": "SM2",
            "node_chains": len(self.node_chains),
            "chain_tip": self.chain_status.get("tip"),
            "distinct_node_keys": len({s.private_key for s in self.node_signers.values()}),
            "keyring_fingerprints": {k: v["fingerprint"] for k, v in self.keyring.items()},
        }

    def query_credentials(
        self,
        rule_id: str | None = None,
        hospital_id: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict:
        items = self.credentials
        if rule_id:
            items = [c for c in items if c["rule_id"] == rule_id]
        if hospital_id:
            items = [c for c in items if c["hospital_id"] == hospital_id]
        total = len(items)
        slice_ = items[offset : offset + limit]
        return {"total": total, "offset": offset, "limit": limit, "items": slice_}


engine = AuditEngine()
