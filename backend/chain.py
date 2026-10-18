from __future__ import annotations
import os
import sys

import hashlib

import pandas as pd

from .crypto_signer import SM2Signer
from .rules import RULES


GENESIS = "0" * 64


def collect_hits(df: pd.DataFrame, masks: dict) -> pd.DataFrame:
    frames = []
    for rule_id, rule_key, rule_label in RULES:
        mask = masks[rule_id]
        if not mask.any():
            continue
        sub = df.loc[
            mask,
            [
                "timestamp",
                "hospital_id",
                "volume_kb",
                "latency_ms",
                "action",
                "risk_type",
                "risk_label",
                "dest_platform",
                "is_trusted_line",
                "response_status",
            ],
        ].copy()
        sub["rule_id"] = rule_id
        sub["rule_key"] = rule_key
        sub["rule_label"] = rule_label
        sub["source_index"] = df.index[mask].astype(int)
        frames.append(sub)
    if not frames:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "hospital_id",
                "volume_kb",
                "latency_ms",
                "action",
                "risk_type",
                "risk_label",
                "dest_platform",
                "is_trusted_line",
                "response_status",
                "rule_id",
                "rule_key",
                "rule_label",
                "source_index",
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(
        ["timestamp", "source_index", "rule_id"], kind="mergesort"
    ).reset_index(drop=True)


# G2 改造（2026-09-30）：元数据字段纳入摘要与签名覆盖范围。
# 原实现的签名对象仅含九个载荷字段，rule_key / rule_label / risk_type / latency_ms
# 等元数据不在摘要内，修改它们不改变 hash 因而不被任何校验发现（审计 A3：检出率 0%）。
# 现将这四项一并纳入摘要，使元数据篡改同样触发摘要失配与验签失败。
# 说明：node_id 由链内 hospital_id 决定、seq 由位置自明，二者通过 verify_chain 的
# 结构一致性校验约束，不进入摘要以避免位置依赖。
_META_FIELDS = ("rule_key", "rule_label", "risk_type", "latency_ms")

# 摘要与签名的字段清单，build_chain 与 verify_chain 必须共用同一份定义，
# 否则二者拼接顺序一旦漂移就会互相判定为不一致。
#
# 2026-10-03 新增 risk_label（实测依据 chain_field_coverage_results.json）：
#   该列原不在摘要内，但 R1 两处都要读它——既是必要条件 `risk_label == 0`，
#   又被 compute_p99 用来筛正常事件以算 p99 阈值。实测篡改该列后
#   verify_chain 仍判 valid，而 R1 命中数由 562 降至 312、总命中
#   1,183 → 933——即「不可检出且影响审计结果」的 material gap。
#
#   现覆盖语料 10 列中的 9 列：timestamp / hospital_id / volume_kb /
#   latency_ms / risk_label / risk_type / action / is_trusted_line /
#   dest_platform；另加链内元数据 rule_id / rule_key / rule_label /
#   source_index，合计 13 字段（+ prev_hash 共 14 段）。
#   余下 response_status 既不在摘要内、凭证也不携带，且无任何规则读取
#   （实测篡改后命中数不变，1,183 → 1,183），属 silent gap 而非 material
#   gap。覆盖范围声明见 REPRODUCIBILITY.md §4.6。
_PAYLOAD_FIELDS = (
    "timestamp",
    "hospital_id",
    "rule_id",
    "source_index",
    "volume_kb",
    "action",
    "dest_platform",
    "is_trusted_line",
) + _META_FIELDS + ("risk_label",)

# is_trusted_line 的位置。**不得硬编码下标**——_PAYLOAD_FIELDS 一旦在它之前
# 插入字段，parts[7] 会静默作用到别的字段上：摘要照算、验签照过，语义全错。
_TRUSTED_INDEX = _PAYLOAD_FIELDS.index("is_trusted_line")


def _payload_from_values(values, prev_hash: str) -> str:
    parts = [str(values[f]) for f in _PAYLOAD_FIELDS]
    # is_trusted_line 统一规范化为布尔字符串，避免 True/1/"True" 混用
    parts[_TRUSTED_INDEX] = str(bool(values["is_trusted_line"]))
    parts.append(prev_hash)
    return "|".join(parts)


def _payload(row, prev_hash: str) -> str:
    return _payload_from_values({f: getattr(row, f, "") for f in _PAYLOAD_FIELDS}, prev_hash)


def _payload_from_item(item: dict, prev_hash: str) -> str:
    return _payload_from_values(item, prev_hash)


def build_chain(
    hits: pd.DataFrame,
    signer: SM2Signer | None = None,
    node_id: str | None = None,
    signers_by_node: dict | None = None,
) -> tuple[list[dict], dict]:
    """构建凭证链。

    - 哈希连续性：genesis 全 0，每条凭证引用前序哈希（P4 可追溯）
    - SM2 签名：signer 非空时对凭证哈希签名（P3 防篡改 / 不可抵赖）
    - G4 改造：传入 signers_by_node 时，每条凭证由其 hospital_id 对应节点的私钥
      签名，中心方无法代签，节点级来源真实性成立。
    返回 (credentials, 链状态)。
    """
    prev = GENESIS
    creds = []
    for seq, row in enumerate(hits.itertuples(index=False), start=1):
        payload = _payload(row, prev)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        sig_signer = signer
        if signers_by_node is not None:
            sig_signer = signers_by_node.get(str(row.hospital_id), signer)
        signature = sig_signer.sign(digest) if sig_signer else ""
        creds.append(
            {
                "seq": seq,
                "node_id": node_id or row.hospital_id,
                "timestamp": str(row.timestamp),
                "hospital_id": str(row.hospital_id),
                "rule_id": row.rule_id,
                "rule_key": row.rule_key,
                "rule_label": row.rule_label,
                "source_index": int(row.source_index),
                "volume_kb": float(row.volume_kb),
                "latency_ms": float(row.latency_ms),
                "action": str(row.action),
                "risk_type": str(row.risk_type),
                "dest_platform": str(row.dest_platform),
                "is_trusted_line": bool(row.is_trusted_line),
                "risk_label": int(row.risk_label),
                "prev_hash": prev,
                "hash": digest,
                "signature": signature,
            }
        )
        prev = digest
    status = {
        "valid": True,
        "length": len(creds),
        "genesis": GENESIS,
        "tip": creds[-1]["hash"] if creds else GENESIS,
        "node_id": node_id,
    }
    return creds, status



# ---------------------------------------------------------------------------
# 验签抽样策略（2026-10-03）
#
# "uniform"：全局等间隔，即 `step = n / sp; {int(i*step) for i in range(sp)}`。
#   这是 §9.2.3 全部已入库产物所用的口径，**逐字保持不变**，以便旧产物可复现。
#
# "stratified"：按节点分层。每节点保底 1 点，余下按各节点链上占比用最大余数法
#   分配，节点内部再等间隔取点。点数严格等于 sp。
#   动机：uniform 在负载不均衡下会让小节点零覆盖（实测 3/12 节点），
#   而"某节点从未被验签"在verify_chain 的返回里不可见。
# ---------------------------------------------------------------------------

SAMPLING_MODES = ("uniform", "stratified")
DEFAULT_SAMPLE_MODE = "uniform"


def active_sample_mode() -> str:
    name = os.environ.get("AUDIT_SAMPLE_MODE",
                          DEFAULT_SAMPLE_MODE).strip().lower()
    if name not in SAMPLING_MODES:
        sys.stderr.write(
            f"[chain] 未知抽样模式 {name!r}，可选 {SAMPLING_MODES}；"
            f"已回退到 {DEFAULT_SAMPLE_MODE}\n")
        name = DEFAULT_SAMPLE_MODE
    return name


def sampling_indices(
    creds: list[dict],
    sp: int,
    mode: str | None = None,
) -> list[int]:
    """返回待验签的链内下标。点数 == min(sp, len(creds))。

    mode="uniform"    等间隔（历史口径）
    mode="stratified" 按节点分层，每节点保底 1 点
    """
    n = len(creds)
    if n <= sp:
        return list(range(n))
    m = mode or active_sample_mode()
    if m == "uniform":
        step = n / sp
        return sorted({int(i * step) for i in range(sp)})
    if m != "stratified":
        raise ValueError(f"未知抽样模式 {m!r}，可选 {list(SAMPLING_MODES)}")

    # 按节点归集链内下标
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(creds):
        key = str(c.get("node_id") or c.get("hospital_id") or "")
        groups.setdefault(key, []).append(i)

    k = len(groups)
    if k == 1:
        step = n / sp
        return sorted({int(i * step) for i in range(sp)})

    # 目标配额：点数够每节点保底 1 个时先各给 1 个，余下按占比分配（最大余数法）；
    # 点数不够 k 个时退化为纯按占比分配（此时无法保证覆盖，只能最大化覆盖节点数）。
    if sp >= k:
        quota = {g: 1 for g in groups}
        for g, q in _largest_remainder(
                {g: len(v) for g, v in groups.items()}, sp - k, n).items():
            quota[g] += q
    else:
        quota = _largest_remainder({g: len(v) for g, v in groups.items()},
                                   sp, n)

    # 逐节点取点。某节点配额超过其链上条数时只能给满该节点全部条数，总点数会
    # 少于 sp —— 把余量补给仍有富余的节点（同样按占比分配）。
    out: list[int] = []
    got: dict[str, int] = {}
    for g, pos in groups.items():
        take = _pick_spread(pos, quota.get(g, 0))
        got[g] = len(take)
        out.extend(take)

    shortfall = sp - len(out)
    if shortfall > 0:
        spare = {g: len(pos) - got.get(g, 0) for g, pos in groups.items()}
        spare = {g: c for g, c in spare.items() if c > 0}
        for g, q in _largest_remainder(spare, shortfall,
                                       sum(spare.values())).items():
            if q <= 0:
                continue
            pos = groups[g]
            have = got.get(g, 0)
            take = _pick_spread(pos, have + q)[have:]
            got[g] = have + len(take)
            out.extend(take)

    return sorted(set(out))


def _pick_spread(pos: list[int], want: int) -> list[int]:
    """从 pos（某节点的链内下标，递增）中取 want 个等间隔点。"""
    if want <= 0:
        return []
    if want >= len(pos):
        return list(pos)
    step = len(pos) / want
    return [pos[int(i * step)] for i in range(want)]



def _node_count(creds: list[dict]) -> int:
    """链上出现的不同节点数（node_id 优先，回落 hospital_id）。"""
    return len({str(c.get("node_id") or c.get("hospital_id") or "")
                for c in creds})

def _largest_remainder(weights: dict[str, int], total: int,
                       denom: int) -> dict[str, int]:
    """按 weights/denom 的比例把 total 个名额分给各项，总和恰为 total。"""
    if total <= 0:
        return {k: 0 for k in weights}
    exact = {k: total * w / denom for k, w in weights.items()}
    floors = {k: int(v) for k, v in exact.items()}
    short = total - sum(floors.values())
    order = sorted(weights, key=lambda k: (-(exact[k] - floors[k]), k))
    for k in order[:max(short, 0)]:
        floors[k] += 1
    return floors


def verify_chain(
    creds: list[dict],
    signer: SM2Signer | None = None,
    sm2_sample: int = 40,
    verifiers: dict | None = None,
    keyring_fp_published: str | None = None,
    sample_mode: str | None = None,
) -> dict:
    """验证凭证链。

    - 哈希连续性全量验证（快）
    - SM2 签名抽样验证（真实性的统计抽验，全量验签过慢）
    - G4 改造：传入 verifiers（node_id -> 仅公钥验证器）时，按每条凭证的 node_id
      选择对应节点公钥验签；此路径**不需要任何私钥**。
    - G5 改造：传入 keyring_fp_published（公示的公钥清单聚合指纹）且 verifiers
      非空时，先校验当前公钥清单的聚合指纹是否与公示值一致。

      为什么必须校验：公钥清单是验签的信任根。若它在**建链阶段**被替换，
      攻击者可用自己的私钥签发声称来自受害节点的凭证，此时链、签名与锚定记录
      三者完全自洽——链尾锚定反而为伪造背书。实测该攻击在未校验清单指纹时
      检出率为 0（见 9.2.3.2.2 节跨节点汇总实验）。
    - 抽样策略：sample_mode="uniform"（默认，历史口径）/ "stratified"
      （按节点分层，每节点保底 1 点）。亦可由环境变量 AUDIT_SAMPLE_MODE 切换。
      返回值含 nodes_total / nodes_sampled，使「某节点零覆盖」在产物中可见。
    返回 {valid, length, hash_ok, signatures_checked, signatures_verified,
          break_at, keyring_fp_ok, sample_mode, nodes_total, nodes_sampled}。
    """
    # G5 改造：公钥清单完整性校验（信任根）
    if keyring_fp_published and verifiers:
        from .crypto_signer import aggregate_keyring_fingerprint
        cur_fp = aggregate_keyring_fingerprint(
            {nid: v.public_key for nid, v in verifiers.items()})
        if cur_fp != keyring_fp_published:
            return {
                "valid": False,
                "length": len(creds),
                "hash_ok": False,
                "break_at": 0,
                "keyring_fp_ok": False,
                "reason": "公钥清单聚合指纹与公示值不符：信任根已被替换，签名可被伪造",
                "signatures_checked": 0,
                "signatures_verified": 0,
                "sample_mode": (sample_mode or active_sample_mode()),
                "nodes_total": _node_count(creds),
                "nodes_sampled": 0,
            }

    # G2 改造：结构一致性校验。seq 必须自 1 连续递增；单节点链中 node_id 应与
    # hospital_id 一致。这两项不进摘要（seq 具位置依赖性），改以显式校验约束，
    # 使"改 seq / 改 node_id"不再是不可检出的盲区。
    for idx, item in enumerate(creds, start=1):
        if item.get("seq") != idx:
            return {
                "valid": False,
                "length": len(creds),
                "hash_ok": False,
                "break_at": idx,
                "reason": f"seq 不连续：位置 {idx} 的 seq={item.get('seq')}",
                "signatures_checked": 0,
                "signatures_verified": 0,
                "sample_mode": (sample_mode or active_sample_mode()),
                "nodes_total": _node_count(creds),
                "nodes_sampled": 0,
            }

    prev = GENESIS
    for idx, item in enumerate(creds):
        if item["prev_hash"] != prev:
            return {
                "valid": False,
                "length": len(creds),
                "hash_ok": False,
                "break_at": idx + 1,
                "signatures_checked": 0,
                "signatures_verified": 0,
                "sample_mode": (sample_mode or active_sample_mode()),
                "nodes_total": _node_count(creds),
                "nodes_sampled": 0,
            }
        digest = hashlib.sha256(
            _payload_from_item(item, prev).encode("utf-8")
        ).hexdigest()
        if digest != item["hash"]:
            return {
                "valid": False,
                "length": len(creds),
                "hash_ok": False,
                "break_at": idx + 1,
                "signatures_checked": 0,
                "signatures_verified": 0,
                "sample_mode": (sample_mode or active_sample_mode()),
                "nodes_total": _node_count(creds),
                "nodes_sampled": 0,
            }
        prev = digest

    checked, verified = 0, 0
    unknown_nodes: set[str] = set()
    # 未传verifiers/signer 时不进行抽样验签，indices 保持为空，
    # 以免返回值的 nodes_sampled 引用未绑定变量。
    indices: list[int] = []
    if creds and (verifiers is not None or signer is not None):
        n = len(creds)
        indices = sampling_indices(creds, sm2_sample,
                                   sample_mode or active_sample_mode())
        for i in indices:
            item = creds[i]
            checked += 1
            if verifiers is not None:
                key = str(item.get("node_id") or item.get("hospital_id"))
                vf = verifiers.get(key)
                if vf is None:
                    # fail-closed：公钥清单中没有该节点，无法确认其签名合法性，
                    # 不得与"验证通过"同等对待（旧实现静默跳过，会把"来源不可验证"
                    # 误计为"未检出"，低估攻击成功率）。
                    unknown_nodes.add(key)
                    continue
                if vf.verify(item["hash"], item["signature"]):
                    verified += 1
            elif signer.verify(item["hash"], item["signature"]):
                verified += 1

    return {
        "valid": True,
        "length": len(creds),
        "hash_ok": True,
        "signatures_checked": checked,
        "signatures_verified": verified,
        "unknown_node_count": len(unknown_nodes),
        "unknown_nodes": sorted(unknown_nodes)[:8],
        "sample_mode": (sample_mode or active_sample_mode()),
        "nodes_total": len({str(c.get("node_id") or c.get("hospital_id") or "")
                            for c in creds}),
        "nodes_sampled": len({str(creds[i].get("node_id")
                                  or creds[i].get("hospital_id") or "")
                              for i in indices}),
        "genesis": GENESIS,
        "tip": creds[-1]["hash"] if creds else GENESIS,
        "keyring_fp_ok": True if keyring_fp_published else None,
    }
