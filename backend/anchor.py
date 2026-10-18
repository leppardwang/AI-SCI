#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""链尾外部锚定（G3）—— 不依赖外部服务的实现

原理
----
纯哈希链只能发现「改了却没接着改后续链接」的局部篡改：链尾既可被整段重写，
也可被截断，而链自身不含任何长度或内容的外部承诺。本模块用**两条互补承诺**
把链尾状态外部化：

  承诺一：Merkle 根（内容承诺）
      对链上全部凭证的 hash 构树，根即链内容的唯一指纹。链尾任何增删改都会改变根。

  承诺二：链式公示日志（时间与顺序承诺）
      每次核验生成一条公示记录 anchor = {
          chain_id, length, tip, merkle_root, utc_time, prev_anchor_hash
      }
      其中 prev_anchor_hash = SHA256(前一条记录 ‖ 本条其余字段)，
      形成**独立的第二条哈希链**。公示日志按追加写入，核验方比对最新一条与当前链，
      长度不符或 Merkle 根不符即判定链尾被增删改。

能与不能
--------
能：发现针对**单条链**的截断、尾部重写、尾部追加、凭证重放（前提是公示记录已存在，
    且攻击者未同时改写公示日志）。
不能：防御「攻击者同时握有链存储与公示日志写权限」的情形——真正的外部锚定需要
    第三方存证或可信时间戳服务（RFC 3161）。本文如实标注该边界，不作过度声明。

用法
----
    from backend.anchor import ChainAnchorLog
    log = ChainAnchorLog("data/anchors/chain_anchors.jsonl")
    rec = log.publish("main", credentials)          # 核验通过后公示
    ok, why = log.verify("main", credentials)        # 核验最新公示与当前链是否一致

命令行自检：
    python -m backend.anchor --selftest
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from typing import Iterable, Sequence


# --------------------------------------------------------------------------- Merkle
def merkle_root(leaf_hashes: Sequence[str]) -> str:
    """对凭证 hash 序列求 Merkle 根。空链返回 64 个 '0'。"""
    if not leaf_hashes:
        return "0" * 64
    level = [bytes.fromhex(h) for h in leaf_hashes]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left   # 奇数节点自复制
            nxt.append(hashlib.sha256(left + right).digest())
        level = nxt
    return level[0].hex()


def _anchor_hash(rec: dict) -> str:
    """公示记录的链式哈希：SHA256(prev_anchor_hash ‖ 规范化字段)。"""
    body = {k: v for k, v in rec.items() if k != "anchor_hash"}
    payload = rec.get("prev_anchor_hash", "") + json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- 日志
class ChainAnchorLog:
    """链式公示日志（append-only JSONL）。"""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # ---- 读 ----
    def records(self, chain_id: str | None = None) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if chain_id is None or rec.get("chain_id") == chain_id:
                    out.append(rec)
        return out

    def latest(self, chain_id: str) -> dict | None:
        rs = self.records(chain_id)
        return rs[-1] if rs else None

    # ---- 写 ----
    def publish(self, chain_id: str, credentials: Iterable[dict],
                ts: str | None = None,
                keyring_fp: str | None = None) -> dict:
        """公示链长、链尾摘要与内容根；可选一并公示公钥清单的聚合指纹。

        keyring_fp（G5 改造）：把公钥清单指纹与链状态放进**同一信任根**。
        公钥清单是验签的信任根——若它可在建链阶段被替换，链、签名与本锚定记录
        三者会完全自洽，锚定反而为伪造背书。两者同公示即共用一条信任链，
        比另设独立公示渠道更自洽（后者只增加信任假设而不增加防护）。
        """
        creds = list(credentials)
        prev = self.latest(chain_id)
        rec = {
            "chain_id": chain_id,
            "length": len(creds),
            "tip": creds[-1]["hash"] if creds else "0" * 64,
            "merkle_root": merkle_root([c["hash"] for c in creds]),
            "utc_time": ts or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "prev_anchor_hash": (prev or {}).get("anchor_hash", "0" * 64),
        }
        if keyring_fp:
            rec["keyring_fp"] = keyring_fp
        rec["anchor_hash"] = _anchor_hash(rec)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    # ---- 核验 ----
    def verify_state(self, chain_id: str,
                     credentials: Iterable[dict],
                     keyring_fp: str | None = None) -> tuple[str, str]:
        """把当前链与最新公示记录比对，返回三态 (state, 原因)。

        state ∈ {"ok", "pending", "violation"}

        为什么需要 pending：链尾锚定只能承诺"公示时点的链长与内容根"。若节点仍在
        持续追加凭证，当前链就可能比公示记录**短**——这与链尾被截断在密码学上
        **无法区分**（都表现为"变短"）。若为消除该误报而把"变短"一律判为正常，
        则真正的截断将无法检出。因此这里把两者在接口层分开：

          ok         与最新公示记录完全一致
          violation  安全事件：内容被改写、链尾被截断、存在未公示的追加、
                     公示缺失、或公示日志自身被改写
          pending    运行状态：当前链短于公示链长，可能是节点仍在追加而锚定未追平。
                     **由业务侧决定**是等待下一锚定期限，还是按期限转为 violation。

        需要安全语义（默认、把 pending 当 violation）时请用 verify()。
        """
        creds = list(credentials)
        rec = self.latest(chain_id)
        if rec is None:
            return "violation", "无公示记录，链尾状态未外部承诺"
        # G5：公示记录含公钥清单指纹时，一并校验信任根是否被替换
        if keyring_fp and rec.get("keyring_fp") and keyring_fp != rec["keyring_fp"]:
            return "violation", ("公钥清单聚合指纹与公示值不符：信任根已被替换，"
                                 "签名可被伪造")
        cur_n = len(creds)
        rec_n = rec["length"]
        if cur_n > rec_n:
            return "violation", (f"当前链长 {cur_n} 超过公示链长 {rec_n}："
                                 f"存在未公示的追加")
        cur_root = merkle_root([c["hash"] for c in creds])
        if cur_n == rec_n:
            if rec["merkle_root"] != cur_root:
                return "violation", "Merkle 根不符：链内容已被改写"
            cur_tip = creds[-1]["hash"] if creds else "0" * 64
            if rec["tip"] != cur_tip:
                return "violation", "链尾 tip 不符"
            return "ok", "与最新公示记录一致"
        # 变短：密码上无法区分"仍在追加"与"被截断"，交由业务侧按三态处理
        return "pending", (f"当前链长 {cur_n} 小于公示链长 {rec_n}："
                          f"节点可能仍在追加（锚定未追平），也可能是链尾被截断；"
                          f"须待锚定追平后复核")

    def verify(self, chain_id: str, credentials: Iterable[dict],
               keyring_fp: str | None = None) -> tuple[bool, str]:
        """把当前链与最新公示记录比对。返回 (是否一致, 原因)。

        严格二态口径：verify_state 的 pending 在此**计为不一致**，
        以保证链尾截断（A6）始终被判为安全事件。
        """
        state, why = self.verify_state(chain_id, credentials, keyring_fp)
        if state == "ok":
            return True, why
        return False, f"[{state}] {why}"

    def verify_internal_chain(self, chain_id: str) -> tuple[bool, str]:
        """核验公示日志自身的链式连续性（检测公示日志被中段改写/删除）。"""
        rs = self.records(chain_id)
        if not rs:
            return False, "无公示记录"
        prev = "0" * 64
        for i, rec in enumerate(rs, 1):
            if rec.get("prev_anchor_hash") != prev:
                return False, f"第 {i} 条公示记录的 prev 链接断裂"
            if rec.get("anchor_hash") != _anchor_hash(rec):
                return False, f"第 {i} 条公示记录内容被改写"
            prev = rec["anchor_hash"]
        return True, f"{len(rs)} 条公示记录链式完整"


# --------------------------------------------------------------------------- 自检
def _selftest() -> int:
    import random
    ok = fail = 0

    def t(desc, cond, det=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  [PASS] {desc}" + (f"  | {det}" if det else ""))
        else:
            fail += 1
            print(f"  [FAIL] {desc}" + (f"  | {det}" if det else ""))

    tmp = os.path.join(os.environ.get("TEMP", "."), f"anchor_selftest_{random.randint(1_000_000, 9_999_999)}")
    log = ChainAnchorLog(os.path.join(tmp, "chain_anchors.jsonl"))
    n = 300
    creds = [{"hash": hashlib.sha256(f"c{i}".encode()).hexdigest()} for i in range(n)]

    print("=" * 70)
    print("G3 链尾锚定自检")
    print("=" * 70)

    rec = log.publish("main", creds)
    print(f"  公示记录: length={rec['length']} merkle_root={rec['merkle_root'][:24]}…")
    v, why = log.verify("main", creds)
    t("原链与公示一致", v, why)
    v, why = log.verify_internal_chain("main")
    t("公示日志自身链式完整", v, why)

    # 攻击1：截断
    cut = creds[: int(n * 0.8)]
    v, why = log.verify("main", cut)
    t("攻击A6 截断尾部被锚定检出", (not v), why)

    # 攻击2：尾部追加
    app = creds + [{"hash": hashlib.sha256(b"forged").hexdigest()} for _ in range(5)]
    v, why = log.verify("main", app)
    t("攻击A7 链尾追加伪造被锚定检出", (not v), why)

    # 攻击3：尾部重写（改载荷并重算后续链接）
    rw = [dict(c) for c in creds]
    tail = int(n * 0.9)
    rw[tail]["hash"] = hashlib.sha256(b"rewritten").hexdigest()
    for j in range(tail + 1, n):
        rw[j]["hash"] = hashlib.sha256((rw[j]["hash"] + str(j)).encode()).hexdigest()
    v, why = log.verify("main", rw)
    t("攻击A4/A5 尾部重写被锚定检出", (not v), why)

    # 攻击4：重放（复制中段一条到链尾，长度变）
    rp = creds + [dict(creds[50])]
    v, why = log.verify("main", rp)
    t("攻击A8 凭证重放被锚定检出", (not v), why)

    # 攻击5：整链重写（等长、内容全换）
    full = [{"hash": hashlib.sha256(f"x{i}".encode()).hexdigest()} for i in range(n)]
    v, why = log.verify("main", full)
    t("攻击：等长整链重写被锚定检出", (not v), why)

    # 攻击6：公示日志中段被改写
    log2 = ChainAnchorLog(os.path.join(tmp, "log2.jsonl"))
    log2.publish("main", creds[:100], ts="2026-01-01T00:00:00+00:00")
    log2.publish("main", creds, ts="2026-01-02T00:00:00+00:00")
    lines = open(log2.path, encoding="utf-8").read().splitlines()
    r0 = json.loads(lines[0]); r0["length"] = 999
    lines[0] = json.dumps(r0, ensure_ascii=False)
    open(log2.path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    v, why = log2.verify_internal_chain("main")
    t("公示日志中段被改写可检出", (not v), why)

    # 正常流程：追加公示后仍一致
    log.publish("main", creds)
    v, why = log.verify("main", creds)
    t("重复公示后仍一致", v, why)
    v, why = log.verify_internal_chain("main")
    t("重复公示后日志链式仍完整", v, why)

    print(f"\nPASS={ok} FAIL={fail}")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    raise SystemExit(_selftest() if a.selftest else 0)