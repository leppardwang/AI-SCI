from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

CORE_PLATFORMS = frozenset({"san_yi"})
R1_EXCLUDE_ACTIONS = frozenset({"bulk_export", "nightly_backup"})
HIGH_RISK_TYPES = frozenset({"data_leak", "brute_force"})
R3_VOLUME_KB = 5000.0
R4_RATE = 0.002
R4_SEED = 42

# ---------------------------------------------------------------------------
# 规则集注册表
# ---------------------------------------------------------------------------
# v1（冻结）：原始四条规则。**不得修改**——第 9 章已入库的 12 组产物
#   （experiment/results/*.json）与 §7.3.1 / §7.4.1 的表格数字全部基于此版本。
#   保留它是为了让旧产物可复现，而不是继续推荐使用。
#
# v2（现行）：把 R4 从「检出规则」降级为「基线」。
#   实测依据：R4 的实现是 `RandomState(42).choice(n, 1600, replace=False)`，
#   不读取任何字段，输出恒为抽样点数；而语料无账号状态字段，该规则无法被
#   真实实现。把它计入检出规则，等于把随机数发生器接到规则引擎上。
#   真正实现 R4 需要语料新增 account_status 字段——平台语料不可再生成
#   （见 REPRODUCIBILITY.md §4.1.1），故 v2 直接移除而非降级保留。
#
# 切换方式：环境变量 AUDIT_RULESET=v1|v2（默认 v2）。
#   复现旧产物： AUDIT_RULESET=v1 python3 experiment/<脚本>.py

_RULES_V1 = (
    ("R1", "minimum_necessary", "最小必要违规"),
    ("R2", "cross_border_unmasked", "跨境未脱敏"),
    ("R3", "unmasked_export", "未脱敏导出"),
    ("R4", "deactivated_account_login", "已注销账号登录"),
)

# 非检出项：实现上不读取语料字段，输出由抽样点数决定。仍随 apply_rules 返回
# （保持返回键不变），但不计入 RULES，故不进入凭证链。
_BASELINES = (
    ("R4", "random_sample_baseline",
     "随机抽样基线（原「已注销账号登录」占位实现）"),
)

RULESETS = {
    "v1": {"rules": _RULES_V1, "baselines": (),
           "note": "冻结版，支撑已入库产物；R4 为占位实现"},
    "v2": {"rules": tuple(r for r in _RULES_V1 if r[0] != "R4"),
           "baselines": _BASELINES,
           "note": "移除占位 R4；§9.2.3 链长因此由 23,110 变为 21,510"},
}

DEFAULT_RULESET = "v2"


def active_ruleset() -> str:
    name = os.environ.get("AUDIT_RULESET", DEFAULT_RULESET).strip().lower()
    if name not in RULESETS:
        sys.stderr.write(
            f"[rules] 未知规则集 {name!r}，可选 {sorted(RULESETS)}；"
            f"已回退到 {DEFAULT_RULESET}\n")
        name = DEFAULT_RULESET
    return name


ACTIVE_RULESET = active_ruleset()
# 规则元组：(code, key, label)
RULES = RULESETS[ACTIVE_RULESET]["rules"]
# 非检出项元组
BASELINES = RULESETS[ACTIVE_RULESET]["baselines"]

_BANNER_SHOWN = False


def ruleset_info() -> dict:
    """供产物与论文标注当前口径。"""
    return {
        "active": ACTIVE_RULESET,
        "rule_codes": [r[0] for r in RULES],
        "baseline_codes": [b[0] for b in BASELINES],
        "note": RULESETS[ACTIVE_RULESET]["note"],
    }


def _show_banner() -> None:
    """每次进程运行只提示一次，避免旧产物在无提示的情况下被静默改变口径。"""
    global _BANNER_SHOWN
    if _BANNER_SHOWN:
        return
    _BANNER_SHOWN = True
    if ACTIVE_RULESET != "v1":
        sys.stderr.write(
            f"[rules] 生效规则集 = {ACTIVE_RULESET}：检出规则 "
            f"{[r[0] for r in RULES]}；基线（非检出）"
            f" {[b[0] for b in BASELINES]}"
            f"（{RULESETS[ACTIVE_RULESET]['note']}）\n"
            f"[rules] 复现 v1 旧产物请设 AUDIT_RULESET=v1\n")


def compute_p99(volume: pd.Series, risk_label: pd.Series) -> float:
    normal = volume[risk_label.astype(int) == 0]
    if normal.empty:
        raise ValueError("无法计算 p99：语料中没有 risk_label=0 的正常事件")
    return float(normal.quantile(0.99))


def apply_rules(
    df: pd.DataFrame,
    p99: float | None = None,
    seed: int = R4_SEED,
    r4_rate: float = R4_RATE,
) -> dict:
    """返回 R1..R4 的布尔掩码。

    **返回键与 v1 完全一致**（含 R4），故任何按 key 取值的既有代码不会因本次
    改动而 KeyError。变化仅在于 `RULES` 不再把 R4 当作检出规则——调用方
    （9 个实验脚本）均以 `{rid: res[rid] for rid, _, _ in RULES}` 取掩码，
    故口径切换自动生效。

    R4 为非检出基线：它按固定种子抽 r4_rate 比例的行，不读取任何语料字段，
    计入检出规则会把随机数发生器接入规则引擎。详见 RULESETS 注释。
    """
    _show_banner()
    volume = df["volume_kb"].astype(float)
    risk_label = df["risk_label"].astype(int)
    action = df["action"].astype(str)
    risk_type = df["risk_type"].astype(str)
    dest = df["dest_platform"].astype(str)
    trusted = df["is_trusted_line"].map(_as_bool)

    if p99 is None:
        p99 = compute_p99(volume, risk_label)

    r1 = (
        (risk_label == 0)
        & (volume > p99)
        & (~action.isin(R1_EXCLUDE_ACTIONS))
    )
    r2 = (~dest.isin(CORE_PLATFORMS)) & (~trusted)
    r3 = (
        (action == "bulk_export")
        | (volume > R3_VOLUME_KB)
        | (risk_type == "data_leak")
    ) & (~trusted)

    r4 = pd.Series(False, index=df.index)
    n = len(df)
    k = int(round(n * r4_rate))
    if k > 0 and n > 0:
        rng = np.random.RandomState(seed)
        chosen = rng.choice(n, size=min(k, n), replace=False)
        r4.iloc[chosen] = True

    return {
        "p99": p99,
        "R1": r1,
        "R2": r2,
        "R3": r3,
        "R4": r4,
    }


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"无法解析 is_trusted_line: {value!r}")
