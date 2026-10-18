from pathlib import Path

import pandas as pd

from backend.chain import (GENESIS, build_chain, collect_hits,
                        sampling_indices, verify_chain)
from backend.crypto_signer import SM2Signer
from backend.rules import (BASELINES, RULES, RULESETS, active_ruleset,
                        apply_rules, compute_p99, ruleset_info)

# AGENTS §9.4：仓内语料缺失即报错，不静默回落到另一份语料。
CSV = (Path(__file__).resolve().parent.parent
       / "data" / "lecheng_security_logs_2024H1_pipeline10.csv")
if not CSV.is_file():
    raise FileNotFoundError(
        f"仓内语料缺失：{CSV}（AGENTS §9.4 禁止回退）")


def test_p99_on_tiny_normal_sample():
    df = pd.DataFrame(
        {
            "volume_kb": [1, 2, 3, 4, 100],
            "risk_label": [0, 0, 0, 0, 1],
        }
    )
    assert abs(compute_p99(df["volume_kb"], df["risk_label"]) - 3.97) < 1e-9


def test_r1_r2_r3_r4_on_fixture():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:00"] * 6,
            "hospital_id": ["ruijin_hn"] * 6,
            "volume_kb": [10.0, 999.0, 20.0, 8000.0, 30.0, 40.0],
            "latency_ms": [10.0] * 6,
            "risk_label": [0, 0, 1, 1, 0, 0],
            "risk_type": ["normal", "normal", "data_leak", "data_leak", "normal", "normal"],
            "action": [
                "api_query",
                "api_query",
                "bulk_export",
                "bulk_export",
                "nightly_backup",
                "api_query",
            ],
            "response_status": [200] * 6,
            "is_trusted_line": [True, True, False, False, True, True],
            "dest_platform": ["san_yi", "san_yi", "san_yi", "other", "san_yi", "san_yi"],
        }
    )
    result = apply_rules(df, p99=50.0, seed=42, r4_rate=0.0)
    assert int(result["R1"].sum()) == 1
    assert bool(result["R1"].iloc[1])
    assert int(result["R2"].sum()) == 1
    assert bool(result["R2"].iloc[3])
    assert int(result["R3"].sum()) == 2
    assert int(result["R4"].sum()) == 0


def test_chain_hashes_link():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:01", "2024-01-01 00:00:02"],
            "hospital_id": ["boai", "wenlan"],
            "volume_kb": [8000.0, 9000.0],
            "latency_ms": [1.0, 2.0],
            "risk_label": [1, 1],
            "risk_type": ["data_leak", "data_leak"],
            "action": ["bulk_export", "bulk_export"],
            "response_status": [200, 200],
            "is_trusted_line": [False, False],
            "dest_platform": ["san_yi", "san_yi"],
        }
    )
    masks = apply_rules(df, p99=1.0, r4_rate=0.0)
    hits = collect_hits(df, {k: masks[k] for k, _, _ in RULES})
    creds, status = build_chain(hits)
    assert status["valid"] is True
    assert len(creds) == 2
    assert creds[1]["prev_hash"] == creds[0]["hash"]
    status = verify_chain(creds)
    assert status["valid"] is True
    tampered = [dict(creds[0]), dict(creds[1])]
    tampered[1]["prev_hash"] = "dead" * 16
    assert verify_chain(tampered)["valid"] is False


def test_source_csv_four_rules_live():
    if not CSV.is_file():
        raise FileNotFoundError(f"数据不可获取：{CSV}")
    df = pd.read_csv(CSV)
    result = apply_rules(df)
    r1 = int(result["R1"].sum())
    r2 = int(result["R2"].sum())
    r3 = int(result["R3"].sum())
    r4 = int(result["R4"].sum())
    assert abs(result["p99"] - 534.76515) < 1e-4
    assert r1 == 7569
    assert r2 == 0
    assert r3 == 5974
    assert r4 == 1600
    assert r1 + r2 + r3 + r4 == 15143
    top = df.loc[result["R1"], "hospital_id"].value_counts()
    assert int(top.iloc[0]) == 7327
    assert top.index[0] == "ruijin_hn"


def test_chain_signed_with_sm2_and_verified():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:01", "2024-01-01 00:00:02"],
            "hospital_id": ["boai", "boai"],
            "volume_kb": [8000.0, 9000.0],
            "latency_ms": [1.0, 2.0],
            "risk_label": [1, 1],
            "risk_type": ["data_leak", "data_leak"],
            "action": ["bulk_export", "bulk_export"],
            "response_status": [200, 200],
            "is_trusted_line": [False, False],
            "dest_platform": ["san_yi", "san_yi"],
        }
    )
    masks = apply_rules(df, p99=1.0, r4_rate=0.0)
    hits = collect_hits(df, {k: masks[k] for k, _, _ in RULES})
    signer = SM2Signer("1" * 64)
    creds, status = build_chain(hits, signer=signer, node_id="boai")
    assert all(c["signature"] for c in creds)
    assert signer.verify(creds[0]["hash"], creds[0]["signature"])
    status = verify_chain(creds, signer=signer, sm2_sample=40)
    assert status["valid"] is True
    assert status["hash_ok"] is True
    assert status["signatures_checked"] == 2
    assert status["signatures_verified"] == 2
    tampered = [dict(c) for c in creds]
    tampered[0]["signature"] = "0" * 140
    bad = verify_chain(tampered, signer=signer, sm2_sample=40)
    assert bad["signatures_verified"] == 1

# ---------------------------------------------------------------------------
# 规则集版本边界（2026-10-03 加入）
#
# 背景：R4「已注销账号登录」的实现是 RandomState(42).choice(n, 1600)，
# 不读取任何语料字段，输出恒为抽样点数，而语料无账号状态字段——它是占位
# 实现，不可能是真实检出规则。v2 把它从检出规则降级为基线。
#
# 下面两条测试的作用是：**任何人改动 RULES 组成或破坏 v1 复现能力时，
# 测试立刻失败**，而不是让 §9.2.3 的产物在无告警的情况下被静默改口径。
# ---------------------------------------------------------------------------


def test_ruleset_registry_shape():
    info = ruleset_info()
    assert set(RULESETS) == {"v1", "v2"}, "规则集应恰为 v1/v2"
    # v1 必须保留全部四条，否则无法复现已入库产物
    assert [r[0] for r in RULESETS["v1"]["rules"]] == ["R1", "R2", "R3", "R4"]
    # R4 无论在哪个版本都必须仍能由 apply_rules 取到（返回键不变是硬约束）
    assert "R4" in [b[0] for b in RULESETS["v2"]["baselines"]], \
        "v2 必须把 R4 列入基线而非删除其实现"


def test_r4_is_not_a_detection_rule_in_active_set():
    """当前生效规则集若为 v2，则 R4 不得计入凭证链。"""
    info = ruleset_info()
    assert info["active"] == active_ruleset()
    if info["active"] != "v2":
        # v1 下本测试不适用，但必须仍能取到 R4 实现
        assert "R4" in info["rule_codes"]
        return
    assert "R4" not in info["rule_codes"], "v2 下 R4 不得作为检出规则"
    assert [b[0] for b in BASELINES] == ["R4"]
    # R4 实现仍在（返回键不变），但不进链
    df = pd.DataFrame(
        {
            "volume_kb": [1.0, 9000.0, 2.0],
            "risk_label": [0, 0, 0],
            "timestamp": ["2026-01-01T00:00:00"] * 3,
            "hospital_id": ["h00", "h00", "h00"],
            "latency_ms": [10.0, 10.0, 10.0],
            "action": ["api_query", "api_query", "api_query"],
            "risk_type": ["normal", "normal", "normal"],
            "response_status": [200, 200, 200],
            "is_trusted_line": [False, False, False],
            "dest_platform": ["san_yi", "san_yi", "san_yi"],
        }
    )
    masks = apply_rules(df, p99=1.0, r4_rate=1.0)
    assert int(masks["R4"].sum()) == 3, "R4 实现应仍在（返回键不变）"
    hits = collect_hits(df, {k: masks[k] for k, _, _ in RULES})
    # 注意：collect_hits 不按事件去重，同一事件被 N 条规则命中即产生 N 行
    # （见 REPRODUCIBILITY.md §4.3）。故此处期望 R1 的 2 行 + R3 的 1 行 = 3。
    # R4 虽命中 3 行，但 v2 下不是检出规则，不进链。
    assert sorted(hits["rule_id"]) == ["R1", "R1", "R3"]
    assert len(hits) == 3, "链中只应含 R1/R3 的 3 条命中，不含 R4"
    assert "R4" not in set(hits["rule_id"])


def test_ruleset_v1_reproduces_legacy_chain_length():
    """v1 必须复现已入库产物的链长 23,110；v2 恰少 R4 的 1,600。"""
    import os
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    snip = (
        "import os,sys\n"
        f"sys.path.insert(0, {root!r})\n"
        f"sys.path.insert(0, os.path.join({root!r}, 'experiment'))\n"
        "from backend.rules import RULES, apply_rules\n"
        "from backend.chain import collect_hits\n"
        "from adversarial_test import make_corpus, SEED\n"
        "df = make_corpus(800_000, SEED, 0.99)\n"
        "r = apply_rules(df)\n"
        "print(len(collect_hits(df, {k: r[k] for k,_,_ in RULES})))\n"
    )
    got = {}
    for v, want in (("v1", 23110), ("v2", 21510)):
        env = dict(os.environ, AUDIT_RULESET=v, PYTHONIOENCODING="utf-8")
        p = subprocess.run([sys.executable, "-c", snip], capture_output=True,
                           text=True, encoding="utf-8", env=env, cwd=root)
        assert p.returncode == 0, p.stderr[-1500:]
        got[v] = int(p.stdout.strip().splitlines()[-1])
        assert got[v] == want, f"{v} 链长 {got[v]} != 预期 {want}"
    assert got["v1"] - got["v2"] == 1600, "两版差值应恰为 R4 的 1600"

# ---------------------------------------------------------------------------
# 验签抽样策略（2026-10-03 加入）
#
# 背景：等间隔抽样在节点负载不均衡时覆盖严重倾斜。09-28 平台语料实测
# （链长 15,143、链级负载比 39.4018×、SP=40）：
#   uniform     零覆盖节点 3 个（ciming/shulan/wenlan），最大倾斜 2.89×
#   stratified  零覆盖节点 0 个，最大倾斜 2.25×
# 「每节点最少抽点数」由 0 提升到 1（保底保证）。
#
# 注意：分层抽样**不是**把倾斜压到 1.0，而是消除零覆盖这个硬缺口。
# 剩余倾斜是 40 点 / 12 节点的算术下限——小节点按保底拿 1 点，
# 而它按占比只应得 0.58 点，比值必然 > 1。
#
# 下面测试的作用：任何人改坏分层逻辑、或让 uniform 口径漂移时立刻失败，
# 而不是让 §9.2.3 的产物在无告警下被改口径。
# ---------------------------------------------------------------------------


def _toy_creds(per_node: dict[str, int]) -> list[dict]:
    out = []
    for node, k in per_node.items():
        for _ in range(k):
            out.append({"node_id": node, "hash": "0" * 64,
                        "signature": "00"})
    return out


def _valid_chain(per_node: dict[str, int]) -> list[dict]:
    """构造一条**合法**链：prev_hash / seq / 摘要三者自洽。

    verify_chain 在 prev_hash 或摘要不符时会提前返回，那些返回不含抽样字段，
    故本测试必须用合法链才能验到 sample_mode。摘要口径用chain.py 的
    _payload_from_item，避免手写复刻而漂移。
    """
    from backend.chain import _payload_from_item

    creds, prev = [], GENESIS
    for node, k in per_node.items():
        for _ in range(k):
            item = {
                "seq": len(creds) + 1,
                "node_id": node,
                "timestamp": f"2026-01-01T00:00:{len(creds):02d}",
                "hospital_id": node,
                "rule_id": "R1",
                "rule_key": "minimum_necessary",
                "rule_label": "最小必要违规",
                "risk_type": "normal",
                "risk_label": 0,
                "latency_ms": 10.0,
                "response_status": 200,
                "source_index": len(creds),
                "volume_kb": 1.0,
                "action": "api_query",
                "dest_platform": "san_yi",
                "is_trusted_line": False,
            }
            item["prev_hash"] = prev
            item["hash"] = __import__("hashlib").sha256(
                _payload_from_item(item, prev).encode("utf-8")).hexdigest()
            item["signature"] = "00"
            creds.append(item)
            prev = item["hash"]
    return creds


def test_sampling_uniform_is_frozen_legacy_formula():
    """uniform 必须逐字保持历史公式，否则已入库产物不可复现。"""
    creds = [{"node_id": f"n{i}", "hash": "0" * 64, "signature": "00"}
             for i in range(1000)]
    n, sp = len(creds), 40
    step = n / sp
    expect = sorted({int(i * step) for i in range(sp)})
    assert sampling_indices(creds, sp, "uniform") == expect


def test_sampling_stratified_guarantees_every_node_one_point():
    """每节点保底 1 点——这是分层抽样的核心目的。"""
    per_node = {"big": 8629, "mid": 1058, "s1": 834, "s2": 699,
                "s3": 648, "s4": 731, "s5": 805, "s6": 424,
                "s7": 393, "s8": 366, "s9": 337, "s10": 219}
    creds = _toy_creds(per_node)
    idx = sampling_indices(creds, 40, "stratified")
    assert len(idx) == 40
    assert len(set(idx)) == len(idx)
    assert all(0 <= i < len(creds) for i in idx)
    got = {}
    for i in idx:
        got[creds[i]["node_id"]] = got.get(creds[i]["node_id"], 0) + 1
    missing = [k for k in per_node if k not in got]
    assert not missing, f"分层抽样漏掉节点：{missing}"
    assert min(got.values()) >= 1


def test_sampling_stratified_eliminates_zero_coverage():
    """对照：等间隔在同一负载分布下确实会漏节点。"""
    per_node = {"big": 8629, "mid": 1058, "s1": 834, "s2": 699,
                "s3": 648, "s4": 731, "s5": 805, "s6": 424,
                "s7": 393, "s8": 366, "s9": 337, "s10": 219}
    creds = _toy_creds(per_node)

    def covered(mode):
        idx = sampling_indices(creds, 40, mode)
        return {creds[i]["node_id"] for i in idx}

    u, s = covered("uniform"), covered("stratified")
    assert len(s) == len(per_node), "分层应覆盖全部节点"
    # uniform 是否漏节点取决于负载分布，这里只断言分层不劣于 uniform
    assert len(s) >= len(u)


def test_sampling_point_count_invariants():
    for total in (1, 39, 40, 41, 100, 15143):
        for nodes in (1, 2, 12, 50):
            per = {f"n{i}": max(1, total // nodes) for i in range(nodes)}
            per["n0"] += total - sum(per.values())
            creds = _toy_creds({k: v for k, v in per.items() if v > 0})
            for mode in ("uniform", "stratified"):
                idx = sampling_indices(creds, 40, mode)
                want = min(40, len(creds))
                assert len(idx) == want, \
                    f"{mode} total={total} nodes={nodes} -> {len(idx)} != {want}"
                assert len(set(idx)) == len(idx)


class _AlwaysFailVerifier:
    """占位验签器：永远返回 False。用于只验抽样覆盖、不引入真实密码运算。"""

    def verify(self, digest: str, signature: str) -> bool:
        return False


def test_payload_covers_every_rule_input():
    """链的摘要必须覆盖全部规则判据字段——否则规则输入可被改而链察觉不到。

    实测依据（experiment/results/chain_field_coverage_results.json）：
    risk_label 曾不在摘要内，而 R1 的唯一判据就是 risk_label == 0、
    compute_p99 也用它筛正常事件。篡改该列后 verify_chain 仍判 valid，
    而 R1 命中数由 562 降至 312。该列已于 2026-10-03 纳入摘要。

    response_status 仍不在摘要内：当前无任何规则读取它（实测其改动不改变
    规则命中集合），故为已知且可接受的范围外字段。若日后有规则改读该列，
    本断言会失败，提示必须同步扩摘要。
    """
    from backend.chain import _PAYLOAD_FIELDS
    for col in ("volume_kb", "latency_ms", "risk_label", "risk_type",
                "action", "dest_platform", "is_trusted_line",
                "hospital_id", "timestamp"):
        assert col in _PAYLOAD_FIELDS, f"{col} 是规则判据，必须在摘要内"
    # 摘要字段顺序若在 is_trusted_line 之前插入新字段，硬编码下标会错位
    assert _PAYLOAD_FIELDS.index("is_trusted_line") == 7, \
        "is_trusted_line 的位置变了——检查 _TRUSTED_INDEX 是否仍正确"


def test_verify_chain_reports_sampling_coverage():
    """返回值必须让「某节点零覆盖」在产物中可见。"""
    creds = _valid_chain({"a": 5, "b": 1})
    vfs = {"a": _AlwaysFailVerifier(), "b": _AlwaysFailVerifier()}
    st = verify_chain(creds, verifiers=vfs, sm2_sample=2,
                      sample_mode="uniform")
    assert st["valid"] is True, st.get("reason")
    assert st["sample_mode"] == "uniform"
    assert st["nodes_total"] == 2
    # 验签必然失败（占位验签器），但抽样覆盖仍应被如实统计
    assert st["signatures_checked"] == 2
    assert st["signatures_verified"] == 0
    assert 1 <= st["nodes_sampled"] <= 2
    # 分层模式下每节点保底 1 点：sp=2、2 节点 → 各抽 1 个，零覆盖不可能
    st2 = verify_chain(creds, verifiers=vfs, sm2_sample=2,
                       sample_mode="stratified")
    assert st2["nodes_sampled"] == 2, "分层抽样应覆盖全部 2 个节点"
    # 不传 verifiers/signer 时不进行抽样，nodes_sampled 记 0 且不得抛错
    st3 = verify_chain(creds, sm2_sample=2)
    assert st3["signatures_checked"] == 0
    assert st3["nodes_sampled"] == 0
    # 链已判坏时的提前返回也必须带这三个字段（消费方统一读取不应 KeyError）
    broken = [dict(c) for c in creds]
    broken[1]["prev_hash"] = "dead" * 16
    st4 = verify_chain(broken, verifiers=vfs, sm2_sample=2)
    assert st4["valid"] is False
    for k in ("sample_mode", "nodes_total", "nodes_sampled"):
        assert k in st4, f"提前返回缺字段 {k}"