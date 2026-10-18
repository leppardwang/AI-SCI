#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild the lost synthetic corpus `lecheng_security_logs_2024H1.csv`.

The original 800,000-row / 12-hospital corpus that fed
`edge-audit-system` (backend/rules.py, engine.py, chain.py, features.py,
scripts/precompute_metrics.py, scripts/train_lstm.py) was deleted from disk.
This generator reconstructs a corpus that (a) matches the exact input schema
those modules require and (b) reproduces every aggregate documented in
REPRODUCIBILITY.md / results/metrics_live.json.

Schema required by the pipeline (see backend/engine.py:63-74 `required` set):
    timestamp, hospital_id, volume_kb, latency_ms, risk_label, risk_type,
    action, response_status, is_trusted_line, dest_platform
plus the 13 clinical-context columns that the downstream generator
`archive/historical/recent_20260721/generate_behavior_logs.py` adds, so the
rebuilt corpus is a drop-in replacement for both consumers.

Rule semantics mirrored here (backend/rules.py):
    R1 minimum_necessary  : risk_label==0 & volume_kb > p99_normal &
                           action not in {bulk_export, nightly_backup}
    R2 cross_border      : dest_platform not in {san_yi} & not is_trusted_line
    R3 unmasked_export   : (action==bulk_export | volume_kb > 5000 |
                            risk_type==data_leak) & not is_trusted_line
    R4 deactivated_login : fixed 0.2% index sample, np.random.RandomState(42)

All design constants that the four rules see are asserted at generation time.
Every Chinese clinical string is read from clinical_config.json (UTF-8), so
this file stays pure ASCII.

Usage:
    python gen_lecheng_logs.py [--out CSV] [--config JSON] [--no-clinical]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "lecheng_security_logs_2024H1.csv"
DEFAULT_CONFIG = HERE / "clinical_config.json"
SPEC_OUT = HERE / "corpus_spec.json"

# --------------------------------------------------------------------------
# 1. Reverse-engineered design spec (numbers quoted from the old artifacts)
# --------------------------------------------------------------------------
N_EVENTS = 800_000
SEED = 42

# Corpus-level targets (results/metrics_live.json -> summary)
N_ANOMALY = 40_000
P99_TARGET = 534.76515
R1_TOTAL = 7_569
R2_TOTAL = 0
R3_TOTAL = 5_974
R4_RATE = 0.002
RULE_HITS_TOTAL = 15_143
UNIQUE_HIT_EVENTS = 15_117
R1_RUIJIN = 7_327
R1_OTHERS = R1_TOTAL - R1_RUIJIN          # 242
R3_VOLUME_THRESHOLD = 5_000.0
R3_EXCLUDE_ACTIONS = ("bulk_export", "nightly_backup")
CORE_PLATFORMS = ("san_yi",)

# Per-hospital targets (old metrics_live.json -> hospitals[]).
# id, events, anomalies, high_risk(data_leak+brute_force), rule_hits(R1..R4)
HOSPITAL_TARGETS = [
    ("ruijin_hn",   133_874, 6_813, 2_442, 8_629),
    ("shulan",      106_275, 5_355, 1_895, 1_058),
    ("super_hn",     88_466, 4_371, 1_530,   834),
    ("huaxi",        80_404, 4_033, 1_416,   805),
    ("prov_people",  71_333, 3_452, 1_215,   731),
    ("yiling",       71_124, 3_533, 1_270,   699),
    ("hainan_med",   61_943, 3_135, 1_077,   648),
    ("evergrande",   44_730, 2_240,   758,   393),
    ("boai",         44_088, 2_211,   748,   424),
    ("ciming",       35_580, 1_751,   621,   366),
    ("jixin",        35_456, 1_772,   595,   337),
    ("wenlan",       26_727, 1_334,   447,   219),
]
N_HOSP = len(HOSPITAL_TARGETS)

# Time span 2024-01-01 00:00:00 .. 2024-07-01 00:00:00 (181+1 leap days)
TS_START = "2024-01-01T00:00:00"
TS_SPAN_S = 15_724_800                     # 182 days incl. 2024-02-29
TS_MIN_S = 8                               # old corpus: 2024-01-01 00:00:08
TS_MAX_S = TS_SPAN_S - 17                  # old corpus: 2024-06-30 23:59:43

# train_lstm.py: split = int(n*0.8) = 640000, window = 24 -> labels[640023:]
TEST_POSITIVES = 7_981
TRAIN_SPLIT = 640_000
WINDOW = 24

# R4 / rule-overlap bookkeeping.
# #normals strictly above p99 is fixed by the quantile definition:
#   pos = 0.99*(n_normal-1); lo = floor(pos); n_above = n_normal-(lo+1)
# so R1 = n_above - (above-p99 normals carrying an R1-excluded action).
TAIL_MIN = 534.8                           # lowest volume of the "bulk read" stratum
R4_IN_R1 = 18                              # designed R4 x R1 overlap
R4_IN_R3_NORMAL = 8                        # designed R4 x R3 overlap


# --------------------------------------------------------------------------
# 2. helpers
# --------------------------------------------------------------------------
def largest_remainder(weights: list[float], total: int) -> list[int]:
    """Split `total` over `weights` so the parts sum exactly to total."""
    wsum = float(sum(weights))
    raw = [total * w / wsum for w in weights]
    base = [int(math.floor(x)) for x in raw]
    rem = total - sum(base)
    order = sorted(range(len(raw)), key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:rem]:
        base[k] += 1
    return base


def trunc_lognorm(
    rng: np.random.Generator, n: int, mu: float, sigma: float,
    lo: float, hi: float,
) -> np.ndarray:
    """Inverse-CDF sampling of a lognormal truncated to [lo, hi] (no rejection)."""
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    z_lo = (math.log(lo) - mu) / sigma
    z_hi = (math.log(hi) - mu) / sigma
    c_lo, c_hi = ndtr(z_lo), ndtr(z_hi)
    assert 0.0 < c_lo < c_hi < 1.0, (lo, hi, mu, sigma)
    u = c_lo + rng.random(n) * (c_hi - c_lo)
    u = np.clip(u, 1e-15, 1.0 - 1e-15)
    return np.exp(mu + sigma * ndtri(u))


def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


# --------------------------------------------------------------------------
# 3. corpus assembly
# --------------------------------------------------------------------------
def build(cfg: dict) -> tuple[pd.DataFrame, dict]:
    t_start = time.time()
    hosp_cfg = cfg["hospitals"]
    hids = [h["id"] for h in hosp_cfg]
    assert hids == [t[0] for t in HOSPITAL_TARGETS], "hospital id order drift"

    events_t = np.array([t[1] for t in HOSPITAL_TARGETS], dtype=np.int64)
    anom_t = np.array([t[2] for t in HOSPITAL_TARGETS], dtype=np.int64)
    high_t = np.array([t[3] for t in HOSPITAL_TARGETS], dtype=np.int64)
    hits_t = np.array([t[4] for t in HOSPITAL_TARGETS], dtype=np.int64)
    assert events_t.sum() == N_EVENTS
    assert anom_t.sum() == N_ANOMALY
    assert high_t.sum() == 14_014, high_t.sum()
    assert hits_t.sum() == RULE_HITS_TOTAL
    normal_t = events_t - anom_t
    n_normal = int(normal_t.sum())

    # ---- exact p99 -> exact |{normal : volume > p99}| --------------------
    pos = 0.99 * (n_normal - 1)
    q_lo = int(math.floor(pos))
    q_frac = pos - q_lo
    n_above_p99 = n_normal - (q_lo + 1)
    # p99 = A + frac*(TAIL_MIN - A)  =>  A = (target - frac*TAIL_MIN)/(1-frac)
    vol_cap = (P99_TARGET - q_frac * TAIL_MIN) / (1.0 - q_frac)
    p99_rebuilt = vol_cap + q_frac * (TAIL_MIN - vol_cap)
    assert abs(p99_rebuilt - P99_TARGET) < 1e-9, p99_rebuilt
    n_excluded_above = n_above_p99 - R1_TOTAL          # 31
    assert n_excluded_above == 31, n_excluded_above

    # ---- R4 index sample (must equal backend/rules.py apply_rules) --------
    r4_k = int(round(N_EVENTS * R4_RATE))
    r4_idx = np.random.RandomState(SEED).choice(N_EVENTS, size=r4_k, replace=False)
    is_r4 = np.zeros(N_EVENTS, dtype=bool)
    is_r4[r4_idx] = True

    rng = np.random.default_rng(SEED)

    # ---- hospital assignment, permuted over the row index ----------------
    hid = np.repeat(np.arange(N_HOSP, dtype=np.int8), events_t)
    rng.shuffle(hid)
    test_block = np.arange(TRAIN_SPLIT + WINDOW - 1, N_EVENTS)   # labels seen by the test split
    in_test = np.zeros(N_EVENTS, dtype=bool)
    in_test[test_block] = True

    # ---- R1 per hospital: ruijin_hn 7327, the other 11 share 242 ---------
    others = [h for h in range(N_HOSP) if h != 0]
    r1_per_h = [0] * N_HOSP
    r1_per_h[0] = R1_RUIJIN
    alloc = largest_remainder([float(normal_t[h]) for h in others], R1_OTHERS)
    for h, v in zip(others, alloc):
        r1_per_h[h] = v
    assert sum(r1_per_h) == R1_TOTAL

    # ---- anomalies: exact per-hospital counts, exact test-block count -----
    test_alloc = largest_remainder(
        [float((hid[test_block] == h).sum()) for h in range(N_HOSP)], TEST_POSITIVES
    )
    is_anom = np.zeros(N_EVENTS, dtype=bool)
    anom_per_h = []
    for h in range(N_HOSP):
        posh = np.flatnonzero(hid == h)
        posh = posh[~is_r4[posh]]          # keep the 1600 R4 rows out of the anomaly set
        pt = posh[in_test[posh]]
        pr = posh[~in_test[posh]]
        t_h, r_h = test_alloc[h], int(anom_t[h])
        assert t_h <= len(pt) and (r_h - t_h) <= len(pr), h
        pick = np.concatenate([
            rng.choice(pt, size=t_h, replace=False),
            rng.choice(pr, size=r_h - t_h, replace=False),
        ])
        is_anom[pick] = True
        anom_per_h.append(len(pick))
    assert anom_per_h == anom_t.tolist()
    assert int(is_anom[test_block].sum()) == TEST_POSITIVES
    assert not (is_anom & is_r4).any()

    # ---- R3 budget per hospital: hits_h - R4_h - R1_h ---------------------
    r4_per_h = np.array([int(((hid == h) & is_r4).sum()) for h in range(N_HOSP)])
    r3_per_h = hits_t - r4_per_h - np.array(r1_per_h)
    assert (r3_per_h > 0).all(), dict(zip(hids, r3_per_h.tolist()))
    assert int(r3_per_h.sum()) == R3_TOTAL
    # data_leak rows are in R3 by definition (risk_type==data_leak & untrusted);
    # keep them a third of the high-risk budget but never above 55% of R3_h.
    leak_per_h, bulk_per_h, bf_per_h, other_per_h = [], [], [], []
    for h in range(N_HOSP):
        leak = min(int(round(high_t[h] / 3.0)), max(1, int(0.55 * r3_per_h[h])))
        leak_per_h.append(leak)
        bulk_per_h.append(int(r3_per_h[h] - leak))          # normal untrusted bulk_export
        bf_per_h.append(int(high_t[h] - leak))              # brute_force
        other_per_h.append(int(anom_t[h] - high_t[h]))      # api_scan/sync_conflict/replay
    assert sum(leak_per_h) + sum(bulk_per_h) == R3_TOTAL
    assert sum(bf_per_h) + sum(leak_per_h) == int(high_t.sum())
    assert sum(other_per_h) + sum(bf_per_h) + sum(leak_per_h) == N_ANOMALY

    # ---- row groups ------------------------------------------------------
    # 0 base normal | 1 tail R1 | 2 tail backup | 3 R3 normal | 4 leak
    # 5 brute_force | 6 other anomaly
    grp = np.zeros(N_EVENTS, dtype=np.int8)
    pool = np.flatnonzero(~is_anom)

    # designated R4 x R1 rows (ruijin gets 6, the rest 12)
    rem_r4 = np.flatnonzero(is_r4)
    d_r1 = np.concatenate([
        rng.choice(rem_r4[hid[rem_r4] == 0], size=6, replace=False),
        rng.choice(rem_r4[hid[rem_r4] != 0], size=R4_IN_R1 - 6, replace=False),
    ])
    for h in range(N_HOSP):
        assert (hid[d_r1] == h).sum() <= r1_per_h[h], h
    rem_r4 = np.setdiff1d(rem_r4, d_r1)
    # designated R4 x R3-normal rows: one per hospital with the largest R3 budget
    big = sorted(range(N_HOSP), key=lambda h: (-bulk_per_h[h], h))[:R4_IN_R3_NORMAL]
    d_r3 = []
    for h in big:
        cand = rem_r4[hid[rem_r4] == h]
        assert len(cand) > 0, h
        d_r3.append(cand[0])
        assert bulk_per_h[h] >= 1, h
    d_r3 = np.array(sorted(d_r3), dtype=np.int64)

    used = np.zeros(N_EVENTS, dtype=bool)
    grp[d_r1] = 1
    grp[d_r3] = 3
    # Every R4-sampled index is reserved: it may only reach R1/R3 through the
    # designated overlap design above, so that the R4 x (R1|R3) overlap -- and
    # therefore unique_hit_events -- is exact instead of incidental.
    used[is_r4] = True
    used[is_anom] = True
    used[d_r1] = True
    used[d_r3] = True

    def take(n_needed: int, hospital: int | None, tag: str) -> np.ndarray:
        cand = pool[~used[pool]]
        if hospital is not None:
            cand = cand[hid[cand] == hospital]
        assert len(cand) >= n_needed, (tag, hospital, len(cand), n_needed)
        return rng.choice(cand, size=n_needed, replace=False)

    for h in range(N_HOSP):
        need = r1_per_h[h] - int((hid[d_r1] == h).sum())
        idx = take(need, h, "tailA")
        grp[idx] = 1
        used[idx] = True
    assert int((grp == 1).sum()) == R1_TOTAL
    idx_b = take(n_excluded_above, None, "tailB")
    grp[idx_b] = 2
    used[idx_b] = True
    for h in range(N_HOSP):
        need = bulk_per_h[h] - int((hid[d_r3] == h).sum())
        idx = take(need, h, "r3normal")
        grp[idx] = 3
        used[idx] = True
    assert int((grp == 3).sum()) == int(sum(bulk_per_h))

    # anomaly archetypes are assigned TO the anomaly rows themselves
    leak_i, bf_i, oth_i = [], [], []
    for h in range(N_HOSP):
        idx_h = rng.permutation(np.flatnonzero((hid == h) & is_anom))
        a, b = leak_per_h[h], bf_per_h[h]
        assert a + b + other_per_h[h] == len(idx_h), h
        leak_i.append(idx_h[:a])
        bf_i.append(idx_h[a:a + b])
        oth_i.append(idx_h[a + b:])
    leak_idx = np.concatenate(leak_i)
    bf_idx = np.concatenate(bf_i)
    oth_idx = np.concatenate(oth_i)
    grp[leak_idx] = 4
    grp[bf_idx] = 5
    grp[oth_idx] = 6
    used[leak_idx] = True
    used[bf_idx] = True
    used[oth_idx] = True
    assert int((grp == 4).sum()) == sum(leak_per_h)
    assert int((grp == 5).sum()) == sum(bf_per_h)
    assert int((grp == 6).sum()) == sum(other_per_h)
    assert int((is_anom & (grp >= 4)).sum()) == N_ANOMALY
    n_base = int((grp == 0).sum())
    assert n_base + n_above_p99 + int((grp == 3).sum()) == n_normal
    assert n_base + int((grp != 0).sum()) == N_EVENTS
    diag = {f"grp{g}": int((grp == g).sum()) for g in range(7)}
    print("[design] row archetypes: base=%d tail_R1=%d tail_backup=%d r3_normal=%d "
          "leak=%d brute=%d other_anom=%d | anomalies=%d | R4-overlap design "
          "R1=%d R3=%d"
          % (diag["grp0"], diag["grp1"], diag["grp2"], diag["grp3"], diag["grp4"],
             diag["grp5"], diag["grp6"], int(is_anom.sum()),
             R4_IN_R1, R4_IN_R3_NORMAL),
          flush=True)

    # ---- volume_kb -------------------------------------------------------
    # hospital-specific lognormal bulk. This per-hospital shape difference is
    # the documented "hospital-identifying, not compliance-measuring" quirk of
    # R1: ruijin_hn owns essentially the whole heavy tail.
    scale = np.array([h["scale"] for h in hosp_cfg], dtype=np.float64)
    mu_h = 2.26 + 0.30 * np.log10(scale / 1000.0)
    sig_h = 0.78 + 0.10 * ((np.arange(N_HOSP) * 7) % 5) / 4.0
    mu_h[0] += 0.05                                  # ruijin_hn slightly fatter bulk
    vol = np.zeros(N_EVENTS, dtype=np.float64)
    is_base = grp == 0
    for h in range(N_HOSP):
        m = is_base & (hid == h)
        vol[m] = trunc_lognorm(rng, int(m.sum()), mu_h[h], sig_h[h], 0.5,
                               vol_cap - 0.02)
    # tail A: bulk PACS/EMR reads, per-hospital, hard floor just above p99
    m1 = np.flatnonzero(grp == 1)
    for h in range(N_HOSP):
        m = m1[hid[m1] == h]
        if len(m):
            vol[m] = trunc_lognorm(rng, len(m), 6.45 + 0.22 * math.log10(scale[h] / 1000.0),
                                   0.88, TAIL_MIN + 0.011, 4_999.0)
    # tail B: 31 oversized nightly backups, intranet line only
    m2 = np.flatnonzero(grp == 2)
    vol[m2] = np.round(rng.uniform(6_000.0, 30_000.0, len(m2)), 2)
    # R3 normals: masked exports leaving on an untrusted line, small volume
    m3 = np.flatnonzero(grp == 3)
    vol[m3] = np.round(rng.uniform(40.0, 480.0, len(m3)), 2)
    # anomalies
    m4 = np.flatnonzero(grp == 4)
    vol[m4] = np.round(trunc_lognorm(rng, len(m4), 9.35, 0.52, 5_200.0, 42_000.0), 2)
    m5 = np.flatnonzero(grp == 5)
    vol[m5] = np.round(trunc_lognorm(rng, len(m5), 3.15, 0.95, 0.6, 780.0), 2)
    m6 = np.flatnonzero(grp == 6)
    vol[m6] = np.round(trunc_lognorm(rng, len(m6), 3.55, 1.05, 0.6, 1_450.0), 2)
    vol = np.round(vol, 2)
    # Exact p99 anchors, applied AFTER rounding so the two boundary values keep
    # their full double precision: base max = vol_cap, tail min = TAIL_MIN.
    m0 = np.flatnonzero(is_base)
    vol[m0[np.argmax(vol[m0])]] = vol_cap
    vol[m1[np.argmin(vol[m1])]] = TAIL_MIN
    _nv = np.sort(vol[~is_anom])
    print("[design-vol] p99 anchors: base_max=%.10f tail_min=%.4f | sorted normals "
          "at 752397..752402 = %s" % (vol_cap, TAIL_MIN,
                                     [round(float(x), 4) for x in _nv[752_397:752_403]]),
          flush=True)

    # ---- latency_ms: base lognormal x mild volume coupling ---------------
    lat = trunc_lognorm(rng, N_EVENTS, 3.34, 0.52, 0.6, 400.0)
    lat = lat * (1.0 + 0.08 * np.log10(1.0 + vol))
    lat = np.round(np.minimum(lat, 900.0), 2)

    # ---- action / risk_type / trusted / dest / status --------------------
    action = np.full(N_EVENTS, "api_query", dtype=object)
    action[grp == 3] = "bulk_export"
    action[grp == 4] = "bulk_export"
    action[(grp == 2)] = "nightly_backup"
    p = rng.random(N_EVENTS)
    b_mask = is_base & (p >= 0.926) & (p < 0.965)
    action[b_mask] = "nightly_backup"
    e_mask = is_base & (p >= 0.965)
    action[e_mask] = "bulk_export"

    risk_type = np.full(N_EVENTS, "normal", dtype=object)
    oth = m6
    risk_type[oth] = rng.choice(
        np.array(["api_scan", "sync_conflict", "replay"], dtype=object), size=len(oth)
    )
    risk_type[m5] = "brute_force"
    risk_type[m4] = "data_leak"

    trusted = np.zeros(N_EVENTS, dtype=bool)
    trusted[b_mask | e_mask] = True                    # backups + masked exports: intranet
    trusted[grp == 2] = True
    u = rng.random(N_EVENTS)
    trusted[is_base & (action == "api_query")] = u[is_base & (action == "api_query")] < 0.80
    trusted[grp == 1] = u[grp == 1] < 0.55
    trusted[grp == 5] = u[grp == 5] < 0.18
    trusted[grp == 6] = u[grp == 6] < 0.25
    trusted[grp == 3] = False
    trusted[grp == 4] = False

    # R2 = 0 is an enforced design invariant: every flow that leaves the
    # hospital network terminates on the domestic core platform `san_yi`;
    # the other platforms are only reachable over a trusted intranet line.
    plat_ids = list(cfg["platforms"].keys())
    plat_w = np.array([0.45 if cfg["platforms"][p]["is_core"] else 0.11
                       for p in plat_ids], dtype=np.float64)
    plat_w = plat_w / plat_w.sum()
    dest = np.array(plat_ids, dtype=object)[
        rng.choice(len(plat_ids), size=N_EVENTS, p=plat_w)]
    dest[~trusted] = "san_yi"

    status = np.full(N_EVENTS, 200, dtype=np.int64)
    rmap = cfg["response_status"]
    for rt, codes in rmap.items():
        m = risk_type == rt
        status[m] = rng.choice(np.array(codes, dtype=np.int64), size=int(m.sum()))
    risk_label = is_anom.astype(np.int64)

    # ---- timestamps: uniform over 2024H1, forced global min/max ----------
    sec = rng.integers(0, TS_SPAN_S, size=N_EVENTS)
    night = np.flatnonzero(action == "nightly_backup")
    sec[night] = sec[night] - (sec[night] % 86_400) + rng.integers(0, 6 * 3_600, size=len(night))
    free = np.flatnonzero(action != "nightly_backup")
    sec[free[np.argmin(sec[free])]] = TS_MIN_S
    sec[free[np.argmax(sec[free])]] = TS_MAX_S
    ts = (np.datetime64(TS_START) + sec.astype("timedelta64[s]"))
    ts_str = np.char.replace(np.datetime_as_string(ts, unit="s"), "T", " ")

    df = pd.DataFrame({
        "timestamp": ts_str,
        "hospital_id": np.array(hids, dtype=object)[hid],
        "volume_kb": vol,
        "latency_ms": lat,
        "risk_label": risk_label,
        "risk_type": risk_type,
        "action": action,
        "response_status": status,
        "is_trusted_line": trusted,
        "dest_platform": dest,
    })

    if clinical:
        add_clinical_context(df, cfg, grp, hid, rng)

    df = df.reset_index(drop=True)
    spec = {
        "rows": N_EVENTS,
        "anomaly": int(risk_label.sum()),
        "n_normal": n_normal,
        "p99_target": P99_TARGET,
        "p99_reconstructed": float(p99_rebuilt),
        "n_above_p99_normal": n_above_p99,
        "n_above_p99_with_excluded_action": n_excluded_above,
        "r1_per_hospital": dict(zip(hids, r1_per_h)),
        "r3_per_hospital": dict(zip(hids, r3_per_h.tolist())),
        "r4_per_hospital": dict(zip(hids, r4_per_h.tolist())),
        "leak_per_hospital": dict(zip(hids, leak_per_h)),
        "brute_force_per_hospital": dict(zip(hids, bf_per_h)),
        "other_anomaly_per_hospital": dict(zip(hids, other_per_h)),
        "r3_normal_bulk_export_per_hospital": dict(zip(hids, bulk_per_h)),
        "test_positives": int(is_anom[test_block].sum()),
        "build_seconds": round(time.time() - t_start, 2),
    }
    return df, spec


def add_clinical_context(df: pd.DataFrame, cfg: dict, grp: np.ndarray,
                         hid: np.ndarray, rng: np.random.Generator) -> None:
    """Attach the 13 clinical-context columns used by generate_behavior_logs.py."""
    random.seed(SEED)
    hosp_cfg = cfg["hospitals"]
    hids = [h["id"] for h in hosp_cfg]
    roles = cfg["roles"]
    role_en = cfg["role_en"]
    pharm, it_dept = cfg["pharmacy_dept"], cfg["it_dept"]
    sens_pool = cfg["patient_sensitivity"]
    clin = cfg["clinical_actions"]
    r2c = cfg["risk_to_clinical"]

    n = len(df)
    vol = df["volume_kb"].to_numpy()
    risk = df["risk_type"].to_numpy()
    action = df["action"].to_numpy()

    # ---- pools: same construction rule as the original generator ---------
    user_ids: list[list[str]] = []
    user_role: list[list[str]] = []
    user_dept: list[list[str]] = []
    pat_ids: list[list[str]] = []
    pat_dept: list[list[str]] = []
    pat_sens: list[list[str]] = []
    pat_by_dept: list[dict[str, np.ndarray]] = []
    for h in hosp_cfg:
        depts = h["depts"]
        npat = random.randint(h["scale"] // 2, h["scale"])
        pids, pdep, psen = [], [], []
        for i in range(npat):
            pids.append(f"PAT_{h['id'][:4].upper()}_{i + 1:06d}")
            pdep.append(random.choice(depts))
            psen.append(random.choice(sens_pool))
        pat_ids.append(pids)
        pat_dept.append(pdep)
        pat_sens.append(psen)
        dmap: dict[str, np.ndarray] = {}
        for d in sorted(set(depts)):
            dmap[d] = np.array([i for i, x in enumerate(pdep) if x == d], dtype=np.int64)
        pat_by_dept.append(dmap)

        users, urole, udept = [], [], []
        uidc = 0
        specs = [
            (roles[0], max(1, len(depts))), (roles[1], max(1, len(depts))),
            (roles[2], max(2, len(depts) * 3)), (roles[3], max(2, len(depts) * 2)),
            (roles[4], max(1, len(depts) * 2)), (roles[5], max(1, len(depts))),
            (roles[6], max(3, len(depts) * 5)),
            (roles[7], max(1, 3) if pharm in depts else 0),
            (roles[8], random.randint(3, 8)),
        ]
        for role, cnt in specs:
            for _ in range(cnt):
                uidc += 1
                if role == roles[7] and pharm in depts:
                    d = pharm
                elif role == roles[8]:
                    d = it_dept
                else:
                    pool = [x for x in depts if x != pharm] or depts
                    d = random.choice(pool)
                users.append(f"{h['id'][:3].upper()}_{role_en[role]}_{uidc:04d}")
                urole.append(role)
                udept.append(d)
        user_ids.append(users)
        user_role.append(urole)
        user_dept.append(udept)

    # ---- per-row role requirement (original mapping) ---------------------
    want_role = np.full(n, "", dtype=object)
    want_role[action == "nightly_backup"] = roles[8]
    want_role[risk == "data_leak"] = roles[2]
    want_role[risk == "brute_force"] = roles[4]
    for rt, key in r2c.items():
        want_role[risk == rt] = clin[key]["role"]
    want_role[want_role == ""] = "*"                  # any user

    uid_idx = np.zeros(n, dtype=np.int64)
    dept_code = np.zeros(n, dtype=np.int64)
    dept_names: list[list[str]] = []
    pat_pick = np.full(n, -1, dtype=np.int64)
    cyclic_state: dict[tuple[int, str], int] = {}

    for h, hcfg in enumerate(hosp_cfg):
        rows = np.flatnonzero(hid == h)
        depts = hcfg["depts"] + [it_dept]
        dnames = depts
        dept_names.append(dnames)
        d_index = {d: i for i, d in enumerate(dnames)}
        du = np.array(user_dept[h], dtype=object)
        ur = np.array(user_role[h], dtype=object)
        # candidate user positions per role
        by_role = {r: np.flatnonzero(ur == r) for r in set(ur.tolist())}
        for r in (roles[2], roles[4], roles[8]):
            assert len(by_role.get(r, [])) > 0, (hcfg["id"], r)
        any_users = np.arange(len(ur), dtype=np.int64)
        pos_want = want_role[rows]
        pick = np.empty(len(rows), dtype=np.int64)
        ucodes = np.array([d_index[d] for d in du], dtype=np.int64)
        for r in np.unique(pos_want).tolist():
            m = pos_want == r
            cand = any_users if r == "*" else by_role[r]
            assert len(cand) > 0, (hcfg["id"], r)
            pick[m] = cand[rng.integers(0, len(cand), size=int(m.sum()))]
        uid_idx[rows] = pick
        dept_code[rows] = ucodes[pick]
        # department actually used (IT admins act for a real clinical dept)
        used_dept = dept_code[rows].copy()
        it_code = d_index[it_dept]
        m_it = used_dept == it_code
        n_real = len(hcfg["depts"])
        used_dept[m_it] = rng.integers(0, n_real, size=int(m_it.sum()))
        dept_code[rows] = used_dept
        # ---- patient assignment (cyclic -> every patient >= 3 accesses) ---
        want_pat = np.ones(len(rows), dtype=bool)
        want_pat[action[rows] == "nightly_backup"] = False
        for key in ("api_probe", "data_sync_fail"):
            want_pat[risk[rows] == key] = False
        rp = np.flatnonzero(want_pat)
        if len(rp):
            same = rng.random(len(rp)) < 0.90          # same-dept access
            dsel = used_dept[rp]
            out = np.empty(len(rp), dtype=np.int64)
            for d_code in np.unique(dsel):
                d = dnames[d_code]
                pool = pat_by_dept[h].get(d)
                if pool is None or len(pool) == 0:
                    pool = np.arange(len(pat_ids[h]), dtype=np.int64)
                m = dsel == d_code
                want_same = m & same
                want_any = m & ~same
                for tag, mask, cand, kc in (
                    ("dept", want_same, pool, d_code),
                    ("all", want_any, np.arange(len(pat_ids[h])), 0),
                ):
                    if not mask.any():
                        continue
                    key2 = (h, tag, int(kc))
                    start = cyclic_state.get(key2, 0)
                    order = rng.permutation(len(cand))
                    cnt = int(mask.sum())
                    seq = order[(start + np.arange(cnt)) % len(cand)]
                    cyclic_state[key2] = start + cnt
                    out[mask] = cand[seq]
            pat_pick[rows[rp]] = out

    # ---- materialise the string columns ---------------------------------
    uids = np.empty(n, dtype=object)
    uroles = np.empty(n, dtype=object)
    udep = np.empty(n, dtype=object)
    hname = np.empty(n, dtype=object)
    dept_used = np.empty(n, dtype=object)
    pids = np.full(n, "", dtype=object)
    psens = np.full(n, "", dtype=object)
    for h, hcfg in enumerate(hosp_cfg):
        rows = np.flatnonzero(hid == h)
        dnames = dept_names[h]
        uids[rows] = np.array(user_ids[h], dtype=object)[uid_idx[rows]]
        uroles[rows] = np.array(user_role[h], dtype=object)[uid_idx[rows]]
        udep[rows] = np.array(user_dept[h], dtype=object)[uid_idx[rows]]
        dept_used[rows] = np.array(dnames, dtype=object)[dept_code[rows]]
        hname[rows] = hcfg["name"]
        pp = pat_pick[rows]
        m = pp >= 0
        if m.any():
            pids[rows[m]] = np.array(pat_ids[h], dtype=object)[pp[m]]
            psens[rows[m]] = np.array(pat_sens[h], dtype=object)[pp[m]]

    # pharmacist-only-in-pharmacy / IT-admins-do-backups invariants
    assert all(
        (uroles[i] != roles[7]) or (udep[i] == pharm) for i in np.flatnonzero(uroles == roles[7])
    )
    nk = np.flatnonzero(action == "nightly_backup")
    assert (uroles[nk] == roles[8]).all(), "nightly backups must be run by IT admins"
    accessed = pd.Series(pids[pids != ""]).value_counts()
    assert int(accessed.min()) >= 3, int(accessed.min())

    # clinical action <-> volume correspondence (original mapping)
    ca = np.empty(n, dtype=object)
    api = action == "api_query"
    ca[api & (vol <= 5)] = "query_pharmacy_stock"
    ca[api & (vol > 5) & (vol <= 20)] = "view_nursing_record"
    ca[api & (vol > 20) & (vol <= 200)] = "read_emr"
    ca[api & (vol > 200) & (vol <= 5_000)] = "read_lab_report"
    ca[api & (vol > 5_000)] = "read_imaging_report"
    ca[action == "nightly_backup"] = "nightly_backup"
    ca[(action == "bulk_export") & (risk == "normal")] = "masked_export"
    for rt, key in r2c.items():
        ca[risk == rt] = key
    category = np.array([clin[k]["category"] for k in ca], dtype=object)
    label = np.array([clin[k]["label"] for k in ca], dtype=object)

    # reason columns
    is_ab = risk != "normal"
    reason = np.full(n, "", dtype=object)
    for rt, key in r2c.items():
        reason[risk == rt] = clin[key]["reason"]
    choices = {k: v["reasons"] for k, v in clin.items() if v.get("reasons")}
    acc_reason = np.full(n, "", dtype=object)
    nr = np.flatnonzero(~is_ab)
    for k in np.unique(ca[nr]).tolist():
        m = nr[ca[nr] == k]
        pool = choices[k]
        acc_reason[m] = np.array(pool, dtype=object)[rng.integers(len(pool), size=len(m))]

    records = np.maximum(1, (vol / rng.uniform(0.05, 3.0, size=n)).astype(np.int64))
    subnet = np.array([h["subnet"] for h in hosp_cfg], dtype=object)[hid]
    t1 = rng.integers(1, 255, size=n)
    t2 = rng.integers(1, 255, size=n)
    t3 = rng.integers(27, 211, size=n)
    tr = df["is_trusted_line"].to_numpy()
    src_ip = np.empty(n, dtype=object)
    src_ip[tr] = [f"{subnet[i]}.{t1[i]}.{t2[i]}" for i in np.flatnonzero(tr)]
    src_ip[~tr] = [f"{t3[i]}.{t1[i]}.{t2[i]}.{rng.integers(1, 255)}"
                   for i in np.flatnonzero(~tr)]
    suf = np.array([u[-6:] for u in uids], dtype=object)
    hid2 = np.array([h[-2:] for h in hids], dtype=object)[hid]
    ser = rng.integers(100, 1000, size=n)
    sess = np.array([f"SESS_{hid2[i]}{suf[i]}{ser[i]}" for i in range(n)], dtype=object)

    df["hospital_name"] = hname
    df["department"] = dept_used
    df["user_id"] = uids
    df["user_role"] = uroles
    df["patient_id"] = pids
    df["patient_sensitivity"] = psens
    df["clinical_action"] = label
    df["access_reason"] = acc_reason
    df["anomaly_reason"] = reason
    df["data_category"] = category
    df["records_affected"] = pd.Series(records).astype(str).to_numpy()
    df["source_ip"] = src_ip
    df["session_id"] = sess


# --------------------------------------------------------------------------
# 4. rule re-implementation (mirrors backend/rules.py) + generation-time asserts
# --------------------------------------------------------------------------
def audit(df: pd.DataFrame) -> dict:
    vol = df["volume_kb"].astype(float)
    lbl = df["risk_label"].astype(int)
    act = df["action"].astype(str)
    rt = df["risk_type"].astype(str)
    dest = df["dest_platform"].astype(str)
    trusted = df["is_trusted_line"].map(_as_bool)

    normal = vol[lbl == 0]
    p99 = float(normal.quantile(0.99))
    r1 = (lbl == 0) & (vol > p99) & (~act.isin(R3_EXCLUDE_ACTIONS))
    r2 = (~dest.isin(CORE_PLATFORMS)) & (~trusted)
    r3 = ((act == "bulk_export") | (vol > R3_VOLUME_THRESHOLD) |
          (rt == "data_leak")) & (~trusted)
    r4 = pd.Series(False, index=df.index)
    n = len(df)
    k = int(round(n * R4_RATE))
    rng = np.random.RandomState(SEED)
    r4.iloc[rng.choice(n, size=min(k, n), replace=False)] = True

    hits = int(r1.sum() + r2.sum() + r3.sum() + r4.sum())
    unique = int((r1 | r2 | r3 | r4).sum())
    top = df.loc[r1, "hospital_id"].value_counts()
    high = int(((lbl == 1) & rt.isin(["data_leak", "brute_force"])).sum())
    return {
        "events": int(n),
        "anomaly": int((lbl == 1).sum()),
        "high_risk": high,
        "p99_volume_kb": p99,
        "R1": int(r1.sum()),
        "R2": int(r2.sum()),
        "R3": int(r3.sum()),
        "R4": int(r4.sum()),
        "rule_hits": hits,
        "unique_hit_events": unique,
        "r1_top_hospital": str(top.index[0]),
        "r1_top_hospital_hits": int(top.iloc[0]),
        "r1_top_hospital_share": float(top.iloc[0] / max(int(r1.sum()), 1)),
    }


def _as_bool(value) -> bool:
    if isinstance(value, (bool,)):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse is_trusted_line: {value!r}")


def assert_targets(a: dict) -> None:
    checks = [
        ("events", a["events"], N_EVENTS),
        ("anomaly", a["anomaly"], N_ANOMALY),
        ("high_risk", a["high_risk"], 14_014),
        ("R1", a["R1"], R1_TOTAL),
        ("R2", a["R2"], R2_TOTAL),
        ("R3", a["R3"], R3_TOTAL),
        ("R4", a["R4"], 1_600),
        ("rule_hits", a["rule_hits"], RULE_HITS_TOTAL),
        ("unique_hit_events", a["unique_hit_events"], UNIQUE_HIT_EVENTS),
        ("r1_top_hospital", a["r1_top_hospital"], "ruijin_hn"),
        ("r1_top_hospital_hits", a["r1_top_hospital_hits"], R1_RUIJIN),
    ]
    bad = []
    for name, got, want in checks:
        if isinstance(want, float):
            ok = abs(got - want) < 1e-4
        else:
            ok = got == want
        if not ok:
            bad.append(f"{name}: got {got!r} want {want!r}")
    if abs(a["p99_volume_kb"] - P99_TARGET) > 1e-4:
        bad.append(f"p99: got {a['p99_volume_kb']!r} want {P99_TARGET!r}")
    if bad:
        raise AssertionError("generation-time target mismatch:\n  " + "\n  ".join(bad))
    print("[assert] all generation-time targets reproduced")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--no-clinical", action="store_true",
                    help="emit only the 10 pipeline columns (smaller CSV)")
    ap.add_argument("--spec", default=str(SPEC_OUT))
    args = ap.parse_args()

    global clinical
    clinical = not args.no_clinical

    cfg = load_config(Path(args.config))
    df, spec = build(cfg)

    order = ["timestamp", "hospital_id", "volume_kb", "latency_ms", "risk_label",
             "risk_type", "action", "response_status", "is_trusted_line",
             "dest_platform",
             "hospital_name", "department", "user_id", "user_role", "patient_id",
             "patient_sensitivity", "clinical_action", "access_reason",
             "anomaly_reason", "data_category", "records_affected", "source_ip",
             "session_id"]
    df = df[[c for c in order if c in df.columns]]
    assert len(df) == N_EVENTS

    print("[audit] in-memory aggregate check")
    a = audit(df)
    assert_targets(a)
    for k, v in a.items():
        print(f"    {k} = {v}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write] {out}")
    df.to_csv(out, index=False, encoding="utf-8", lineterminator="\n")
    mb = out.stat().st_size / 1024 / 1024
    print(f"[write] {len(df):,} rows x {len(df.columns)} cols, {mb:.1f} MB")

    print("[audit] re-read aggregate check")
    back = pd.read_csv(out)
    b = audit(back)
    assert_targets(b)
    spec["achieved"] = b
    spec["columns"] = list(df.columns)
    spec["csv"] = str(out)
    spec["csv_mb"] = round(mb, 1)
    spec["pandas"] = pd.__version__
    spec["numpy"] = np.__version__
    spec["python"] = sys.version.split()[0]
    spec["seed"] = SEED
    Path(args.spec).write_text(json.dumps(spec, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    print(f"[write] {args.spec}")
    return 0


clinical = True

if __name__ == "__main__":
    raise SystemExit(main())
