# 语料与生成器

本目录是第 7 章平台口径语料的**可复现来源**。论文与报告引用第 7 章数字时，
一律以此处语料为准。

## 文件

| 文件 | 大小 | 说明 |
|---|---:|---|
| `gen_lecheng_logs.py` | 36,580 B | 生成器，868 行，`SEED = 42`，5 处种子调用 |
| `clinical_config.json` | 6,078 B | 生成器唯一外部依赖（临床/机构配置） |
| `make_pipeline_inputs.py` | 2,676 B | 23 列 → 10 列投影（行值逐字复制，只drop 列） |
| `corpus_card.json` | 6,524 B | 语料统计卡：逐院/逐类型分位数、特征重叠率 |
| `csv_sha256.txt` | 64 B | 交付方记录的语料指纹 |

## 语料本体

```
../lecheng_security_logs_2024H1_pipeline10.csv
63,884,514 B   800,000 行 × 10 列
sha256 186d429703780306165c3c59a54dab22c0d39e462c9f525abd6ab036ddd1e6e5
```

10 列即`backend/engine.py` 所需的全部列：
`timestamp, hospital_id, volume_kb, latency_ms, risk_label, risk_type, action, response_status, is_trusted_line, dest_platform`

## 字节级复现

```bash
python data/generator/gen_lecheng_logs.py \
    --out data/lecheng_security_logs_2024H1_pipeline10.csv \
    --no-clinical \
    --config data/generator/clinical_config.json
```

已于 2026-10-03 实测：产出 63,884,514 B，sha256 `186d4297…`，与入库文件**完全一致**。

## 未入库的 23 列全量版

交付方另有 23 列全量版（193,788,645 B，sha256 `d57243f6…`），**不入库**，原因是含四个可标识字段：

| 字段 | 唯一值数 |
|---|---:|
| `patient_id` | 13,910 |
| `user_id` | 1,079 |
| `source_ip` | 582,460 |
| `session_id` | 485,145 |

这四列**没有任何分析用途** —— `corpus_card.json` 中所有统计仅用到 10 列加类别列。若需完整版，用投影脚本从交付目录取：

```bash
python data/generator/make_pipeline_inputs.py --csv <23列版路径>
```

## 已弃用：git 历史版语料

`lecheng_security_logs_2024H1.csv`（61,724,859 B，sha256 `c594baf5…`）曾于 commit `7965070`
入库，现已移除。取回方式：

```bash
git show 7965070:edge-audit-system/data/lecheng_security_logs_2024H1.csv > old.csv
```

弃用理由见 `REPRODUCIBILITY.md` §4.1.2：无法由现存代码重新生成，且其 R3 退化为标签回读。