# 可复现过程记录：乐城边缘审计（合成语料现算）四件套

> 按 AGENTS.md §9「实验可复现性铁律」，本记录与源码、JSON 产物、完整工作流一起交付。
> 本项目为**竞赛演示包**，不是医院生产平台；页面与 JSON 中的数字均由 80 万行合成 CSV 按附录 C 四条确定性规则**现场计算**，不能当作真实医院审计实证。

## 1. 结果

本报告对应的 JSON 产物：`results/metrics_live.json`（本目录）。核心数字：

| 指标 | 值 |
|---|---|
| 语料 | `data/lecheng_security_logs_2024H1_pipeline10.csv`（仓内，合成，80 万行 × 10 列 / 12 院；sha256 `186d429703780306…`；**可由仓内生成器字节级重现**，见 §4.1.1） |
| 事件数 | 800,000 |
| 异常事件（risk_label=1） | 40,000（5.00%） |
| 高风险（data_leak + brute_force） | 14,014（1.75%） |
| 正常流量 volume_kb p99 | 534.76515 |
| R1 最小必要 | 7,569 |
| R2 跨境未脱敏 | 0 |
| R3 未脱敏导出 | 5,974 |
| R4 已注销账号登录（seed=42 抽 0.2%） | 1,600 |
| 规则命中次数（同事件可多命中） | **15,143** |
| 唯一命中事件 | 15,117 |
| 凭证链 | 完整（genesis 全 0，链长 15,143，tip 见 JSON） |
| 凭证链 SM2 签名 | SHA-256 链式散列 + 国密 SM2 签名，验签 40/40 通过（`/api/verify`） |
| 按院 Edge 链 | 12 条，全部 valid，每链 SM2 抽样 8/8 通过（`/api/nodes`） |

构造痕迹说明（与正文 7.3.1 一致）：R1 的 7,569 次命中中 7,327 次（96.8%）集中于 `ruijin_hn` 一家医院——合成语料的 `volume_kb` 分布院别特异，该规则实为「识别医院」而非「测量合规行为」；R2 为 0 是因为全部未脱敏流量的 `dest_platform` 都是 `san_yi`，跨境判据在本语料上不成立。

## 2. 源码

- `backend/rules.py`：四条确定性规则（R1 p99、R2 核心平台、R3 批量导出/阈值/data_leak、R4 固定种子 0.2% 抽样）
- `backend/crypto_signer.py`：国密 SM2 签名器（gmssl），私钥 `data/keys/edge_demo_sm2_private.hex`，自校验+自愈
- `backend/chain.py`：SHA-256 凭证链（genesis 全 0，链式散列）+ SM2 签名（P3 防篡改/不可抵赖）+ 链验真/验断/抽样验签
- `backend/engine.py`：加载 CSV、聚合院统计、摘要、12 条按院 Edge 凭证链
- `backend/app.py`：FastAPI `/api/*` 接口（含 `/api/nodes`、`/api/signatures`、`/api/verify`、`/api/model/infer`）
- `backend/features.py`：5 维确定性事件特征（不依赖全局 scaler，保证可复现）
- `backend/lstm_model.py` + `backend/model_inference.py`：本地 LSTM 风险推理（torch CPU），仅加载 `results/lstm/` 真实产物，无产物时明确报错
- `frontend/`：Vite 看板，`/api` 反代理到 127.0.0.1:3001
- `scripts/precompute_metrics.py`：产出本 JSON 的预计算脚本
- `scripts/train_lstm.py`：训练本地 LSTM，产出 `results/lstm/model.pt` + `metrics_lstm.json` + `model_meta.json` + `params_train.json`
- `tests/test_rules.py`：规则、凭证链、SM2 签名单测（含 80 万行现算断言）

## 3. 可复现过程

运行环境：Linux，Python 3.11，pandas 3.x，numpy 2.x，fastapi/uvicorn（后端），vite 5.4（前端），torch 2.13.0+cpu（本地 LSTM），gmssl 3.2.2（SM2）。随机种子：seed=42（仅 R4 抽样与 LSTM 训练使用，其余规则确定性）。

复现命令：

```bash
# 1) 预计算 JSON（含 SM2 签名链与 12 条节点链）
#    口径必须显式设为 v1：默认 v2 会把 R4 转为基线、不计命中，
#    rule_hits 将得到 13,543 而非 §1 声称的 15,143（产物 meta.ruleset 会记为 v2）。
cd <repo-root>
AUDIT_RULESET=v1 python3 scripts/precompute_metrics.py \
  --csv data/lecheng_security_logs_2024H1_pipeline10.csv \
  --out results/metrics_live.json

# 2) 单测（14 项，5 s；含 80 万行现算断言：p99=534.765、R1=7569、R2=0、
#    R3=5974、R4=1600、合计 15143；SM2 验签；v1/v2 口径与 uniform/stratified
#    抽样各有锁定测试）
python3 -m pytest tests -q

# 3) 本地 LSTM 风险推理训练（真实产出 metrics_lstm.json，约 4 分钟）
python3 scripts/train_lstm.py \
  --csv data/lecheng_security_logs_2024H1_pipeline10.csv \
  --window 24 --hidden 64 --max-train-seqs 120000 --epochs 8 --seed 42

# 4) 启动双服务（前端 5173 反代理 /api → 后端 3001）
bash start.sh
```

> 语料路径用仓内副本而非 `<external-archive>/historical/...`：后者指向已弃用的
> git 历史版语料（sha `c594baf5…`，见 §4.1.2），且现行 `backend/engine.py` /
> `scripts/train_lstm.py` / `tests/test_rules.py` 已在该路径缺失时直接抛
> `FileNotFoundError`，不再静默回落（见 §4.1.4）。

依赖：`pip install --break-system-packages -r requirements.txt` 与 `cd frontend && npm install`。

## 4. 完整工作流

### 4.1 平台口径（第 7 章 / `results/metrics_live.json`）

数据获取 → 80 万行 CSV → 加载 → 四规则现算 → 全局凭证链（SHA-256 链 + SM2 签名）→ 12 条按院 Edge 凭证链 → 本地 LSTM 风险推理（`results/lstm/`）→ 院统计 → 看板 `/api` 现算展示 → JSON 产物。

#### 4.1.1 语料与生成器（2026-10-03 定版）

| 项 | 值 |
|---|---|
| 语料 | `data/lecheng_security_logs_2024H1_pipeline10.csv` |
| 大小 | 63,884,514 B |
| 行× 列 | 800,000 × 10 |
| sha256 | `186d429703780306165c3c59a54dab22c0d39e462c9f525abd6ab036ddd1e6e5` |
| 生成器 | `data/generator/gen_lecheng_logs.py`（868 行，`SEED = 42`，5 处种子调用） |
| 生成器唯一依赖 | `data/generator/clinical_config.json` |
| 语料统计卡 | `data/generator/corpus_card.json` |

**已实测字节级重现**：

```bash
python data/generator/gen_lecheng_logs.py \
    --out data/lecheng_security_logs_2024H1_pipeline10.csv \
    --no-clinical \
    --config data/generator/clinical_config.json
# 产出 63,884,514 B，sha256 186d4297…，与入库文件完全一致
```

`--no-clinical` 直接产出 10 列版（`backend/engine.py` 所需的全部列）。交付方另有23 列全量版（193,788,645 B，sha256 `d57243f6…`），**含 `patient_id` / `user_id` / `source_ip` / `session_id` 四个可标识字段，不入库**；`data/generator/make_pipeline_inputs.py` 是 23 列→10 列的投影脚本（行值逐字复制，只 drop 列）。

`.gitignore` 第 34 行 `*.csv` 全局忽略语料，故以 `git add -f` 强制入库（依 AGENTS §6全量保留原则）。`.gitattributes` 对 `edge-audit-system/data/*.csv` 设 `-text`，禁止行尾转换——本仓 `core.autocrlf=true`，若不管，新 clone 的工作树会拿到 CRLF 版本、sha256 对不上。已用 `file://` 全新 clone 实测验证。

#### 4.1.2 已弃用：git 历史版语料

`data/lecheng_security_logs_2024H1.csv`（61,724,859 B，sha256 `c594baf5…`）曾于 commit `7965070` 入库，**现已移除**，仍可取回：

```bash
git show 7965070:edge-audit-system/data/lecheng_security_logs_2024H1.csv > old.csv
```

**弃用理由（两条，均实测）**：

1. **无法由现存代码重新生成**。历史生成器 `lecheng_fast_simulator.py`（commit `271c5ba`，162 行）未固定种子（`np.random.seed` / `default_rng` / `RandomState` 出现 0 次，`np.random.*` 调用 12 处）；且结构上产不出语料中的 `is_trusted_line=False`（L89 与 L135 对该字段只赋 `True`）。对比之下，09-28 版的 `gen_lecheng_logs.py` 种子固定且已验证可字节级重现。
2. **R3 退化为标签回读**。该版上 R3 的三个条件（`action=='bulk_export'` / `volume>5000` / `risk_type=='data_leak'`）与 `¬is_trusted_line` 选中的**是同一批 5,974 行**，零判别力。09-28 版上三者互相独立：交付方的 `corpus_card.json` 给出判别力证据 `volume_gt_5000_given_anomaly = 0.082025` vs `volume_gt_5000_given_normal = 0.000041`（约 200× 提升）。

两版语料的 §7 指标**完全一致**，故换语料不改变任何已发表数字；改动的是**依据的强度**。

#### 4.1.3 换语料的逐字段对账结果

以 v1 规则集在两版语料上各跑一次 `precompute_metrics.py`，逐字段比对：

| 字段 | 旧（git 版） | 新（09-28 版） | |
|---|---:|---:|:--|
| events | 800,000 | 800,000 | 同 |
| anomaly / high_risk | 40,000 / 14,014 | 40,000 / 14,014 | 同 |
| p99_volume_kb | 534.765150000000**2** | 534.765150000000**1** | 浮点末位噪声 |
| R1 / R2 / R3 / R4 | 7,569 / 0 / 5,974 / 1,600 | 7,569 / 0 / 5,974 / 1,600 | 同 |
| rule_hits | 15,143 | 15,143 | 同 |
| unique_hit_events | 15,117 | 15,117 | 同 |
| r1_top_hospital / hits / share | ruijin_hn / 7,327 / 0.9680274805126172 | 同 | 同 |
| chain.length / valid / hash_ok | 15,143 / true / true | 同 | 同 |
| chain.tip | `ed7af6c7…` ⚠ | `e75c5e48…` | **应异**（末条链节内容不同） |
| hospitals 段逐字段 | — | — | **0 处差异**（12 院全同） |
| signatures.distinct_node_keys | `None` | `12` | **修复** |
| signatures.node_keyring_fingerprints | `{}`（0 个） | 12 个互不相同 | **修复** |

后两项是修复而非差异：旧产物根本没有记录节点级密钥，新产物记录了 12 个互不相同的 SM2 公钥指纹（每次运行重新生成，故跨运行必然不同——这是设计如此，不是不可复现）。

> ⚠ **`chain.tip` 取决于「语料 + 摘要字段清单」两者**，任一变化都会使其刷新。上表「新」列的
> `e75c5e48…` 是 2026-10-03 纳入 `risk_label` 后的 **13 字段**口径实测值（见 §4.6）；「旧」列的
> `ed7af6c7…` 为 **12 字段**口径所测，两格口径不同，仅可读作「两版语料的 tip 必不相同」这一结论，
> 不可互相加减。本表其余各行均与摘要字段清单无关，不受该变更影响。

#### 4.1.4 缺失即报错，不静默回落

`backend/engine.py` / `scripts/train_lstm.py` / `tests/test_rules.py` 三处的语料路径已改为**仓内缺失即抛 `FileNotFoundError`**，并打印恢复命令。此前它们会静默回落到 `<external-archive>/historical/...`，而该路径指向的是**另一份语料**——会导致产物记录「语料名相同、内容不同」的指纹而无人察觉（依 AGENTS §9.4）。`AUDIT_CSV` 环境变量仍可覆盖，用于容器部署时显式指向挂载卷（显式指定不算静默回落）。

### 4.2 实验口径（第 9 章 / `experiment/results/`）

`experiment/` 下 **13 个实验脚本**，语料来源分三类：**8 个不复用平台语料**，由 `adversarial_test.make_corpus()` 现场生成合成语料（`seed=20260930`，`trusted_rate=0.99`）；**3 个直接读平台语料**（`corpus_load_imbalance_test.py` / `sampling_stratification_test.py` / `chain_field_coverage_test.py`）；另有 `crypto_cost_benchmark.py`（无语料，纯微基准）与 `real_data_chain_test.py`（读外部 Excel 工作簿）不属于上述任一口径。因此平台口径与合成口径是**两套生成器、两套统计特征**，差异显著：

| 属性 | 实验语料 `make_corpus`（代码可查） | 平台语料（09-28 交付版，实测） |
|---|---:|---:|
| 节点负载（事件级） | 1.0000×（严格等量） | 5.0089× |
| 节点负载（链级，即命中） | ≈1.0× | **39.4018×** |
| volume_kb | `uniform(1, 12000)` | p50 209 / p99 620（长尾） |
| latency_ms | `uniform(3, 260)` | p50 49 / p99 96 |
| 异常率 | 20%（`i%5==0`） | 5.00%（40,000/800,000） |
| hospital_id | `h00`–`h11` | 真实机构名，按权重分配 |
| 时间分布 | 合成日内单调 | 按小时峰谷比 1.1788（00-05 约 3.7 万、06-23 约 3.2 万）；**未按时间排序** |
| 可否由代码重新生成 | ✅ 固定 seed，可完全复现 | ✅ **可**，见 §4.1.1（字节级重现已验证） |

⚠️ **昼夜效应无法用 `scaling_variance_test.py` 检验**，原因不是语料时间分布，而是该实验的链长旋钮是**位置跨步抽样**：

```python
step = max(1, N // tgt)
sub = hits_all.iloc[::step][:tgt]          # scaling_variance_test.py:287-288
```

`hits_all` 的行序即语料行序，而两版语料**都不是时间有序**（`timestamp.is_monotonic_increasing == False`）。在时间被打乱的序列上跨步等价于随机抽样，故时间结构无法进入该实验。

各实验的复现命令形如：

```bash
python3 experiment/<脚本>.py            # 默认 events/seed 见各脚本 argparse
python3 experiment/corpus_load_imbalance_test.py \
    --csv data/lecheng_security_logs_2024H1_pipeline10.csv
```

### 4.3 实测复现结果（2026-10-02 首次，2026-10-03 在新语料上复验）

以仓库现行 `backend/rules.py` 对恢复出的 80 万行平台语料重算，与本文件 §1 声称值**逐项一致（9/9）**：

| 指标 | 声称 | 实测 | |
|---|---:|---:|:--|
| p99 正常流量 volume_kb | 534.76515 | 534.76515 | ✅ |
| R1 / R2 / R3 / R4 | 7,569 / 0 / 5,974 / 1,600 | 同 | ✅ |
| 命中总次数 | 15,143 | 15,143 | ✅ |
| 唯一命中事件 | 15,117 | 15,117 | ✅ |
| 异常 / 高风险 | 40,000 / 14,014 | 同 | ✅ |

#### 4.3.1 节点负载不均衡实验在两版语料上的对比

`experiment/corpus_load_imbalance_test.py` 在两版平台语料上各跑一次（各18/18 自检通过，~420 s）：

| 项 | 旧（git 版 c594baf5…） | 新（09-28 版 186d4297…） |
|---|---|---|
| 链长 | 15,143 | 15,143 |
| 链级负载比 | 39.4018× | 39.4018× |
| 事件级负载比 | 5.0089× | 5.0089× |
| 完全未被抽中的节点 | **4** 个：`huaxi/jixin/shulan/yiling` | **3** 个：`ciming/shulan/wenlan` |
| 抽样占比最高者 | `ruijin_hn` 72.50%（29/40） | `ruijin_hn` **60.00%**（24/40） |
| 最大倾斜倍数 | 2.07×（`ciming`） | **2.89×**（`evergrande`） |
| 受控对照 A（VICTIM 未抽中） | 8/10 | **8/10** |
| 受控对照 B（VICTIM 被抽中） | 10/10 | **10/10** |

**结论在两版语料上稳定**（A 组因抽样未覆盖而失败 2 条 A3 断言，B 组全通过），但**倾斜表的具体数字不同**——因为 R3 命中的行集变了：旧版 R3 的三个条件恒同（标签回读），新版 `volume>5000` 与 `risk_type=='data_leak'` 互相独立（见 §4.1.2），落在链上的位置随之改变。

这说明 §9.2.3.2.4 的 A3 对照**不是一个稳健结论**：它在「被污染节点恰好落在抽样覆盖内」时成立，落在覆盖外即失败。这一点尚未写入论文，待决。

> 口径提示：`collect_hits()` **不去重**，直接对返回结果取 `len()` 得到的是命中**总次数**。唯一命中事件须按事件统计命中了几条规则（实测：命中 1 条者 15,091 个、命中 2 条者 26 个，合计 15,117）。

### 4.4 验签抽样策略：等间隔 vs 按节点分层（2026-10-03）

`backend/chain.py` 新增两种抽样策略，由环境变量 `AUDIT_SAMPLE_MODE` 切换
（或`verify_chain(..., sample_mode=...)` 显式指定）：

| 模式 | 说明 | 用途 |
|---|---|---|
| `uniform`（默认） | 全局等间隔 `step = n/sp; {int(i*step)}` | §9.2.3 全部已入库产物的口径，**逐字冻结** |
| `stratified` | 每节点保底 1 点，余下按链上占比用最大余数法分配，节点内再等间隔 | 新增，消除节点零覆盖 |

#### 实测（`experiment/sampling_stratification_test.py`，10/10 自检，1.4 s）

09-28 平台语料，v1 规则集，链长 15,143、节点 12、链级负载比 39.4018×、SP=40：

| 指标 | 等间隔 | 分层 |
|---|---:|---:|
| 零覆盖节点数 | **3**（`ciming` / `shulan` / `wenlan`） | **0** |
| 每节点最少抽点 | **0** | **1**（保底保证） |
| 最大倾斜倍数 | 2.8899× | 2.2467× |
| 倾斜极差 | 2.8899 | 1.5009 |
| `verify_chain` nodes_sampled / nodes_total | **9 / 12** | **12 / 12** |

#### 两个必须说清的边界

1. **分层抽样不是把倾斜压到 1.0，而是消除「零覆盖」这个硬缺口。** 40 点 / 12 节点时，小节点按保底拿 1 点，而按链上占比它只应得约 0.58 点，比值必然大于 1。这是点数与节点数的算术下限。因此在保底机制下，**倾斜倍数不再是缺陷指标**——真正的覆盖指标是 `zero_coverage_count` 与 `min_points_per_node`。
2. **旧产物里这个缺口完全不可见。** `verify_chain` 此前只返回 `signatures_verified`，实测 `40/40` 与「12 家全都验过」在JSON 里无法区分。现新增 `sample_mode` / `nodes_total` / `nodes_sampled` 三个返回字段（5 处返回路径全部补齐，含链已判坏时的提前返回），使「某节点从未被验签」成为产物中可读的事实。

#### 影响面

- **抽样策略这一项不影响已入库产物**：`uniform` 为默认且逐字冻结，`experiment/results/` 各组无需因本节改动重跑。`metrics_live.json` 的统计量亦已回归验证可复现——`events / anomaly / high_risk / rule_hits / unique_hit_events / p99 / R1-R4 / chain.length / chain.valid / chain.hash_ok` 逐项一致。**`chain.tip` 不在此列**：它取决于「语料 + 摘要字段清单」两者，属口径指纹而非统计量，见 §4.1.3 的 ⚠。
- **摘要字段清单这一项则确有影响**（与抽样策略无关，见 §4.6）：`experiment/results/` 现有 **13 组**（2026-10-03 增 `chain_field_coverage_test`；原 12 组见 WORKLOG「第9章实验四件套入库」）。其中 **3 组固化了凭证链哈希、已重跑**：`adversarial_test` / `anchor_boundary` / `crypto_cost`；其余 10 组或只记语料 sha256、或用 `baseline_comparison.FIELDS_12` 自带字段集，均不受影响。
- §9.2.3.2.4 的 A3 对照在 `uniform` 下仍依赖抽样覆盖（受控对照 A 组 8/10、B 组 10/10）；在 `stratified` 下 VICTIM 必有抽样点，该依赖被消除。**该项尚未重跑**，见 §4.3.1。
- 新增 5 条测试锁定该边界（`tests/test_rules.py`），pytest 在 {v1,v2} × {uniform,stratified} 四种组合下均 **14/14** 通过（第 14 条 `test_payload_covers_every_rule_input` 为 2026-10-03 新增的摘要字段覆盖回归，见 §4.6）。

### 4.5 规则集注册表（2026-10-03）

`backend/rules.py` 引入 `RULESETS` 注册表，由 `AUDIT_RULESET` 切换：

| 规则集 | 检出规则 | 基线（非检出） | 链长（实验语料） |
|---|---|---|---:|
| `v1`（冻结） | R1 / R2 / R3 / R4 | — | 23,110 |
| `v2`（默认） | R1 / R2 / R3 | R4 | 21,510 |

**R4「已注销账号登录」的实现是 `RandomState(42).choice(n, 1600)`，不读取任何语料字段，输出恒为抽样点数；而语料无账号状态字段，该规则无法被真实实现。** v2 把它从检出规则降级为基线，`apply_rules` 的返回键保持不变（故按 key 取值的旧代码不会 KeyError），变化的只是「哪些规则算检出」。

`metrics_live.json` 用 **v1** 生成，因为 §7.3.1 报的是 15,143（含 R4）。其 `meta.ruleset` 字段记录了这一点。

### 4.6 摘要字段覆盖范围与 `risk_label` 入链（2026-10-03）

`experiment/chain_field_coverage_test.py` 把「能否检出」与「是否影响审计」分两轴实测，
产物 `experiment/results/chain_field_coverage_results.json` / `_summary.md`（9/9 自检通过；
平台语料前 60,000 行，覆盖 12 节点；v1 口径）：

| 轴 | 操作 | 判据 | 结果 |
|---|---|---|---|
| A 检出 | 篡改**已建凭证**中的某列 | `verify_chain` 是否判 `invalid` | 凡在摘要内的列，一律判 invalid |
| B 影响 | 篡改**源数据**中的某列后重算规则 | 命中集合是否变化 | 5/10 列会改变命中（`volume_kb` / `risk_label` / `action` / `is_trusted_line` / `dest_platform`） |

> 初版设计有误：它从篡改后的数据重建链再与自身比对，重建链必然自洽，故测不到检出。
> 已改为篡改**已建**凭证，并由自检条目「在摘要内的列 → 凭证携带 → 篡改必被检出」锁定该口径。

**实测发现 1 处 material gap**：`risk_label` 不在摘要内，而 R1 两处都要读它——既是必要条件
`risk_label == 0`，又被 `compute_p99` 用来筛正常事件算阈值。篡改该列后 `verify_chain` 仍判
`valid`，而 R1 命中由 562 降至 312、总命中 1,183 → 933，即**能改变审计结论却不可检出**。

处置：`backend/chain.py` 的 `_PAYLOAD_FIELDS` 纳入 `risk_label`，`build_chain` 的凭证同步携带。

| 项 | 变更前 | 变更后 |
|---|---:|---:|
| `_PAYLOAD_FIELDS` | 12 | **13** |
| 凭证摘要段数（+ `prev_hash`） | 13 | **14** |
| material gaps | 1（`risk_label`） | **0** |
| 摘要覆盖的语料列 | 8/10 | **9/10** |

**覆盖范围声明**：语料 10 列中 9 列在摘要内；唯一未覆盖的是 `response_status`——它既不在摘要、
凭证也不携带，且**没有任何规则读取**（B 轴实测篡改后命中数 1,183 → 1,183 不变），故属
silent gap 而非 material gap。另有 4 列已受保护但当前无规则读取
（`timestamp` / `hospital_id` / `latency_ms` / `risk_type`，无害）。
按 §4.6 实验自身的处理建议，material_gaps 为空时只需写入覆盖范围声明、不必再改摘要——
**本节即该声明**。

两处附带修正：

1. `chain.py` 原硬编码 `parts[7]` 指向 `is_trusted_line`，改为
   `_TRUSTED_INDEX = _PAYLOAD_FIELDS.index("is_trusted_line")`。下标当时仍然正确，
   但 `_PAYLOAD_FIELDS` 一旦在它之前插字段就会**静默错位**——摘要照算、验签照过、语义全错。
   属预防性修正，非既有缺陷。
2. 新增测试 `test_payload_covers_every_rule_input`（`tests/test_rules.py`），断言每条规则的
   输入字段都在摘要内，锁住该缺口不再回归。pytest 现为 **14 项**（默认口径 5 s 通过）。

**对手造凭证的连带影响（本次实测踩到）**：`_PAYLOAD_FIELDS` 变更后，凡**不带** `risk_label`
的字典喂给 `_payload_from_item` 会直接 `KeyError`。三处手造凭证命中此问题并已补齐：

| 位置 | 用途 | 处置 |
|---|---|---|
| `experiment/adversarial_test.py` A7 伪造追加 | 攻击样本 | 补 `risk_label: 0`（`risk_type=normal` 与 `make_corpus` 对应） |
| `experiment/anchor_boundary_test.py` B3 伪造追加 | 攻击样本 | 同上 |
| `experiment/crypto_cost_benchmark.py` 微基准凭证 | 计时样本 | 补 `risk_label: 1`（`risk_type=data_leak`），并加断言锁定段数 |

`build_chain` 的 **8 处调用方**（`backend/engine.py` 两处、`adversarial_test.py`、
`anchor_boundary_test.py`、`chain_field_coverage_test.py`、`sampling_stratification_test.py`、
`tests/test_rules.py` 两处）数据源均含 `risk_label`，**均未受影响**——三处失败都发生在
`build_chain` 之后的伪造/基准环节，本身即反证了 `build_chain` 正常通过。
`baseline_comparison` / `real_data_chain_test` 等自带固定字段集、不调用 `_payload_from_item`，
刻意保持原口径，不受影响。

**对已入库产物的影响**：字段清单一变，全部节点 `hash` / `prev_hash` 必变。但
**全部统计数字与论文引用值不受影响**（已实测重跑前后逐项一致）：

- `results/metrics_live.json`：已重跑（142 s），`chain_tip` 与 12 个节点 tip 刷新，
  17 项统计数字逐项不变，`meta.corpus_sha256` 仍为 `186d4297…`。
- `experiment/results/{adversarial,anchor_boundary,crypto_cost}_*`：产物固化了链哈希，
  **已重跑**（三者均 `EXIT=0`），逐键比对结果见 §4.7。
- **论文侧零改动**：正文未印任何 64 位链哈希（已全文检索 `thesis_build/*.md` 确认），
  §7 / §9.2.3 引用的均为统计量。

### 4.7 三实验重跑与逐键比对（2026-10-03）

```bash
AUDIT_RULESET=v1 python3 experiment/crypto_cost_benchmark.py    # 32 s
AUDIT_RULESET=v1 python3 experiment/anchor_boundary_test.py     # 223 s
AUDIT_RULESET=v1 python3 experiment/adversarial_test.py         # 493 s
```

以 `git HEAD` 旧产物为基准逐键比对（**按值过滤 64 位哈希与时间戳**，而非按键名——
否则 `hash_ok` / `rate` 这类结论键会被一并跳过）：

| 产物 | 一致 | 变化 | 变化的都是什么 | 结论数字 |
|---|---:|---:|---|---|
| `anchor_boundary_results.json` | 1,975 | 1 | `anchor_record.utc_time` | **零影响** |
| `adversarial_test_results.json` | 18,522 | 19 | 1 时间戳 + 12 个 SM2 公钥指纹（每次运行重生成，设计如此）+ 6 个新增 `ruleset.*` 字段 | **零影响** |
| `crypto_cost_results.json` | 14 | 34 | `payload_fields` 13→14 + 全部计时值 | 见下 |

`adversarial` 单独核对：3,101 个含 `detect / rate / pass / coverage` 的键中，
**除 `generated_at` 被正则误命中外，0 个发生变化**——A1–A8 的 L1/L2/L3 检出率与改动前逐位相同。

**`crypto_cost` 的计时值不可逐次比对**，它测的是纯 Python `gmssl` 在本机的耗时，
受机器负载影响，本次与 10-01 那次的差异属运行间噪声，不是摘要字段造成的：

| 指标（2,000 次中位数 / P95，ms） | 10-01 | 10-03 |
|---|---:|---:|
| SHA-256 摘要 | 0.0012 | **0.0011** |
| SM3 摘要 | 0.9127 | **0.796** |
| SM2 签名 | 4.9263 | **4.3019** |
| SM2 验签 | 10.0499 | **8.8747** |
| Merkle 根（链长 23,110） | 24.5451 | **18.2889** |
| 锚定核验全流程 | 24.7789 | **21.2347** |

引用该表须标明「本机、`gmssl` 纯 Python 实现、运行间有噪声」；`payload_fields` 13→14
是唯一由本次改动引入的确定性变化（多一个字段参与 SHA-256，耗时影响在噪声之下）。

**小结**：`risk_label` 入链刷新了链哈希与密钥指纹，**未改变任何一条实验结论**。

#### 附：`metrics_live.json` 的确定性验证（连跑两次）

同一代码、同一语料连跑两次 `precompute_metrics.py`（141.6 s / 137.4 s），逐键比对：

| 项 | 结果 |
|---|---|
| `summary.chain_tip` | **完全一致** `e75c5e48f1d3c0b2e7d441932bd42ceb3252b3cf7ab93c9b787fdad6390c2eb7` |
| 12 条节点链 tip | **全部一致** |
| 10 项关键统计（events / anomaly / high_risk / p99 / rule_hits / unique_hit_events / R1 院命中 / chain_valid / chain_hash_ok / node_chains） | **10/10 一致** |
| `meta.corpus_sha256` | 一致 `186d4297…` |
| 差异项 | **仅 3 处**：`chain.head.signature`、`chain.tail.signature`、`meta.load_seconds` |

> **SM2 签名不可逐次比对，这是算法属性而非复现失败**：SM2 每次签名使用随机临时密钥，
> 同一消息两次签名必然不同。因此可复现性校验应以**链哈希**（`chain_tip` / 节点 tip）
> 为判据，签名只能校验「用对应公钥验签是否通过」。同理，`g4_public_keyring.*`
> 的 12 个指纹每次运行重新生成，也不属可比对项。

## 5. 与既有文档数字的对照

- 与 7.3.1「80 万源语料口径 15,143」逐项一致：R1 7,569 / R2 0 / R3 5,974 / R4 1,600。**已按 §4.3 实测复现（9/9）。**
- 与正文 7.4.1 的「178 万扩展集口径 26,588」分母不同（178 万 vs 80 万），不可互相加总；两口径使用同一套 4 条规则。
- 异常/高风险与本文第 7 章 80 万口径直接计数一致（40,000 / 14,014），其中高风险定义与正文显式约定相同（data_leak+brute_force 且 risk_label=1）。
- **第 9 章实验口径（`make_corpus`）与本章平台口径数字不可互相引用**，两者语料不同。引用时须标明取自哪一套。
- 既往「23 条规则 / 2,847 条 / 6 合规域」无产物，已作废，与本系统无关。
- **⚠ 正文 6.3.3 / 8.2.3 引用的国密开销无仓内出处**：`SM3 0.787 ms / SM2 签名 2.053 ms /
  SM2 验签 7.835 ms（2,000 次均值）`注为出自 `measure_crypto.py` 与 `密码学运算开销实测.json`，
  但**二者在本仓均不存在**（`archive/historical/recent_20261001/output/…/缺失文件与原因分类_20261001.md`
  已将其记为「有结论无实现」缺口）。仓内 `experiment/crypto_cost_benchmark.py` 的文档串亦自陈该组
  数字为 **G5 随机数整改前**的实测值、「已与当前代码不符」，本脚本即为重测脚本，
  但正文尚未替换为 §4.7 的重测值。**处置待定，须与正文作者确认后修订。**

### 5.1 Marvis 平台语料规模敏感性补跑的验收结论（2026-10-03，方案 A：仅记录）

2026-10-03 Marvis 在平台语料上补跑了一次规模敏感性实验，产物落在其会话工作区
（`MarvisData\...\conv_1a0c73c74fd_e928caffe048\temp\scaling_platform_results.json`
与 `output\scaling_platform_summary.md`）。经逐键核对，该轮结果**不作为本仓有效产物**，
也不满足 §9 四件套，登记 5 处缺陷如下。**处置方式：方案 A —— 仅此处文档记录，不改代码、
不重跑**（用户 2026-10-03 决定）。

| # | 缺陷 | 证据 |
|:--:|---|---|
| 1 | **md 与 JSON 的断言条数对不上**：md 写「断言（18 条）」、其中 1 条 FAIL（`两口径检出率之差随链长单调不减`），JSON 的 `checks` 实际只有 **3 条且全 PASS**。两个文件不是同一份判定结果，无法作为验收依据 | `checks` 长度 3 ≠ md「18 条」 |
| 2 | **md 与 JSON 的最大档位对不上**：md 写档位 `[1000, 3000, 5000, 8000, 12000, 15143]` 并称「全量 15,143」，JSON 实际 `scaling[5].chain_length = 15117`（15143 是 `chain_input_hits`，不是档位值），**相差 26 条**；且该组档位与仓内 `scaling_variance_test.py` 默认 `1000,5000,10000,23110,50000,100000` 也不同，两套结果不可对照 | `chain_length=15117` vs md「15143」 |
| 3 | **语料无 sha256 且为仓外 23 列版**：`corpus` 只记了 path/rows/columns/note，**缺 `sha256`**；path 指向 `<external-archive>/lecheng_security_logs_2024H1.csv`（23 列 800,000 行），按 `data/generator/README.md`，23 列全量版含 `patient_id`/`user_id`/`source_ip`/`session_id` 四个可标识字段、**明确不入库** ⇒ 第三方拿不到该语料，结果**不可复现** | `corpus` 无 `sha256` 键 |
| 4 | **规则集非本分支现行版**：`ruleset.source = "worktree B backend/rules.py (83行, 无注册表)"`，而 `opencode` 分支现行 `backend/rules.py` 为 146 行、含 `RULESETS` 注册表。产物本身如实记录了这一点（好），但意味着该结果**不对应本分支代码**，与 §4.5 的注册表口径无法互证 | `ruleset.source` 字段 |
| 5 | **产物未入库**：JSON/MD 均在 MarvisData 的 `temp\`、`output\` 下，**不在 `experiment/results/`**，且 `temp\` 属临时目录随时可能被清理 ⇒ 违反 AGENTS.md §9「结果 + 源码 + 可复现过程 + 完整工作流」四件套 | 文件路径不在仓内 |

> **共同点**：缺陷 2、3、4 的根因都是「**产物没有自记口径**」——语料 sha、规则集来源、
> 档位取值这三项都应由脚本写进 JSON，而不是靠事后在 md 里手工补述。这与 Marvis 同期另一份
> 报告（`科研项目结构梳理报告_20261003.md` L83）指出的
> 「`scaling_variance_test.py` 产物 JSON 不记录规则集版本」是同一个问题的两个侧面。
>
> **本仓对应口径**：仓内 `experiment/results/scaling_variance_results.json`（2026-10-01 22:09）
> 自检 18/18 PASS、语料与规则集口径见 §4.2，**不受本轮影响**，仍是规模敏感性实验的唯一有效产物。

## 6. 本地 LSTM 风险推理（真实产物）

训练脚本 `scripts/train_lstm.py`，产物目录 `results/lstm/`（model.pt + model_meta.json + metrics_lstm.json + params_train.json）。任务定义：滑窗含当前事件（窗口 24 × 5 维特征）→ 预测当前事件 `risk_label`，时间序切 8:2，seed=42。

| 指标 | 值 |
|---|---|
| 训练序列 | 120,000 |
| 测试窗口 | 159,977（正例 7,981 / 负例 151,996） |
| Accuracy | 0.9729 |
| Precision | 0.6827 |
| Recall | 0.8540 |
| F1 | 0.7588 |
| AUC | 0.9192 |
| 混淆矩阵 | TP=6,816 TN=148,828 FP=3,168 FN=1,165 |
| 训练耗时 | 186.4 s（torch 2.13.0+cpu，pos_weight=18.73） |

> **数字真实性声明**：以上全部由 `scripts/train_lstm.py` 真实训练+评估产生。若任何文档（博论第六章等）引用的 LSTM 指标（如 0.939/0.877）与本 JSON 不一致，一律以本 JSON 为准；无脚本无产物的历史指标视为未验证，禁止入文（AGENTS.md §9）。
