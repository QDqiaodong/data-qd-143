# 数据契约

本仓包含两条相互独立但共享同一套 PAVA 校准内核的链路：

1. **v1 单次 JSON 分析（保持不变）**：`score-calibration --source F --output F`。
2. **v2 修订账本与版本化报告**：`score-calibration ingest` 与 `score-calibration report`，
   仅使用 Python 标准库与 `sqlite3`。设计与崩溃恢复说明见 [design-ledger.md](design-ledger.md)。

---

## 1. v1 单次 JSON 分析（兼容保留）

为评分模型分析者拟合单调分数→经验概率映射，训练和独立评估各 <=20000 条，行
id/score/label/weight；score 为 0..10^6 整数排序分数，不当概率，label 严格 0/1，weight 为
1..10^6 整数。各集合 id 唯一，训练与评估 id 不得重叠，训练非空，评估可空。

同分数先加权合并，再使用 PAVA：相邻块经验阳性率下降时合并，得到非递减分段常数概率。相等率
块不强制合并。概率为精确 Fraction。新分数选择起始分数 <= score 的最后一块，区间左闭；低于
最小夹到第一块、高于最大夹到最后一块。不是插值。

输出训练块边界/权重/阳性权重及评估每行预测、加权 Brier 与 log-loss。Brier 精确分数；
log-loss 用自然对数，出现真实标签对应概率 0 时返回 null 并显式 infinite_log_loss=true，不裁剪
伪造有限值。空评估两指标 null。样例块 10→0、20..30→1/2、40→1；两个评估预测 1/2 与 0，
加权 Brier=1/8，log-loss=ln2/2。评估标签不参与训练。

输入 JSON 最多 8MB，拒绝重复键、非有限数值和未定义字段；源字节 SHA256 随结果保留。输出
原子替换，输入/计算失败退出 2 并保留旧报告；输入和输出不得相同，输出符号链接拒绝。

---

## 2. v2 修订账本

### 2.1 ingest 批次契约

`score-calibration ingest --ledger DB --batch FILE`，batch 为 UTF-8 JSON 对象
`{"events":[...]}`，数组非空。每个事件必须且仅含字段：

| 字段 | 契约 |
| --- | --- |
| `event_id` / `observation_id` / `entity_id` | 非空字符串、无控制字符（`ord<32`）、长度 <=4096 |
| `observed_at` | 仅接受带时区的 RFC3339；规范为 UTC 固定宽度 `YYYY-MM-DDTHH:MM:SS.ffffffZ` |
| `score` | 0..1_000_000 整数 |
| `label` | 0/1 整数 |
| `weight` | 1..1_000_000 整数 |
| `retract` | 布尔（不接受 0/1 冒充）；撤回事件仍须给全部字段，但其观察值不参与计算 |

- 批内 `event_id` 必须唯一，否则整批拒绝且不写入任何行。
- 事件规范身份：严格按契约字段顺序、UTF-8、无空白的规范 JSON 字节。批次摘要为组内规范
  事件字节按数组顺序拼接的 SHA256。
- 提交语义（单事务）：
  - 与某已提交批次**完整规范事件集合及顺序字节等价** → 重放，返回原 `knowledge_seq`、
    原摘要、`replay=true`，不推进序号。
  - 全新事件 → 原子追加，取得下一个正整数 `knowledge_seq`，批内位置记为 `position`。
  - 来自不同旧批次的混合重放、重复 `event_id` 内容冲突、批内重复 → 一律全拒。
- 事件表只追加，任何路径都不物理删除事件。

### 2.2 report 请求契约

`score-calibration report --ledger DB --request FILE --output FILE`，request 为：

```json
{"observed_at_cutoff": "<RFC3339 带时区>",
 "knowledge_seq": "latest",
 "split_seed": "非空字符串"}
```

- `knowledge_seq` 为正整数或 `"latest"`；报告启动时把 latest 解析为当时已提交的最大 seq；
  数值 seq 高于当时最大值报业务错误。
- 对截止 seq 的每个 `observation_id`，按 `(knowledge_seq DESC, position DESC)` 取最后事件；
  最后事件为撤回则该 ID 不存在；再保留 `observed_at <= cutoff` 的观察（UTC 固定宽度字符串
  字典序比较，边界含等号）。
- 实体整体切分：`SHA256(split_seed UTF-8 + NUL + entity_id UTF-8)` 首 8 字节无符号大端值
  除以 2^64，`< 4/5` 进训练，否则评估；同一 entity 只进一侧。
- 训练集空 → 业务错误且不写输出；评估集可空，Brier/log-loss 为 null、
  `infinite_log_loss=false`。
- 训练调用 v1 同一套 PAVA（`isotonic.fit`），评估逐行预测与指标口径与 v1 完全一致。

### 2.3 报告内容、依赖身份与发布

报告（确定性 JSON，键排序、无时间戳/随机字段）含：

- 解析后的完整 request（`latest` 落为具体整数 seq）；
- 固定 `knowledge_seq`、事件数与事件日志摘要 `log_digest`；
- 物化视图摘要 `view_digest` 与有效观察数；
- 排序后的训练/评估 observation_id 及各自依赖摘要；
  - 训练依赖身份 = 有效训练观察（id + 规范值）逐行哈希；
  - 评估依赖摘要 = 有效评估观察同法哈希；
- PAVA 校准块、逐行预测与评估指标；
- `request_sha256` 与 `report_sha256`（剥离这两个键后对正文取规范 SHA256，可外部复算）。

分层失效：

- 仅评估侧 score/label/weight 变化 → 训练依赖身份不变、训练曲线复用，重算本版评估结果；
- 训练成员或其 score/label/weight 变化（含 cutoff 越界、撤回/恢复、entity 跨 split 迁移）→
  训练依赖身份更新并按新身份取用曲线；曲线值允许与旧值恰好相同。
- 旧 seq 的视图、缓存、报告按 seq/摘要共存，不被新 seq 覆盖。

发布：同一账本版本与 request 逐字节相同；输出已存在且内容完全相同则幂等成功，否则报冲突并
保留原文件、不登记；符号链接拒绝；临时文件 + 原子 `os.replace` 落盘成功后才登记
`report_registry`，失败保留既有输出并允许同请求重试。

### 2.4 版本固定、读视图与崩溃恢复

- 报告启动即在短读事务内固定已提交 seq 与该 seq 事件日志摘要；此后并发提交的更高 seq
  不会混入。
- 读视图采用**不可变事件范围副本**：将 seq 内事件与该 seq 物化观察流式复制到账本同目录临时
  SQLite，复制完即关闭源读事务释放 WAL；副本独立重算事件日志摘要、事件数与视图摘要，任一
  不符即 `ReplicaIncomplete`，删除副本、不写输出、不登记、可重试。
- 物化投影按 seq 寻址，`building` 半成品与失配 `complete` 投影在重启后由事件日志重建；
  只有新批次提交后才推进物化水位，半成品不会被标记完成。
- 折叠、摘要、复制均 `fetchmany` 流式进行；内存上界随选定 seq 的有效观察量增长，不随旧修订
  事件总数增长。
