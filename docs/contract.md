# 数据契约

本仓包含四条相互独立但共享同一套 PAVA 校准内核的链路：

1. **v1 单次 JSON 分析（保持不变）**：`score-calibration --source F --output F`。
2. **v2 修订账本与版本化报告**：`score-calibration ingest` 与 `score-calibration report`，
   仅使用 Python 标准库与 `sqlite3`。设计与崩溃恢复说明见 [design-ledger.md](design-ledger.md)。
3. **v3 双人盲标规划与答案回收**：`score-calibration annotation-plan` 与
   `score-calibration annotation-resolve`，仅使用 Python 标准库；交付自校验 JSON
   标注包，不写入修订账本。设计说明见 [design-annotation.md](design-annotation.md)。
4. **v4 评分漂移监测与通知发件箱**：`score-calibration monitor-sync / monitor-claim /
   monitor-ack / monitor-status`，仅使用 Python 标准库与独立 `sqlite3` 监测库；不读写
   修订账本或盲标包。设计与并发正确性说明见 [design-monitor.md](design-monitor.md)。

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

---

## 3. v3 双人盲标规划与答案回收

### 3.1 annotation-plan 输入契约

`score-calibration annotation-plan --source F --output F`，输入必须且仅含字段
`training / pool / annotators / quotas / budget`，拒绝未知字段与重复 JSON 键。

- `training`：沿用 v1 的 `id/score/label/weight` 行契约（`common.ident`、score
  0..10^6 整数、weight 1..10^6 整数、label 严格 0/1），非空；布尔不得冒充整数。
- `pool`：1..18 行，每行严格为
  `{id, entity_id, group, score, weight, eligible}`。
  - `id` 为标识符且全池唯一，不得与任一 `training.id` 相交；
  - `eligible` 为允许接单的标注员 ID 数组，非空、无重复、只引用已声明人员；
  - 同一 `entity_id` 的所有候选必须属于同一个 `group`。
- `annotators`：1..8 行，每行 `{id, capacity}`，`capacity` 为 0..36 整数。
- `quotas`：非空 `group -> {min,max}` 映射，至多 4 个组，
  `0 <= min <= max <= 18`；每个 pool 组必须恰有一条配额。
- `budget`：0..36 整数。

### 3.2 概率、价值与精确分配

- 先在 `training` 上跑与 v1 完全相同的加权 PAVA，得到非递减阶梯；pool 分数按
  “起始分数 <= score 的最后一块”取精确 `Fraction` 概率（区间左闭、两端夹取，非插值）。
- 候选价值固定为精确有理数 `weight * (1 - |2p - 1|)`，在规划时一次性封入包内；
  后续回收答案不会重算概率或价值，本轮排序依据不变。
- 每个被选候选恰分配**两个不同的 eligible 标注员**；同一 entity 至多选一条；一个
  标注员在整个包内不得处理同一 entity 的两条候选。每份任务花费 1 预算并各占两名
  人员 1 点 capacity，组覆盖数必须落在 `[min,max]`。
- 全局目标严格按字典序：先最大化候选价值之和，再最大化覆盖候选数，最后取排序后
  `(id, 较小人员ID, 较大人员ID)` 元组序列字典序最小者。全程精确有理比较，不存在
  “先按价值贪心再补配额”的捷径；高价值贪心使组下界不可达时必须让位。
- 无可行解时包 `status` 显式为 `infeasible` 并给出各组真实缺口 `gaps`，绝不生成
  伪成功的待办；此时仍保留可 staff 的最佳努力任务（best-effort），不伪造可行。

### 3.3 annotation-resolve 答案契约

`score-calibration annotation-resolve --package F --answers F --output F`，答案文件
严格为 `{"package_sha256": "...", "answers": [...]}`：

- 摘要必须与输入包自校验 SHA256 完全一致，否则整批拒绝且不触碰输出。
- 每份答案严格为 `{id, annotator_id, label}`，`label` 为 0/1 整数（布尔拒绝）。
- 未知任务、答卷人不在该任务分配对中、同一 `(任务, 标注员)` 重复答卷、改写包中
  已有答案，一律整批拒绝（先全部校验再落任何变更）。
- 允许只回收部分任务：0 份答案合法且产出逐字节等价的新包。任务状态机：
  无答案 `open` → 一份 `pending` → 两份一致 `agreed` / 两份不同 `disputed`。
- 仅 `agreed` 候选进入 `truth` 真值数据集，行字段为可直接喂给 v1 分析的
  `id/score/label/weight`；`pending` 与 `disputed` 一律不混入。

### 3.4 争议、补位与包不变量

- 候选变为 `disputed` 时：取消其组覆盖贡献，但**保留两份已花预算与两点人员占用
  （不退款）**，封锁该 entity，并在剩余候选上重算补位组合。
- `pending` 与 `agreed` 的原分配全部固定，补位不得挪人或换样；新补位任务同样双人
  且受剩余预算、剩余人员 capacity 与组上界约束，目标只在补位部分上比较。已经分配
  过的候选不得再次入选。
- 补位不可行时仍输出完整回收结果、真值与残余缺口并标记 `infeasible`，不得丢弃已
  收到的答案或通过退款制造可行性；包仍可继续回收其余任务。

### 3.5 包内容、自校验与发布

包为确定性 JSON（键排序、无时间戳/随机字段），含：规范化源数据 `source`、训练块
`training_blocks`、逐候选 `probability/value` 的 `candidates`、全部已分配 `tasks`
（含状态、答卷、generation）、预算与人员 `usage`、`group_coverage`、`gaps`、
`truth` 与自校验 `package_sha256`。

- 摘要算法：去除自身 SHA 字段后，对剩余对象取排序键、紧凑、UTF-8 JSON 的 SHA256。
- resolve 必须重算并核验摘要、源 PAVA 概率/价值、任务状态、预算/容量/覆盖/缺口/
  真值的一致性；用户伪造的 `agreed` 状态或 `truth` 行无法通过校验。
- 源训练、pool 或人员数组换序不影响最终包字节。
- 允许输出路径与输入包相同：先完整读取、校验、计算，再临时文件 + 原子 `os.replace`
  替换；输出符号链接拒绝。输入或计算错误退出 2 并保留旧输出；合法但 infeasible 退出
  0 并显式标注状态，包可继续回收。

---

## 4. v4 评分漂移监测与通知发件箱

监测库是独立 SQLite 文件，仅由本链路读写，不触碰账本/盲标表。业务产物是可纠正的分组
漂移 episode 与可恢复投递的通知，不是版本化校准报告。设计与并发论证见
[design-monitor.md](design-monitor.md)。

### 4.1 monitor-sync 快照契约

`score-calibration monitor-sync --db DB --snapshot FILE`，快照严格且仅含：
`snapshot_id, reference, observations, start_date, period_days, period_count,
min_weight, trigger_delta, clear_delta, trigger_periods, clear_periods`。拒绝未知字段、
重复 JSON 键；两个数组各 <=20000 行。

- `reference` 沿用训练 `id/score/label/weight` 契约（`common.ident`、score 0..10^6、
  weight 1..10^6、label 严格 0/1，布尔不算整数），非空。
- `observations` 每行严格为 `id/cohort/date/score/label/weight`：id 唯一，cohort 与
  snapshot_id 沿用 `common.ident`，date 为合法 `YYYY-MM-DD`，其余沿用现有整数界限；
  两集合 id 不相交。
- 日期必须落入从 `start_date` 起连续、左闭右开的日历窗口。
  `period_days 1..31`、`period_count 1..366`、`min_weight 1..10^9` 整数；
  `trigger_periods/clear_periods 1..period_count`。
- `trigger_delta/clear_delta` 为 `[0,1]` 内有限十进制字符串（拒绝指数/分数/NaN），
  且 `clear_delta < trigger_delta`。
- 首次成功 sync 后除 `snapshot_id/reference/observations` 外的配置固定；后续快照是
  **完整替换**（可纠正过去样本、删除整组），不是增量追加；空 observations 合法。
- 同 `snapshot_id` 同规范内容重放返回原回执，不推进 revision、不复活通知；同 id 不同
  内容整次拒绝。新 id 即使内容相同也推进 revision。首个合法 sync 取得 revision=1。

### 4.2 度量与 episode

PAVA 与左闭阶梯预测复用现有内核，无插值、无概率裁剪。每 cohort 每窗（空窗也输出）计算
精确加权残差 `|Σ weight*(label-p) / Σ weight|`（Fraction）；总 weight 低于 min_weight
（含空窗）标 `insufficient`，残差 null，且不能跳过空窗拼接 streak。

- 连续 `trigger_periods` 个 eligible 且残差 `>= trigger_delta` 才开窗，`start` 回溯到
  首个越阈值窗口；开窗后连续 `clear_periods` 个 eligible 且残差 `<= clear_delta` 才关窗，
  `end` 为完成清除窗口。中间区间重置待定计数；insufficient 重置计数但不关已开 episode。
- 触发 streak 起点至关窗窗口的全部 eligible 窗口参与 peak，平局选最早窗口。规范 payload
  为 `start/end/peak_value/peak_window/open`；episode 身份是 `(cohort,start窗口)`；关闭
  后下一窗才可开新 episode。

### 4.3 通知、租约与确认

- 首次出现 episode 产生 upsert；同身份 payload 变化产生新 upsert；消失按投递历史决定
  retract。未发送的旧同身份通知被取代时 cancelled 且租约立即失效；已 sent 的变化发
  upsert、消失发 retract，历史通知永不物理删除。同身份只有最新所需通知可 pending/leased。
- 从未 sent 即消失只静默取消，不发 retract。已 sent 后消失、retract 未发送又出现：取消
  retract 并按最后实际投递 payload 判定，相同则不再发。A 已 sent、B 待发又回到 A 时
  取消 B 且不另发。判定依据是实际最后投递状态而非最近计算状态。
- `monitor-claim` 按通知 id 取第一个 pending 或租约已过期（`now >= lease_until`）的当前
  有效通知，返回不可复用 token、`lease_until=now+lease_seconds` 与 payload，无通知返回
  null（退出 0）。`now` 非负整数，`lease_seconds 1..86400`，worker 用 `common.ident`，
  每次 token 不同。已 sent 不再被 claim。
- `monitor-ack` 仅在 token 为当前有效租约、未过期且通知仍有效时置 sent 并更新该 episode
  最后实际投递；同 token 重复成功 ack 幂等返回原回执（不受当前 now/租约过期影响），失效
  token 一律拒绝。重开库不丢租约、sent 状态与 ack 回执。无网络发送，外部按通知 id 去重。
- 并发：ack 先提交则删除 episode 的 sync 必须留 retract；sync 先提交并取消租约则旧 ack
  必须拒绝。sync 提交指纹化全部相关表（ack 不改 revision 也可能改所需通知，只核 revision
  不足），事务外旧计算遇到竞争 sync 不得覆盖新状态。

### 4.4 状态发布

`monitor-status --db DB --output FILE` 在单一致读快照内输出 revision、当前每窗精确
Fraction 指标、各组 episode、全部通知/租约/sent 记录与成功回执；排序键 JSON、无运行时
时间戳。符号链接拒绝；输出已存在且内容不同保留旧文件并退出 2；合法空队列退出 0。
