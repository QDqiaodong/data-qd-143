# 评分漂移监测与通知发件箱设计（v4）

本模块是一条**独立链路**：拥有自己的 SQLite 监测库（`--db`），只使用 Python 标准库与
`sqlite3`，从不打开或写入修订账本（`events/batches/...`）与盲标包。既有的单次分析、修订
账本与盲标命令全部保持不变。业务产物是**可纠正的分组漂移 episode** 与**可恢复投递的
通知**，不是另一份版本化校准报告。

四个子命令：

| 命令 | 职责 |
| --- | --- |
| `monitor-sync --db DB --snapshot FILE` | 校验并原子发布一个快照修订，重算 episode 并对账通知 |
| `monitor-claim --db DB --worker ID --now N --lease-seconds N` | 领取第一个可投递的当前有效通知并建立租约 |
| `monitor-ack --db DB --notice ID --token TOKEN --now N` | 凭有效租约确认投递，幂等 |
| `monitor-status --db DB --output FILE` | 单一致读快照发布排序键 JSON 状态报告 |

## 1. 快照契约与语义身份

快照严格且仅含字段：
`snapshot_id, reference, observations, start_date, period_days, period_count,
min_weight, trigger_delta, clear_delta, trigger_periods, clear_periods`。
拒绝未知字段、重复 JSON 键；`reference` 与 `observations` 各自上限 20000 行。

- `reference` 沿用现有训练行契约 `id/score/label/weight`（`common.ident`、score
  0..10^6、weight 1..10^6、label 严格 0/1，布尔不冒充整数），且**非空**。
- `observations` 每行严格为 `id/cohort/date/score/label/weight`：`id` 全表唯一，
  `cohort` 与 `snapshot_id` 沿用 `common.ident`；`date` 为合法 `YYYY-MM-DD`；其余沿用
  现有整数界限。两个集合的 id 必须**不相交**。
- 每个观察日期必须落入从 `start_date` 起、连续、左闭右开的日历窗口
  `[start_date, start_date + period_days*period_count)`，窗口之间无空隙。
- `period_days ∈ [1,31]`、`period_count ∈ [1,366]`、`min_weight ∈ [1,10^9]` 均为整数；
  `trigger_periods, clear_periods ∈ [1, period_count]`。
- `trigger_delta`、`clear_delta` 为 `[0,1]` 内的**有限十进制字符串**（拒绝指数、分数、
  NaN/Infinity 与浮点），内部转精确 `Fraction`；要求 `clear_delta < trigger_delta`。

### 1.1 配置固定与整表替换

首次成功 sync 后，除 `snapshot_id/reference/observations` 以外的字段（含 `start_date`
与全部阈值/窗口配置）在该库中**固定**；后续快照改动配置会被整次拒绝。每次 sync 都携带
**完整**的 reference 与 observations（可纠正过去样本、删除整组），不是增量行追加。空
observations 合法，表示所有当前 episode 消失。

### 1.2 语义内容摘要

对两个样本数组按 `id` 排序（数值字段本就是精确整数，delta 已规范为 Fraction 字符串），
连同固定配置（含 `start_date`，**不含** `snapshot_id`）取排序键、紧凑、UTF-8 JSON 的
SHA256。因此：

- 相同 `snapshot_id` + 相同规范内容 → 重放，返回原回执，不推进 revision、不复活通知；
- 相同 `snapshot_id` + 不同内容 → 整次拒绝（snapshot_id 不被占用，无半修订）；
- 不同 `snapshot_id`，即使内容完全相同 → 仍推进 revision。

## 2. 精确漂移模型核（纯函数，无 I/O）

先在 `reference` 上运行现有的加权 PAVA（`isotonic.fit`）；观察预测沿用“起始分数
<= score 的最后一块”的**左闭阶梯**规则，两端夹取，**无插值、无概率裁剪**。

每个 cohort 的每个日历窗口（共 `period_count` 个，空窗也输出）聚合：

```
residual = | Σ weight*(label - p) | / Σ weight
```

全部用 `fractions.Fraction` 精确计算。窗口 `Σ weight < min_weight`（含空窗，权重 0）标
`insufficient`，残差为 `null`；否则为 `eligible` 并输出精确 Fraction 字符串。空窗照样
占位，**绝不跳过空窗拼接 streak**。

### 2.1 episode 状态机（逐 cohort 按完整日历消费）

- 连续 `trigger_periods` 个 **eligible 且残差 >= trigger_delta** 的窗口才打开 episode；
  `start` 回溯为该 streak 的**首个越阈值窗口**（不是触发的那一窗）。
- 打开后，连续 `clear_periods` 个 **eligible 且残差 <= clear_delta** 才关闭；`end` 是
  完成清除的窗口。
- 中间区间（严格位于 clear 与 trigger 之间）的 eligible 窗口重置待定计数。
- `insufficient` 窗口重置两个待定计数，但**不关闭**已打开的 episode。
- `start` 至 `end`（未关闭则到当前视野末尾）之间的**全部 eligible 窗口**参与 peak；
  peak 取最大残差，平局选最早窗口。
- 关闭后的下一窗才允许开始新 episode。episode 身份是 `(cohort, start窗口)`。
- 规范 payload：`start / end / peak_value / peak_window / open`。

## 3. 持久通知发件箱

### 3.1 表与不变量

`notices`（AUTOINCREMENT 单调整数 id，取消的 id 不复用）、`leases`（仅追加，保留终态）、
`delivery`（每个 episode 身份的**最后实际投递** payload，只由成功 ack 写入）、
`ack_receipts`（成功 ack 的持久回执，按 token 主键）、按 revision 共存的 `windows` 与
`episodes`、以及 `snapshots` 与 `meta`。**通知从不物理删除**：取代只是把 `active` 置 0
并记录 `cancelled_revision`。

### 3.2 对账规则（按 cohort/start 窗口排序处理）

- episode 首次出现 → `upsert`；同一身份 payload 变化 → 新 `upsert`；episode 消失 →
  视投递历史决定是否 `retract`。
- 未发送（pending/leased）的旧同身份通知在被取代时 `cancelled`，其租约**立即失效**。
- 旧 payload 已经 `sent`：变化发新 `upsert`；消失才发 `retract`，不能只删历史。
- 同身份只有最新所需通知可处于 pending/leased。
- episode 从未发送就在发送前消失：只静默取消，不生成 retract。
- 已发送后消失、retract 尚未发送 episode 又出现：取消旧 retract，并把新 payload 与
  **最后实际投递** payload 比较；相同则不再发。
- payload A 已发送、B 待发后又回到 A：取消 B 且不另发（消费者仍持有 A）。
- start 窗口因 payload 纠正而移动属于“同 cohort 的取代”，旧身份不产生 retract；只有整个
  cohort 的 episode 真正消失，且其当前通知已 sent，才欠一个 retract。

判断始终基于 `delivery` 中的**实际最后投递状态**，而不是最近一次计算状态。

## 4. 租约、领取与确认

- `claim` 在一个 `BEGIN IMMEDIATE` 事务内按通知 id 升序取第一个 pending 或租约已过期
  （`now >= lease_until`）的当前有效通知；返回不可复用的随机 token、
  `lease_until = now + lease_seconds` 与 payload；无通知返回 JSON `null`（退出 0）。
- `now` 为非负整数，`lease_seconds ∈ [1,86400]`；worker 沿用 `common.ident`；每次领取的
  token 都不同。
- `ack` 仅当 token 仍是该通知的当前有效租约、未过期、通知仍有效时，才把通知置 `sent`
  并更新该 episode 的最后实际投递状态，同时持久化成功回执。
- 同一 token 的重复成功 ack **幂等**返回原回执，不受当前 `now`/租约过期/后续 sync 更正
  影响；失效 token、过期租约、已取消通知一律拒绝。已 sent 通知不再被 claim。
- 重开数据库不丢失租约、sent 状态或 ack 回执。系统不做网络发送，也不声称网络端
  exactly-once；外部消费者按通知 id 去重。

## 5. 并发正确性：为何只核 revision 不足

存在两种正确的串行次序（用独立 SQLite 连接与可控屏障重现，禁止用 sleep 碰运气）：

1. **ack 先提交**，随后删除 episode 的 sync 必须留下 `retract`；
2. **sync 先提交**并取消旧租约，则旧 ack 必须被拒绝。

关键在于：一次成功的 ack 会改变 `leases/notices/delivery`（从而改变“需要哪些通知”），
却**不推进 snapshot revision**。因此 sync 提交时若只比较 revision，就会漏掉 ack 带来的
通知需求变化。

### 5.1 两种实现方案的正确性义务

- **方案 A：事务内完整读算写。** 长事务在整个计算期间持有写锁/快照，天然隔离并发
  claim/ack；代价是计算期间阻塞其他连接，且长事务会持续保留 WAL 页。
- **方案 B（本实现所选）：事务外计算 + 提交时核完整状态后重试。** 纯计算在事务外完成，
  不持锁；提交在 `BEGIN IMMEDIATE` 内**重新读取并指纹化全部相关表**
  （快照计数、最大 notice id、最大 lease id、delivery 计数与最大已投递 notice id、
  活跃通知计数），核对计算所绑定的指纹。任一竞争 claim/ack/sync 已提交都会改变指纹，
  触发回滚与廉价重试，因此事务外的旧计算永远无法覆盖新状态。屏障特意安排在
  `BEGIN IMMEDIATE` 之前，使竞争连接能在等待期间提交。

该方案避免“计算途中 claim/ack 导致漏更正”的方式是：对账本身在写事务内基于**提交时刻**
的最新通知/投递状态整体重放，而不是基于计算时刻的快照；指纹不匹配即放弃本轮对账重来。

### 5.2 失败与恢复

- 事务失败（含提交前注入异常）整体回滚：不占用 snapshot_id，不留下半个 episode/通知更新。
- 提交成功后若 stdout 失败，可用相同 `snapshot_id` 重放恢复原接收结果（幂等，不推进
  revision、不复活通知）。
- `monitor-status` 在单个只读事务的一致快照内读取，绝不混合某版本 episode 与另一版本
  通知；输出排序键 JSON、无运行时时间戳；符号链接拒绝；输出已存在且内容不同则保留旧
  文件并退出 2；合法空队列退出 0。
