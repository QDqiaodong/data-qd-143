# 修订账本与版本化校准报告设计（v2）

本文说明 `ingest` / `report` 两个子命令的数据模型、隔离与恢复保证。仅使用 Python 标准库与
`sqlite3`。旧的 `--source/--output` JSON 分析链路保持不变。

## 1. 事件与批次（仅追加日志）

### 1.1 事件校验

batch 为 UTF-8 JSON 对象 `{"events": [...]}`，数组非空。每个事件必须且仅能包含字段：

`event_id, observation_id, entity_id, observed_at, score, label, weight, retract`

- 三个 ID：非空字符串、不含控制字符（`ord < 32`）、长度上限 4096。
- `observed_at`：仅接受带时区偏移的 RFC3339（正则限定 `T` 分隔与 `Z`/`±HH:MM` 偏移），
  解析后统一转换为 UTC 并格式化为固定宽度 `YYYY-MM-DDTHH:MM:SS.ffffffZ`，因此字符串字典序
  即时间先后。
- `score`：`0..1_000_000` 整数；`label`：`0/1` 整数；`weight`：`1..1_000_000` 整数；
  `retract`：布尔值。布尔型不冒充整数。
- 撤回事件仍须提供全部字段，但其观察值不参与任何计算。

### 1.2 规范身份

事件的规范身份是**严格按契约字段顺序、UTF-8、无空白**的规范 JSON 字节
（字段顺序即 `event_id, observation_id, entity_id, observed_at, score, label, weight,
retract`，`json.dumps(..., separators=(',',':'), ensure_ascii=False)` 保留该插入顺序）。
报告正文等其余结构的规范字节使用排序键，二者都是确定性格式。
批次摘要是对组内规范事件字节按数组顺序拼接后取 SHA256；事件日志摘要采用链式前缀哈希：
`log_digest_N = SHA256(log_digest_(N-1) 的 UTF-8 || 本批规范事件字节)`，首批前缀为空。
因此每次摄取只对本批字节做一次哈希，工作量不随历史增长；副本校验时在复制出的事件范围上
独立重算同一链式摘要进行核对。

### 1.3 提交、重放与冲突（单事务原子语义）

写入使用 `BEGIN IMMEDIATE`，整个批次在一个事务内提交：

1. 批内 `event_id` 重复 → 全拒。
2. 批次摘要与某个已提交批次完全相同（完整规范事件集合与顺序字节等价）→ 重放，返回该批次
   原接收结果（原 `knowledge_seq`、原摘要、`replay=true`），不写入任何行、不推进序号。
3. 否则只要任一 `event_id` 已存在（来自任何旧批次，无论内容是否相同）→ 判定为混合重放/
   内容冲突，全拒。
4. 全新事件：原子追加，提交后取得下一个正整数 `knowledge_seq`，事件在批内的数组位置记为
   `position`。

事件表只追加；任何路径都不执行物理删除（投影重建只删可再生物化行，不删日志）。

## 2. 物化观察视图（按 seq 寻址）

`projections(knowledge_seq PRIMARY KEY, status, view_digest, effective_count, updated_at)` 与
`effective_observations(knowledge_seq, observation_id, ...)`。

对 seq N，折叠规则为：对截止 N 的事件按 `observation_id` 分区，按
`(knowledge_seq DESC, position DESC)` 取最后事件；最后事件 `retract=1` 的 ID 不存在；
其余为有效观察。折叠由 SQLite 窗口函数在引擎内完成，Python 侧以 `fetchmany` 流式读取，
内存不随修订历史总量增长。

`view_digest` 是对有效观察（固定键序
`observation_id, entity_id, observed_at, score, label, weight` 的规范 JSON 字节，按
`observation_id` 排序，逐行加换行）流式 SHA256。

### 2.1 水位推进与崩溃恢复

- 只有新批次成功提交后才允许为该 seq 建立投影；投影状态先置 `building`，重建/计算完成并在
  同一事务内写入全部物化行与摘要后才置 `complete`。不存在“半成品被标为完成”的状态。
- 每次打开账本先 `recover()`：`building` 的投影无条件重建；`complete` 的投影重算
  `view_digest` 比对，不一致即重建。旧 seq 的投影行永不被新 seq 覆盖（表按 seq 主键共存）。
- 报告需要某 seq 时，先恢复再按 seq 顺序补齐所有缺失投影，水位只能向前推进到已提交批次。

## 3. 报告：固定版本、可验证副本、原子发布

request 为 `{"observed_at_cutoff", "knowledge_seq", "split_seed"}`：

- cutoff 同样规范为 UTC 固定宽度字符串；`knowledge_seq` 为正整数或 `"latest"`；
  `split_seed` 为非空字符串。
- 报告启动时在一个短读事务内固定：当前已提交最大 seq；`latest` 解析为该值，数值 seq 大于
  当时最大值即业务错误。同时读取该 seq 的事件日志摘要与事件数。此后报告只代表该固定版本，
  并发提交的更高 seq 不会混入。

### 3.1 读视图方案：不可变事件范围副本 + 完整性校验（验收选定方案）

长 SQLite 读事务虽能直接固定快照，但在并发写入期间会持续保留 WAL 页面。本实现采用副本方案：

1. 将 seq ≤ 固定版本的不可变事件范围与该 seq 的物化有效观察，分块（`fetchmany`）复制到
   账本同目录下的临时 SQLite 副本；复制期间持有一个只读快照事务，复制并校验完成后立即关闭，
   释放 WAL，后续计算完全在副本上进行，不再触碰主库数据页。
2. **副本未校验完整时绝不登记报告**：复制后在副本上独立重算事件日志 SHA256 与事件数，
   并核对 seq 边界与物化视图摘要；任一不符抛出 `ReplicaIncomplete`，删除副本、不写输出、
   不登记、允许同请求重试。
3. 代价是与该 seq 数据量成正比的临时磁盘；内存上界仍只随选定 seq 的有效观察量增长
   （流式复制 + 流式折叠），不随旧修订事件总数增长。

### 3.2 cutoff、撤回折叠与实体切分

- 副本中的有效观察再按 `observed_at <= cutoff` 过滤（固定宽度字符串比较）；越过截止的观察
  与被撤回的 ID 一样不进入任何集合。
- 切分在实体整体上进行：
  `SHA256(seed UTF-8 + NUL + entity_id UTF-8)` 取首 8 字节无符号大端，除以 2^64，
  `< 0.8` 进训练，否则进评估；同一 entity 的全部观察只进一侧。
- 训练集为空 → 业务错误，不写输出文件。评估集可为空，指标按既有空值语义
  （Brier/log-loss 为 `null`，`infinite_log_loss=false`）。

### 3.3 依赖身份与分层缓存

- 训练依赖摘要：排序后的有效训练 observation_id 及其规范值逐行哈希；训练依赖身份直接采用该
  摘要（身份变化 ⇔ 训练成员或其 score/label/weight/有效集合变化）。
- 评估依赖摘要：对评估侧同法计算。
- 校准层（PAVA 训练曲线）按训练依赖身份缓存；评估层按
  `(训练依赖身份, 评估依赖摘要)` 缓存。
  - 仅评估侧 score/label/weight 变化：训练身份不变、曲线复用，评估摘要变化触发本版评估
    重算。
  - 训练成员或训练值变化（含 cutoff 越界、撤回/恢复、entity 跨 split 迁移导致的成员变化）：
    训练身份更新，曲线按新身份取用（重算结果允许与旧曲线恰好相同）。
- 缓存与旧 seq 的视图、报告均按 seq/摘要共存，新版本不覆盖旧版本。

### 3.4 报告内容与幂等发布

报告为确定性 JSON（无时间戳、无随机字段，键排序输出），包含：解析后的完整 request、固定
seq 与事件日志摘要、物化视图摘要、排序后的训练/评估 observation_id 摘要、依赖身份与复用
标记、PAVA 校准块、逐行预测与评估指标、request 摘要与报告摘要。同一账本版本与 request
逐字节相同。

发布规则：输出为符号链接拒绝；输出已存在时，字节完全相同则幂等成功，否则报冲突并保留原
文件、不登记。仅当计算结果对应固定版本且原子落盘（同目录临时文件 + `os.replace`）成功后，
才向 `report_registry` 登记；任何失败保留既有输出并允许同请求重试。

## 4. 并发与资源边界

- 写入串行化：所有 ingestion 使用 `BEGIN IMMEDIATE`；冲突/重复/批内错误整批回滚。
- 报告版本固定先于计算，副本隔离使并发导入无法污染当前报告。
- 内存：所有折叠、摘要、复制均为分块流式；载入内存的业务数据仅选定版本的有效观察与训练
  行。临时磁盘上界为固定版本事件范围大小。
- 崩溃恢复：日志只追加；`building`/失配投影在重启后由日志重建；报告登记晚于原子发布，
  未登记的报告重试时经字节幂等路径收敛。
