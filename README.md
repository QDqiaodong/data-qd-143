# data-qd-143 · 评分单调校准

Weighted isotonic score calibration with independent evaluation。业务输入、计算和当前边界详见 [数据契约](docs/contract.md)。

## 安装与演示

Python >=3.11，运行和测试仅用标准库；先确认 `python3 --version`。
每仓建立独立环境，构建工具锁版本。首次安装无公共缓存时可能联网；业务运行不访问网络。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-build.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps .
.venv/bin/python -m pip check
.venv/bin/score-calibration --source fixtures/input.json --output run/report.json
.venv/bin/python -m unittest discover -s tests -v
```

本地已有wheel时，安装构建工具可加 `--no-index --find-links <公共wheel缓存目录>`。
修改源码后重新安装。本仓不共享其他仓环境，不依赖Docker、外部账号或持续服务。

输入JSON最多8MB，拒绝重复键、非有限数值和未定义字段；源字节SHA256随结果保留。
输出是可独立读取的JSON；成功后原子替换，输入/计算失败退出2并保留旧报告。
输入和输出不得相同，输出符号链接拒绝。单操作者批处理，不承诺并发发布或断电恢复。
同输入重跑不累加。测试使用临时目录，运行产物在忽略的run/。

src/score_calibration/model.py 为业务模型，相关算法模块独立组织；common.py 提供受限JSON及报告发布，
cli.py 衔接数据链路。fixtures 为明确自制样例，代码与样例采用MIT许可。

## 修订账本与版本化报告（v2）

仅用标准库与 `sqlite3` 的仅追加修订事件账本，按 `knowledge_seq` 维护物化观察视图并产出可逐字节
复现的版本化校准报告。数据契约见 [docs/contract.md](docs/contract.md)，隔离、读视图副本与崩溃恢复
方案见 [docs/design-ledger.md](docs/design-ledger.md)。

```sh
# 追加一个修订批次（全新事件取得下一个 knowledge_seq；整批重放返回原接收结果且不推进序号）
score-calibration ingest --ledger run/ledger.db --batch path/to/batch.json
# 固定版本计算并原子发布报告；同版本同请求逐字节相同，输出冲突时保留原文件并退出 2
score-calibration report --ledger run/ledger.db --request path/to/request.json --output run/report.json
```

- 批次为 `{"events":[...]}`，事件字段、RFC3339(UTC 规范化)、整数边界与撤回语义见契约文档；
  批内重复、跨批 `event_id` 冲突、混合重放整批回滚；事件永不物理删除。
- 请求固定 `observed_at_cutoff / knowledge_seq(latest 或正整数) / split_seed`；按
  `SHA256(seed + NUL + entity_id)` 首 8 字节做实体整体 80/20 切分，训练为空是业务错误且不写输出。
- 报告先固定版本与事件日志摘要，再把该 seq 不可变事件范围复制到经完整性校验的临时 SQLite 副本后
  计算，并发导入不会让在途报告混入更高 seq；副本校验不过绝不登记报告。
- 物化投影按 seq 共存，`building`/失配投影重启后由日志重建；仅评估侧更正复用训练依赖身份，
  训练侧或成员变化才更新训练身份（曲线值允许恰好相同）。

## 双人盲标规划与答案回收（v3）

仅用标准库构建可续用的自校验 JSON 标注包，为评分样本补充双人盲标人工真值；不写入修订
账本。数据契约见 [docs/contract.md](docs/contract.md)，求解器与补位设计见
[docs/design-annotation.md](docs/design-annotation.md)。

```sh
# 依据 training/pool/annotators/quotas/budget 生成标注包（精确 PAVA 概率与价值、双人精确分配）
score-calibration annotation-plan --source path/to/plan.json --output run/package.json
# 回收（可分批）答案；争议自动封锁 entity 并在剩余预算/容量内补位，重封包
score-calibration annotation-resolve --package run/package.json \
    --answers path/to/answers.json --output run/package.json
```

- 分配目标：先最大化候选价值 `weight*(1-|2p-1|)` 之和，再最大化覆盖数，最后取排序后
  `(id, 较小人员, 较大人员)` 元组序列字典序最小；全程精确有理比较，无价值贪心捷径。
- 任务状态 `open/pending/agreed/disputed`；仅 agreed 进入包内 `truth`（id/score/label/
  weight，可直接作为 v1 分析输入）。争议保留两份已花预算与人员占用、封锁 entity 后补位；
  pending/agreed 原分配固定不挪人。补位不可行时输出完整结果与残余缺口并标 infeasible。
- 包含规范化源数据、训练块、逐候选概率/价值、任务状态、预算/人员用量、组覆盖、真值与
  自校验 SHA256；resolve 重算并核验全部不变量，伪造的 agreed 状态不被采信。
- 合法但 infeasible 退出 0 且包可继续回收；输入/计算错误退出 2 并保留旧输出；允许输出与
  输入包同路径（先读后原子替换），符号链接拒绝，无时间戳或随机字段。

## 当前业务与样例
# Weighted isotonic calibration v1

为评分模型分析者拟合单调分数→经验概率映射，训练和独立评估各<=20000条，行id/score/label/weight；score为0..10^6整数排序分数，不当概率，label严格0/1，weight为1..10^6整数。各集合id唯一，训练与评估id不得重叠，训练非空，评估可空。
同分数先加权合并，再使用PAVA：相邻块经验阳性率下降时合并，得到非递减分段常数概率。相等率块不强制合并。概率为精确Fraction。新分数选择起始分数<=score的最后一块，区间左闭；低于最小夹到第一块、高于最大夹到最后一块。不是插值。
输出训练块边界/权重/阳性权重及评估每行预测、加权Brier与log-loss。Brier精确分数；log-loss用自然对数，出现真实标签对应概率0时返回null并显式infinite_log_loss=true，不裁剪伪造有限值。空评估两指标null。
样例块10→0、20..30→1/2、40→1；两个评估预测1/2与0，加权Brier=1/8，log-loss=ln2/2。评估标签不参与训练。无模型训练、随机划分、自动阈值或线上服务，不声称校准后泛化一定改进。

