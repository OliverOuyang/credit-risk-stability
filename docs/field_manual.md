# 字段手册（Freddie Mac Single-Family Loan-Level Dataset）

只摘公开文档，不含任何贷款数据。来源（2026-11-27 下载）：
- [User Guide（2026 年 10 月版）](https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf)
- [File Layout（字段位置表）](https://www.freddiemac.com/fmac-resources/research/pdf/file_layout.xlsx)
- [Release Notes（Release 48）](https://www.freddiemac.com/fmac-resources/research/pdf/release_notes.pdf)
- [FAQ](https://www.freddiemac.com/fmac-resources/research/pdf/faq.pdf)

这四份公开文档也建成了 NotebookLM 笔记本「Freddie Mac SFLLD 字段手册（公开文档）」（私有，仅作者可开），用于带引用查字段含义；笔记本里不放任何贷款数据。2026-11-27 用它核对了宽限期字段，答案和下文一致。

## 数据版本和截止日

| 项 | 值 | 出处 |
| --- | --- | --- |
| 当前版本 | Release 48，2026-10-28 发布 | Release Notes 第 2 页 |
| 发放截止 | 2026-06-30 | 同上 |
| 表现截止 | 2026-06-30 | 同上 |
| 抽样文件 | 每个完整年份随机 5 万笔，未满年份按比例；`sample_YYYY.zip` 内含 `sample_orig_YYYY.txt`、`sample_perf_YYYY.txt`，竖线分隔、无表头 | User Guide「Dataset Sample」「Naming Convention」 |
| 下载方式 | 注册并登录 Clarity Data Intelligence 后下载 | User Guide「Accessing the Dataset」 |

**和设计方案的差异**：方案写的表现截止是 2026-03（Release 47）。现行 Release 48 已到 2026-06，测试段末尾因此从 2024-03 顺延到 2024Q2（`config/split.yaml`）。已按下载文件核对：抽样表现文件最后一个月是 2026-06，一致。

## 文件结构

- 发放文件 31 个字段，全部是发放时已知，可当贷前特征；逐个字段的可用时点和是否入模见 `config/features.yaml`。
- 月度表现文件 35 个字段，只用来造标签，不进模型。
- Release 47 起：信用分字段改名 Classic FICO，新增 VantageScore 4.0；Servicer Name 等从发放文件移到表现文件。老年份的 VantageScore 覆盖率要在数据到手后看，先不入模。
- 贷款号格式 `PYYQnXXXXXXX`：P 是产品（F 固定利率、A 浮动利率），YYQn 是发放年和季度，后 7 位随机。切分用的发放季度就从这里取。

## 造标签要用的字段

| 字段（表现文件） | 取值 | 用法 |
| --- | --- | --- |
| Current Loan Delinquency Status | 00 正常、01 30–59 天、02 60–89 天、03 90–119 天……封顶 99；RA = 转为 REO；XX = 无数据 | ≥03 或 RA 记坏；XX 剔除 |
| Loan Age | 从首次还款月起算的期数 | 表现窗口取第 1–24 期 |
| Zero Balance Code | 01 提前还清/到期、02 第三方拍卖、03 短售或核销、09 REO 处置、15 整贷出售、16 再履约证券化、96 确认有承销或服务缺陷 | 02/03/09 记坏；01 记好不剔除；96 剔除；15/16 暂按非坏处理，数据到手后看数量 |

## 宽限期（疫情、灾害）字段：存在

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| Borrower Assistance Plan | F 宽限（Forbearance）、R 还款计划、T 试用期、空 = 无 | 不论逾期状态都会填；只从 2014-01 起的月份有值。只用 F 识别宽限：R、T 一般跟在普通逾期之后，拿来剔除会误删真实坏样本 |
| Delinquency Due to Disaster | Y / 空 | 服务商报告了灾害导致的困难，可能处于宽限；只从 2014-01 起有值 |
| Payment Deferral Flag | C 本期、P 以前 | 延期还款，不算贷款修改 |

要点：逾期状态是按「最后已还期次的应还日」算的，所以处在宽限里、没有还款的贷款**照样会显示逾期**（已核对：2020 年抽样里带 F 标记的 9474 个贷款月，只有 1444 个是正常状态，其余都显示逾期）。这和设计方案里「宽限期不记为逾期」的说法不同——数据里会记，只是有标记可以识别。做法不变：24 个月窗口里任一月带上述标记的贷款单列，主结果剔除，附录给含宽限的对比（`config/target.yaml` 的 `assistance` 段）。2014 年以前没有这两个标记，2008 年前后的危机期无法用它识别，只能靠年份讲。

## 违约定义（写进 `config/target.yaml`）

- 观测点：发放时。窗口：首次还款后第 1–24 期。
- 坏：窗口内任一月逾期 90 天以上（状态 ≥03）、转为 REO，或以第三方拍卖、短售/核销、REO 处置结清。
- 好：窗口内没有坏事件；24 个月内提前还清的也算好，不剔除。
- 截尾按日历：首次还款月 + 23 个月 ≤ 表现截止月才进集合。
