# credit-risk-stability · 贷前风控模型跨期稳定性

## Summary

An origination-time credit risk project on the Freddie Mac Single-Family Loan-Level Dataset (1999–2026). It builds a WOE scorecard and a LightGBM challenger, then measures how they decay across vintages that span the 2008 crisis and COVID forbearance, with a stability dashboard and business recommendations (cutoff, approval lift, limits of reject inference). Every number in the reports is rendered from `outputs/metrics.json` or `config/`, and `make check` enforces leakage, time-split, reproducibility, number-provenance and data red-line checks. Raw loan data is never committed.

**Status:** data, target, split and the five `make check` checks are in place; modelling (scorecard + LightGBM) is next.

## 这是什么

用 Freddie Mac 公开的房贷贷款级数据，做一个「贷前」违约模型，再看它在不同年份发放的贷款上表现怎么衰减、什么时候该重新校准。

- 目标：发放后 24 个月内逾期 90 天以上或进入止赎（`config/target.yaml`）。
- 切分：按发放季度，训练 2005–2010、验证 2013–2014、跨期测试 2017–2024Q2，相邻段隔 24 个月防止标签成熟期重叠（`config/split.yaml`）。
- 特征：只用发放时就知道的字段，每个字段标了可用时点（`config/features.yaml`）。
- 字段说明：`docs/field_manual.md`；按年坏样本数：`docs/bad_counts.md`。

## 怎么跑

```bash
make setup        # 安装依赖、启用提交钩子
# 把 Clarity 下载的 sample_YYYY.zip 解压到 data/raw/（不进仓库）
make manifest     # 原始文件 sha256 → data/manifest.sha256
make bad-counts   # 按年坏样本数 → docs/bad_counts_raw.md
make check
```

## 硬检查（`make check`）

`make check` 跑测试和 `checks/` 下 5 个检查，任何一个不过就退出码 1 并列出违规位置；GitHub Actions 每次 push 和 PR 都跑。`make setup` 启用提交钩子，提交前对暂存文件跑红线检查。

| 检查 | 拦什么 | 放行名单 |
| --- | --- | --- |
| `check_leakage` | 模型特征不是发放时可得；建模表里出现非特征列；训练段单特征 AUC 超过门槛 | `config/leakage_allowlist.yaml` |
| `check_split` | 相邻段不满 24 个月隔离期；测试段离表现截止不够；非日历截尾；某段坏样本不够下限；建模表里贷款跨段、分错段、未截尾 | — |
| `check_repro` | 依赖没锁；开发子集跑两次指标不一致，或和提交的参考值不一致 | — |
| `check_numbers` | 报告、作品集页、看板文案里的数字对不上 `metrics.json`、`config/` 或常量名单 | `config/constants_allowlist.yaml` |
| `check_redline` | 原始和中间数据、行级数据文件、Notebook 输出、贷款号、原始行样例、密钥、关键词黑名单 | `config/redline_allowlist.yaml` |

- 建模表约定：`data/interim/model_frame.parquet`，一行一笔贷款，列为特征加 `loan_id`、`segment`、`bad`、`first_payment_date` 等（见 `checks/common.py`）。有这个文件时，防泄漏和时间切分检查会连数据一起查。
- 原始文件指纹：`make verify-raw` 重算 sha256 并和 `data/manifest.sha256` 比对。
- 关键词黑名单不进仓库，放本机 `~/.config/crs/redline_blacklist.txt`（或环境变量 `CRS_REDLINE_BLACKLIST`），一行一个词。

## 数据许可

数据来自 [Freddie Mac Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)，按其条款仅作非商业研究使用。本仓库只发布代码、配置、汇总结果和图，不包含、也不能还原任何贷款级原始数据。
