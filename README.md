# credit-risk-stability · 贷前风控模型跨期稳定性

## Summary

An origination-time credit risk project on the Freddie Mac Single-Family Loan-Level Dataset (1999–2026). It builds a WOE scorecard and a LightGBM challenger, then measures how they decay across vintages that span the 2008 crisis and COVID forbearance, with a stability dashboard and business recommendations (cutoff, approval lift, limits of reject inference). Every number in the reports is rendered from `outputs/metrics.json` or `config/`, and `make check` enforces leakage, time-split, reproducibility, number-provenance and data red-line checks. Raw loan data is never committed.

**Status:** skeleton — target definition, split and feature availability are drafted; waiting for the data download.

## 这是什么

用 Freddie Mac 公开的房贷贷款级数据，做一个「贷前」违约模型，再看它在不同年份发放的贷款上表现怎么衰减、什么时候该重新校准。

- 目标：发放后 24 个月内逾期 90 天以上或进入止赎（`config/target.yaml`）。
- 切分：按发放季度，训练 2005–2010、验证 2013、跨期测试 2016–2024Q2，相邻段隔 24 个月防止标签成熟期重叠（`config/split.yaml`）。
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

## 数据许可

数据来自 [Freddie Mac Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)，按其条款仅作非商业研究使用。本仓库只发布代码、配置、汇总结果和图，不包含、也不能还原任何贷款级原始数据。
