<!-- Template: edit reports/model_comparison.tmpl.md, then `make report`. Every number is a placeholder rendered from outputs/metrics.json or config/. -->

# 模型对比：WOE 评分卡 vs LightGBM

## 结论

1. **LightGBM 的领先随时间一路缩小，到最近一段反过来了。** 在同一批重抽样本上算两模型 AUC 之差（LightGBM 减评分卡）：训练 {{metrics.delta_auc_lightgbm_minus_scorecard.train.est:.4f}} → 验证 {{metrics.delta_auc_lightgbm_minus_scorecard.valid.est:.4f}} → 2017–2019 {{metrics.delta_auc_lightgbm_minus_scorecard.2017-2019.est:.4f}} → 2020–2022 {{metrics.delta_auc_lightgbm_minus_scorecard.2020-2022.est:.4f}} → 2023–2024Q2 {{metrics.delta_auc_lightgbm_minus_scorecard.2023-2024.est:.4f}}。{{config.model.evaluation.ci_level:.0%}} 置信区间：训练 {{metrics.delta_auc_lightgbm_minus_scorecard.train.est:.4f}} [{{metrics.delta_auc_lightgbm_minus_scorecard.train.lo:.4f}}, {{metrics.delta_auc_lightgbm_minus_scorecard.train.hi:.4f}}]、验证 {{metrics.delta_auc_lightgbm_minus_scorecard.valid.est:.4f}} [{{metrics.delta_auc_lightgbm_minus_scorecard.valid.lo:.4f}}, {{metrics.delta_auc_lightgbm_minus_scorecard.valid.hi:.4f}}] 显著为正；2017–2019 {{metrics.delta_auc_lightgbm_minus_scorecard.2017-2019.est:.4f}} [{{metrics.delta_auc_lightgbm_minus_scorecard.2017-2019.lo:.4f}}, {{metrics.delta_auc_lightgbm_minus_scorecard.2017-2019.hi:.4f}}]、2020–2022 {{metrics.delta_auc_lightgbm_minus_scorecard.2020-2022.est:.4f}} [{{metrics.delta_auc_lightgbm_minus_scorecard.2020-2022.lo:.4f}}, {{metrics.delta_auc_lightgbm_minus_scorecard.2020-2022.hi:.4f}}] 跨过 0，差别很小；2023–2024Q2 {{metrics.delta_auc_lightgbm_minus_scorecard.2023-2024.est:.4f}} [{{metrics.delta_auc_lightgbm_minus_scorecard.2023-2024.lo:.4f}}, {{metrics.delta_auc_lightgbm_minus_scorecard.2023-2024.hi:.4f}}] 区间刚好在 0 以下，评分卡最多略好，实际差别可以忽略（五段一起比较，没做多重比较校正）。复杂模型多学到的东西没能跨过时间。
2. **测试合计 AUC 不等于段内表现，它混进了段间的基准差，结论以分段为准。** 以 LightGBM 为例，测试合计 AUC {{metrics.lightgbm.segments.test.auc.est:.3f}}，高于它在任何一个测试子段的值（{{metrics.lightgbm.segments.2017-2019.auc.est:.3f}}、{{metrics.lightgbm.segments.2020-2022.auc.est:.3f}}、{{metrics.lightgbm.segments.2023-2024.auc.est:.3f}}）：各段坏账率不同（{{metrics.sample.2017-2019.bad_rate:.2%}} 到 {{metrics.sample.2023-2024.bad_rate:.2%}}），模型给 2023–2024 整体打了更高的分，合并后这种段间差异也被算成了「区分度」。评分卡没有这么明显：它的合计 {{metrics.scorecard.segments.test.auc.est:.3f}} 低于 2023–2024 段的 {{metrics.scorecard.segments.2023-2024.auc.est:.3f}}。分段看，两个模型都是 2017–2019 最低，之后回升。
3. **绝对违约概率普遍偏高，原因不止一个，不能只靠调整基准率修好。**
   - 基准率差：训练段坏账率 {{metrics.sample.train.bad_rate:.2%}}（含 2008 年危机），测试合计 {{metrics.sample.test.bad_rate:.2%}}；评分卡在测试段平均预测 {{metrics.scorecard.segments.test.mean_pred:.2%}}，LightGBM {{metrics.lightgbm.segments.test.mean_pred:.2%}}。
   - 偏高程度随段变化：验证段（坏账率 {{metrics.sample.valid.bad_rate:.2%}}）评分卡只预测 {{metrics.scorecard.segments.valid.mean_pred:.2%}}，2023–2024（坏账率 {{metrics.sample.2023-2024.bad_rate:.2%}}）却预测 {{metrics.scorecard.segments.2023-2024.mean_pred:.2%}}。候选原因是加息后利率回到训练期水平，模型把高利率读成高风险（未验证，交给稳定性分析看利率的 PSI 和分段校准）。所以一个全局的基准率调整没法同时校准 2017–2022 和 2023–2024。
   - LightGBM 概率被压扁：按验证段 AUC 早停在第 {{metrics.lightgbm.best_iteration}} 轮，训练段最低一组预测 {{metrics.lightgbm.segments.train.calibration.0.mean_pred:.2%}}（实际 {{metrics.lightgbm.segments.train.calibration.0.bad_rate:.2%}}），最高一组预测 {{metrics.lightgbm.segments.train.calibration.9.mean_pred:.2%}}（实际 {{metrics.lightgbm.segments.train.calibration.9.bad_rate:.2%}}）；验证段平均预测 {{metrics.lightgbm.segments.valid.mean_pred:.2%}}，是评分卡的两倍多。
   - 宽限剔除压低了测试段的表观坏账率：主样本剔除窗口内带宽限或灾害标记的贷款，含这些贷款时测试合计坏账率是 {{metrics.sample.test_incl_assisted.bad_rate:.2%}}；而训练段在标记字段出现前（{{config.target.assistance.populated_from}} 起才有值），无法同样剔除。见附录。
4. **两个模型靠的是同一批变量。** 评分卡按信息值（IV）、LightGBM 按增益占比，排前两位都是信用分和利率。

## 设置

- 标签：发放后 {{config.target.performance_months}} 个月内逾期 90 天以上（逾期状态 ≥ {{config.target.bad_if_any_in_window.delinquency_status_min}}）、转 REO 或以损失结清（`config/target.yaml`）；主样本剔除缺陷贷款和窗口内带宽限标记的贷款。
- 切分：按发放季度，相邻段隔 {{config.split.gap_months}} 个月（`config/split.yaml`）。
- 特征：只用 `config/features.yaml` 里 `available_at: origination` 且 `use: true` 的字段，共 {{metrics.scorecard.n_candidates}} 个；`make check` 的防泄漏检查对建模表逐列核对。
- 评分卡：训练段分箱（数值最多 {{config.model.woe.max_bins}} 箱加缺失箱，类别占比不到 {{config.model.woe.min_bin_share:.0%}} 的并入其他），IV 低于 {{config.model.woe.iv_min}} 的剔除，留下 {{metrics.scorecard.n_kept}} 个进逻辑回归。
- LightGBM：同样的特征，类别字段原样输入；验证段 AUC 早停，停在第 {{metrics.lightgbm.best_iteration}} 轮。
- 置信区间：每段有放回重抽 {{config.model.evaluation.bootstrap_reps}} 次，两个模型用同一组重抽样本（AUC 之差是配对的），取 {{config.model.evaluation.ci_level:.0%}} 分位区间。KS 在不同分数的阈值上取，同分的贷款一起计入。

| 段 | 贷款数 | 坏样本 | 坏账率 |
| --- | --- | --- | --- |
| 训练 2005–2010 | {{metrics.sample.train.loans:,}} | {{metrics.sample.train.bads:,}} | {{metrics.sample.train.bad_rate:.2%}} |
| 验证 2013–2014 | {{metrics.sample.valid.loans:,}} | {{metrics.sample.valid.bads:,}} | {{metrics.sample.valid.bad_rate:.2%}} |
| 测试 2017–2019 | {{metrics.sample.2017-2019.loans:,}} | {{metrics.sample.2017-2019.bads:,}} | {{metrics.sample.2017-2019.bad_rate:.2%}} |
| 测试 2020–2022 | {{metrics.sample.2020-2022.loans:,}} | {{metrics.sample.2020-2022.bads:,}} | {{metrics.sample.2020-2022.bad_rate:.2%}} |
| 测试 2023–2024Q2 | {{metrics.sample.2023-2024.loans:,}} | {{metrics.sample.2023-2024.bads:,}} | {{metrics.sample.2023-2024.bad_rate:.2%}} |
| 测试合计 | {{metrics.sample.test.loans:,}} | {{metrics.sample.test.bads:,}} | {{metrics.sample.test.bad_rate:.2%}} |

## 区分度（括号内为置信区间）

| 段 | 评分卡 AUC | LightGBM AUC | AUC 之差（LightGBM − 评分卡） | 评分卡 KS | LightGBM KS |
| --- | --- | --- | --- | --- | --- |
| 训练 2005–2010 | {{metrics.scorecard.segments.train.auc.est:.3f}} [{{metrics.scorecard.segments.train.auc.lo:.3f}}, {{metrics.scorecard.segments.train.auc.hi:.3f}}] | {{metrics.lightgbm.segments.train.auc.est:.3f}} [{{metrics.lightgbm.segments.train.auc.lo:.3f}}, {{metrics.lightgbm.segments.train.auc.hi:.3f}}] | {{metrics.delta_auc_lightgbm_minus_scorecard.train.est:.4f}} [{{metrics.delta_auc_lightgbm_minus_scorecard.train.lo:.4f}}, {{metrics.delta_auc_lightgbm_minus_scorecard.train.hi:.4f}}] | {{metrics.scorecard.segments.train.ks.est:.3f}} [{{metrics.scorecard.segments.train.ks.lo:.3f}}, {{metrics.scorecard.segments.train.ks.hi:.3f}}] | {{metrics.lightgbm.segments.train.ks.est:.3f}} [{{metrics.lightgbm.segments.train.ks.lo:.3f}}, {{metrics.lightgbm.segments.train.ks.hi:.3f}}] |
| 验证 2013–2014 | {{metrics.scorecard.segments.valid.auc.est:.3f}} [{{metrics.scorecard.segments.valid.auc.lo:.3f}}, {{metrics.scorecard.segments.valid.auc.hi:.3f}}] | {{metrics.lightgbm.segments.valid.auc.est:.3f}} [{{metrics.lightgbm.segments.valid.auc.lo:.3f}}, {{metrics.lightgbm.segments.valid.auc.hi:.3f}}] | {{metrics.delta_auc_lightgbm_minus_scorecard.valid.est:.4f}} [{{metrics.delta_auc_lightgbm_minus_scorecard.valid.lo:.4f}}, {{metrics.delta_auc_lightgbm_minus_scorecard.valid.hi:.4f}}] | {{metrics.scorecard.segments.valid.ks.est:.3f}} [{{metrics.scorecard.segments.valid.ks.lo:.3f}}, {{metrics.scorecard.segments.valid.ks.hi:.3f}}] | {{metrics.lightgbm.segments.valid.ks.est:.3f}} [{{metrics.lightgbm.segments.valid.ks.lo:.3f}}, {{metrics.lightgbm.segments.valid.ks.hi:.3f}}] |
| 测试 2017–2019 | {{metrics.scorecard.segments.2017-2019.auc.est:.3f}} [{{metrics.scorecard.segments.2017-2019.auc.lo:.3f}}, {{metrics.scorecard.segments.2017-2019.auc.hi:.3f}}] | {{metrics.lightgbm.segments.2017-2019.auc.est:.3f}} [{{metrics.lightgbm.segments.2017-2019.auc.lo:.3f}}, {{metrics.lightgbm.segments.2017-2019.auc.hi:.3f}}] | {{metrics.delta_auc_lightgbm_minus_scorecard.2017-2019.est:.4f}} [{{metrics.delta_auc_lightgbm_minus_scorecard.2017-2019.lo:.4f}}, {{metrics.delta_auc_lightgbm_minus_scorecard.2017-2019.hi:.4f}}] | {{metrics.scorecard.segments.2017-2019.ks.est:.3f}} [{{metrics.scorecard.segments.2017-2019.ks.lo:.3f}}, {{metrics.scorecard.segments.2017-2019.ks.hi:.3f}}] | {{metrics.lightgbm.segments.2017-2019.ks.est:.3f}} [{{metrics.lightgbm.segments.2017-2019.ks.lo:.3f}}, {{metrics.lightgbm.segments.2017-2019.ks.hi:.3f}}] |
| 测试 2020–2022 | {{metrics.scorecard.segments.2020-2022.auc.est:.3f}} [{{metrics.scorecard.segments.2020-2022.auc.lo:.3f}}, {{metrics.scorecard.segments.2020-2022.auc.hi:.3f}}] | {{metrics.lightgbm.segments.2020-2022.auc.est:.3f}} [{{metrics.lightgbm.segments.2020-2022.auc.lo:.3f}}, {{metrics.lightgbm.segments.2020-2022.auc.hi:.3f}}] | {{metrics.delta_auc_lightgbm_minus_scorecard.2020-2022.est:.4f}} [{{metrics.delta_auc_lightgbm_minus_scorecard.2020-2022.lo:.4f}}, {{metrics.delta_auc_lightgbm_minus_scorecard.2020-2022.hi:.4f}}] | {{metrics.scorecard.segments.2020-2022.ks.est:.3f}} [{{metrics.scorecard.segments.2020-2022.ks.lo:.3f}}, {{metrics.scorecard.segments.2020-2022.ks.hi:.3f}}] | {{metrics.lightgbm.segments.2020-2022.ks.est:.3f}} [{{metrics.lightgbm.segments.2020-2022.ks.lo:.3f}}, {{metrics.lightgbm.segments.2020-2022.ks.hi:.3f}}] |
| 测试 2023–2024Q2 | {{metrics.scorecard.segments.2023-2024.auc.est:.3f}} [{{metrics.scorecard.segments.2023-2024.auc.lo:.3f}}, {{metrics.scorecard.segments.2023-2024.auc.hi:.3f}}] | {{metrics.lightgbm.segments.2023-2024.auc.est:.3f}} [{{metrics.lightgbm.segments.2023-2024.auc.lo:.3f}}, {{metrics.lightgbm.segments.2023-2024.auc.hi:.3f}}] | {{metrics.delta_auc_lightgbm_minus_scorecard.2023-2024.est:.4f}} [{{metrics.delta_auc_lightgbm_minus_scorecard.2023-2024.lo:.4f}}, {{metrics.delta_auc_lightgbm_minus_scorecard.2023-2024.hi:.4f}}] | {{metrics.scorecard.segments.2023-2024.ks.est:.3f}} [{{metrics.scorecard.segments.2023-2024.ks.lo:.3f}}, {{metrics.scorecard.segments.2023-2024.ks.hi:.3f}}] | {{metrics.lightgbm.segments.2023-2024.ks.est:.3f}} [{{metrics.lightgbm.segments.2023-2024.ks.lo:.3f}}, {{metrics.lightgbm.segments.2023-2024.ks.hi:.3f}}] |
| 测试合计 | {{metrics.scorecard.segments.test.auc.est:.3f}} [{{metrics.scorecard.segments.test.auc.lo:.3f}}, {{metrics.scorecard.segments.test.auc.hi:.3f}}] | {{metrics.lightgbm.segments.test.auc.est:.3f}} [{{metrics.lightgbm.segments.test.auc.lo:.3f}}, {{metrics.lightgbm.segments.test.auc.hi:.3f}}] | {{metrics.delta_auc_lightgbm_minus_scorecard.test.est:.4f}} [{{metrics.delta_auc_lightgbm_minus_scorecard.test.lo:.4f}}, {{metrics.delta_auc_lightgbm_minus_scorecard.test.hi:.4f}}] | {{metrics.scorecard.segments.test.ks.est:.3f}} [{{metrics.scorecard.segments.test.ks.lo:.3f}}, {{metrics.scorecard.segments.test.ks.hi:.3f}}] | {{metrics.lightgbm.segments.test.ks.est:.3f}} [{{metrics.lightgbm.segments.test.ks.lo:.3f}}, {{metrics.lightgbm.segments.test.ks.hi:.3f}}] |

## 校准

![校准曲线：各段按预测概率分 {{config.model.evaluation.calibration_bins}} 组，横轴平均预测，纵轴实际坏账率，虚线为完全校准；各子图刻度不同](figures/calibration.png)

| 段 | 实际坏账率 | 评分卡平均预测 | LightGBM 平均预测 | 评分卡 Brier | LightGBM Brier |
| --- | --- | --- | --- | --- | --- |
| 训练 2005–2010 | {{metrics.sample.train.bad_rate:.2%}} | {{metrics.scorecard.segments.train.mean_pred:.2%}} | {{metrics.lightgbm.segments.train.mean_pred:.2%}} | {{metrics.scorecard.segments.train.brier:.4f}} | {{metrics.lightgbm.segments.train.brier:.4f}} |
| 验证 2013–2014 | {{metrics.sample.valid.bad_rate:.2%}} | {{metrics.scorecard.segments.valid.mean_pred:.2%}} | {{metrics.lightgbm.segments.valid.mean_pred:.2%}} | {{metrics.scorecard.segments.valid.brier:.4f}} | {{metrics.lightgbm.segments.valid.brier:.4f}} |
| 测试 2017–2019 | {{metrics.sample.2017-2019.bad_rate:.2%}} | {{metrics.scorecard.segments.2017-2019.mean_pred:.2%}} | {{metrics.lightgbm.segments.2017-2019.mean_pred:.2%}} | {{metrics.scorecard.segments.2017-2019.brier:.4f}} | {{metrics.lightgbm.segments.2017-2019.brier:.4f}} |
| 测试 2020–2022 | {{metrics.sample.2020-2022.bad_rate:.2%}} | {{metrics.scorecard.segments.2020-2022.mean_pred:.2%}} | {{metrics.lightgbm.segments.2020-2022.mean_pred:.2%}} | {{metrics.scorecard.segments.2020-2022.brier:.4f}} | {{metrics.lightgbm.segments.2020-2022.brier:.4f}} |
| 测试 2023–2024Q2 | {{metrics.sample.2023-2024.bad_rate:.2%}} | {{metrics.scorecard.segments.2023-2024.mean_pred:.2%}} | {{metrics.lightgbm.segments.2023-2024.mean_pred:.2%}} | {{metrics.scorecard.segments.2023-2024.brier:.4f}} | {{metrics.lightgbm.segments.2023-2024.brier:.4f}} |
| 测试合计 | {{metrics.sample.test.bad_rate:.2%}} | {{metrics.scorecard.segments.test.mean_pred:.2%}} | {{metrics.lightgbm.segments.test.mean_pred:.2%}} | {{metrics.scorecard.segments.test.brier:.4f}} | {{metrics.lightgbm.segments.test.brier:.4f}} |

训练段评分卡贴着对角线，LightGBM 低风险端高估、高风险端低估（概率被压扁）；之后各段两个模型整体落在对角线下方。

## 可解释性

评分卡全部入模特征，按 IV 从高到低：

| 特征 | 评分卡 IV | 评分卡系数（WOE 尺度） | LightGBM 增益占比 |
| --- | --- | --- | --- |
| 信用分（`classic_fico`） | {{metrics.scorecard.features.classic_fico.iv:.3f}} | {{metrics.scorecard.features.classic_fico.coef:.3f}} | {{metrics.lightgbm.gain_share.classic_fico:.1%}} |
| 利率（`original_interest_rate`） | {{metrics.scorecard.features.original_interest_rate.iv:.3f}} | {{metrics.scorecard.features.original_interest_rate.coef:.3f}} | {{metrics.lightgbm.gain_share.original_interest_rate:.1%}} |
| 贷款价值比 LTV（`original_ltv`） | {{metrics.scorecard.features.original_ltv.iv:.3f}} | {{metrics.scorecard.features.original_ltv.coef:.3f}} | {{metrics.lightgbm.gain_share.original_ltv:.1%}} |
| 综合 LTV（`original_cltv`） | {{metrics.scorecard.features.original_cltv.iv:.3f}} | {{metrics.scorecard.features.original_cltv.coef:.3f}} | {{metrics.lightgbm.gain_share.original_cltv:.1%}} |
| 负债收入比 DTI（`original_dti`） | {{metrics.scorecard.features.original_dti.iv:.3f}} | {{metrics.scorecard.features.original_dti.coef:.3f}} | {{metrics.lightgbm.gain_share.original_dti:.1%}} |
| 贷款期限（`original_loan_term`） | {{metrics.scorecard.features.original_loan_term.iv:.3f}} | {{metrics.scorecard.features.original_loan_term.coef:.3f}} | {{metrics.lightgbm.gain_share.original_loan_term:.1%}} |
| 按揭保险比例（`mi_pct`） | {{metrics.scorecard.features.mi_pct.iv:.3f}} | {{metrics.scorecard.features.mi_pct.coef:.3f}} | {{metrics.lightgbm.gain_share.mi_pct:.1%}} |
| 渠道（`channel`） | {{metrics.scorecard.features.channel.iv:.3f}} | {{metrics.scorecard.features.channel.coef:.3f}} | {{metrics.lightgbm.gain_share.channel:.1%}} |
| 借款人数（`number_of_borrowers`） | {{metrics.scorecard.features.number_of_borrowers.iv:.3f}} | {{metrics.scorecard.features.number_of_borrowers.coef:.3f}} | {{metrics.lightgbm.gain_share.number_of_borrowers:.1%}} |
| 州（`property_state`） | {{metrics.scorecard.features.property_state.iv:.3f}} | {{metrics.scorecard.features.property_state.coef:.3f}} | {{metrics.lightgbm.gain_share.property_state:.1%}} |
| 都市区 MSA（`msa`） | {{metrics.scorecard.features.msa.iv:.3f}} | {{metrics.scorecard.features.msa.coef:.3f}} | {{metrics.lightgbm.gain_share.msa:.1%}} |
| 贷款用途（`loan_purpose`） | {{metrics.scorecard.features.loan_purpose.iv:.3f}} | {{metrics.scorecard.features.loan_purpose.coef:.3f}} | {{metrics.lightgbm.gain_share.loan_purpose:.1%}} |
| 贷款金额（`original_upb`） | {{metrics.scorecard.features.original_upb.iv:.3f}} | {{metrics.scorecard.features.original_upb.coef:.3f}} | {{metrics.lightgbm.gain_share.original_upb:.1%}} |

- 没进评分卡的特征（IV 低于门槛）在 LightGBM 里增益占比也都很低，最高的入住类型只有 {{metrics.lightgbm.gain_share.occupancy_status:.2%}}。
- 评分卡每个特征先换成 WOE，系数为正表示和 WOE 方向一致；系数远离 1 的（如贷款金额）说明和其他特征有重叠，单看系数不能读成「影响大小」。
- LightGBM 给都市区的权重（{{metrics.lightgbm.gain_share.msa:.1%}}）明显高于评分卡里它的 IV 排位：高基数类别容易被树记住地区差异，这部分跨期最不稳，是它跨期领先消失的一个候选原因（未验证）。

## 局限

- 验证段同时用于 LightGBM 早停，它在验证段的成绩偏乐观；跨期测试段两个模型都没碰过。早停按 AUC，概率没收敛，留到再校准一步处理。
- 置信区间只重抽评估样本，不含训练样本的波动。
- 评分卡分箱没有强制单调，个别箱的 WOE 可能来回跳；要上线前应加单调约束和人工合箱。
- 用的是官方抽样文件（每个发放年随机抽样），不是全量。

## 附录：含宽限贷款的测试段

主结果剔除窗口内带宽限（F）或灾害标记的贷款。把它们放回测试段（模型不变）：

| 口径 | 贷款数 | 坏账率 | 评分卡 AUC | LightGBM AUC | AUC 之差 |
| --- | --- | --- | --- | --- | --- |
| 测试合计 | {{metrics.sample.test.loans:,}} | {{metrics.sample.test.bad_rate:.2%}} | {{metrics.scorecard.segments.test.auc.est:.3f}} [{{metrics.scorecard.segments.test.auc.lo:.3f}}, {{metrics.scorecard.segments.test.auc.hi:.3f}}] | {{metrics.lightgbm.segments.test.auc.est:.3f}} [{{metrics.lightgbm.segments.test.auc.lo:.3f}}, {{metrics.lightgbm.segments.test.auc.hi:.3f}}] | {{metrics.delta_auc_lightgbm_minus_scorecard.test.est:.4f}} [{{metrics.delta_auc_lightgbm_minus_scorecard.test.lo:.4f}}, {{metrics.delta_auc_lightgbm_minus_scorecard.test.hi:.4f}}] |
| 测试合计（含宽限） | {{metrics.sample.test_incl_assisted.loans:,}} | {{metrics.sample.test_incl_assisted.bad_rate:.2%}} | {{metrics.scorecard.segments.test_incl_assisted.auc.est:.3f}} [{{metrics.scorecard.segments.test_incl_assisted.auc.lo:.3f}}, {{metrics.scorecard.segments.test_incl_assisted.auc.hi:.3f}}] | {{metrics.lightgbm.segments.test_incl_assisted.auc.est:.3f}} [{{metrics.lightgbm.segments.test_incl_assisted.auc.lo:.3f}}, {{metrics.lightgbm.segments.test_incl_assisted.auc.hi:.3f}}] | {{metrics.delta_auc_lightgbm_minus_scorecard.test_incl_assisted.est:.4f}} [{{metrics.delta_auc_lightgbm_minus_scorecard.test_incl_assisted.lo:.4f}}, {{metrics.delta_auc_lightgbm_minus_scorecard.test_incl_assisted.hi:.4f}}] |

放回后坏账率大幅上升、两个模型 AUC 都明显下降；这个口径下 AUC 之差 {{metrics.delta_auc_lightgbm_minus_scorecard.test_incl_assisted.est:.4f}} [{{metrics.delta_auc_lightgbm_minus_scorecard.test_incl_assisted.lo:.4f}}, {{metrics.delta_auc_lightgbm_minus_scorecard.test_incl_assisted.hi:.4f}}] 不跨 0，是测试口径里唯一一处 LightGBM 显著领先，但差距很小。宽限贷款被剔除是因为：处在宽限里的贷款即使没有真实违约也会显示逾期（`docs/field_manual.md` 已核对），这类「坏」和发放时的风险特征关系弱，所以主结果剔除它们；但这也意味着主样本的测试坏账率偏低，和训练段不完全同口径。
