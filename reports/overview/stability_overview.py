"""One-page HTML overview of reports/stability.md for non-specialist readers: every section is a chart first,
then 3-4 plain-language Q&As. All numbers come from outputs/stability.json and app/data/agg/quarterly.csv.
Run: PYTHONPATH=src uv run python reports/overview/stability_overview.py  ->  reports/stability_overview.html
Template (overview_tpl.py, style.css) is vendored from the private agent-eval-docs overview template."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from overview_tpl import E, QA, arrow, box, fig, page, sec, svg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
S = json.loads((ROOT / "outputs" / "stability.json").read_text())
Q = list(csv.DictReader(open(ROOT / "app" / "data" / "agg" / "quarterly.csv", encoding="utf-8")))
A = S["annual"]
YEARS = list(A)
SC = "scorecard"
C = {"fixed": "var(--must)", "no_crisis": "var(--should)", "rolling": "var(--lf)", "actual": "var(--ink)",
     "policy": "var(--mc)", "train": "var(--must)", "ref": "var(--lf)", "muted": "var(--muted)"}


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def yr(y):
    return "2024上半年" if y == "2024" else y


def chart(label, xs, series, lo, hi, ticks, fmt, bands=(), hlines=(), w=760, h=300, every=1):
    """Line chart. series = [(name, color, values, dashed)]; None values break the line."""
    L, R, T, B = 58, 150, 18, 34
    pw, ph = w - L - R, h - T - B
    X = lambda i: L + pw * i / max(len(xs) - 1, 1)
    Y = lambda v: T + ph * (1 - (min(max(v, lo), hi) - lo) / (hi - lo))
    out = ""
    for a, b, t in bands:  # shaded periods
        x0, x1 = X(a) - pw / max(len(xs) - 1, 1) / 2, X(b) + pw / max(len(xs) - 1, 1) / 2
        out += f'<rect x="{x0:.1f}" y="{T}" width="{x1 - x0:.1f}" height="{ph}" fill="var(--wash)"/>'
        out += f'<text x="{(x0 + x1) / 2:.1f}" y="{T + 14}" class="c s">{E(t)}</text>'
    for v in ticks:
        out += f'<line x1="{L}" x2="{L + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--line)"/>'
        out += f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" class="s" style="text-anchor:end">{fmt(v)}</text>'
    for v, t, col in hlines:
        out += f'<line x1="{L}" x2="{L + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{col}" stroke-dasharray="6 4"/>'
        out += f'<text x="{L + pw + 6}" y="{Y(v) + 4:.1f}" class="s" style="fill:{col}">{E(t)}</text>'
    for i, x in enumerate(xs):
        if i % every == 0:
            out += f'<text x="{X(i):.1f}" y="{h - 12}" class="c s">{E(x)}</text>'
    ends = []
    for name, col, vals, dash in series:
        pts, seg = [], []
        for i, v in enumerate(vals):
            if v is None:
                if seg:
                    pts.append(seg)
                seg = []
            else:
                seg.append((X(i), Y(v)))
        if seg:
            pts.append(seg)
        for sgm in pts:
            d = " ".join(f"{'M' if k == 0 else 'L'}{x:.1f} {y:.1f}" for k, (x, y) in enumerate(sgm))
            out += (f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.4"'
                    f'{" stroke-dasharray=\"7 4\"" if dash else ""}/>')
            if len(xs) <= 12:
                out += "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{col}"/>' for x, y in sgm)
        last = [(i, v) for i, v in enumerate(vals) if v is not None][-1]
        ends.append([Y(last[1]), name, col])
    ends.sort()
    for k in range(1, len(ends)):  # keep end labels apart
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 15)
    if ends and ends[-1][0] > T + ph:  # pushed below the plot: stack upwards from the bottom instead
        ends[-1][0] = T + ph
        for k in range(len(ends) - 2, -1, -1):
            ends[k][0] = min(ends[k][0], ends[k + 1][0] - 15)
    out += "".join(f'<text x="{L + pw + 8}" y="{y + 4:.1f}" style="fill:{col};font-weight:600">{E(n)}</text>'
                   for y, n, col in ends)
    return svg(w, h, label, out)


def bars(label, xs, groups, lo, hi, ticks, fmt, w=760, h=270):
    """Grouped bars. groups = [(name, color, values)]."""
    L, R, T, B = 58, 150, 18, 34
    pw, ph = w - L - R, h - T - B
    Y = lambda v: T + ph * (1 - (v - lo) / (hi - lo))
    out = ""
    for v in ticks:
        out += f'<line x1="{L}" x2="{L + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--line)"/>'
        out += f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" class="s" style="text-anchor:end">{fmt(v)}</text>'
    slot = pw / len(xs)
    bw = slot * 0.8 / len(groups)
    for i, x in enumerate(xs):
        for j, (_, col, vals) in enumerate(groups):
            x0 = L + slot * i + slot * 0.1 + bw * j
            out += f'<rect x="{x0:.1f}" y="{Y(vals[i]):.1f}" width="{bw - 2:.1f}" height="{Y(lo) - Y(vals[i]):.1f}" fill="{col}" rx="2"/>'
        out += f'<text x="{L + slot * (i + .5):.1f}" y="{h - 12}" class="c s">{E(x)}</text>'
    for j, (name, col, _) in enumerate(groups):
        out += f'<rect x="{L + pw + 10}" y="{T + 22 * j}" width="12" height="12" fill="{col}" rx="2"/>'
        out += f'<text x="{L + pw + 28}" y="{T + 22 * j + 11}" style="fill:{col};font-weight:600">{E(name)}</text>'
    return svg(w, h, label, out)


a = lambda y, *k: _get(A[y], k)


def _get(x, k):
    for i in k:
        x = x[i]
    return x


R = S["rules"][SC]
import yaml  # noqa: E402
CFG = yaml.safe_load((ROOT / "config" / "stability.yaml").read_text())
BAND = "–".join(str(x) for x in CFG["rules"]["oe_band"])
OVER = A["2023"][SC]["fixed"]["mean_pred"] / A["2023"]["sample"]["bad_rate"]
fx = {y: a(y, SC, "fixed", "auc", "est") for y in YEARS}
lo_y = min(fx, key=fx.get)
cov = [YEARS.index("2019"), YEARS.index("2020")]

# ---------- 0 三个指标 ----------
g0 = svg(760, 150, "三个指标：AUC 看排序，O/E 看概率准不准，PSI 看新客户变没变",
         box(10, 14, 236, 120, "st lf", "AUC：会不会排序", "把坏客户排在好客户前面的能力",
             "0.5 = 瞎猜，1 = 完美；0.75 以上算好") +
         box(262, 14, 236, 120, "st mc", "O/E：概率准不准", "实际违约人数 ÷ 模型预测的违约人数",
             "1 = 准；小于 1 = 模型说得太悲观") +
         box(514, 14, 236, 120, "loc", "PSI：客户变没变", "新客户的分数分布和参照期比，差多少",
             "< 0.1 稳；0.1–0.25 观察；> 0.25 要查"))
s0 = sec("s0", "先看懂三个指标", [fig(g0)], QA(
    ("为什么要三个指标？", "模型可能「排序还行但概率不准」，也可能「客户变了但模型还没坏」。三个指标各看一面，不能互相替代。"),
    ("固定模型、去掉危机、每年重训是什么？",
     "固定模型 = 用 2005–2010 年的贷款训练一次就一直用；去掉危机 = 同样的老数据但删掉 2007–2009 年；"
     "每年重训 = 每年年初用最近 5 年（已知结果的）贷款重新训练。"),
    ("数据是什么？", f"Freddie Mac 公开的房贷数据，每年抽 5 万笔，从 2005 年看到 2024 年上半年；"
                   f"「坏客户」= 放款后 24 个月内逾期 90 天以上。下面的图都用评分卡模型，LightGBM 的结论在问答里说。"),
), h="h2")

# ---------- 1 排序能力 ----------
g1 = chart("每年新贷款上的 AUC：固定模型最低，2020 年跌到谷底；每年重训一直最高",
           [yr(y) for y in YEARS],
           [("固定模型", C["fixed"], [fx[y] for y in YEARS], False),
            ("去掉危机年份", C["no_crisis"], [a(y, SC, "no_crisis", "auc", "est") for y in YEARS], True),
            ("每年重训", C["rolling"], [a(y, SC, "rolling", "auc", "est") for y in YEARS], False)],
           0.60, 0.85, [0.60, 0.65, 0.70, 0.75, 0.80, 0.85], lambda v: f"{v:.2f}",
           bands=[(cov[0], cov[1], "疫情宽限期贷款")])
s1 = sec("s1", "一　模型放到新年份还会排序吗？会，但变差了", [fig(g1)], QA(
    ("这张图看什么？", "横轴是贷款发放年份，纵轴是模型在当年新贷款上的 AUC，越高越会区分好坏客户。三条线是三种用法。"),
    ("看到了什么？", f"固定模型从 2017 年的 {fx['2017']:.2f} 一路掉到 {lo_y} 年的 {fx[lo_y]:.2f}，之后回升到 "
                    f"{fx['2024']:.2f}。每年重训的线几乎每年都在最上面（{len(R['years_rolling_better'])}/{len(YEARS)} 年明显更好）。"),
    ("为什么变差？", "两个原因叠加：① 训练数据里有 2008 危机，那几年的坏客户和后来的不一样，只删掉危机年份（黄色虚线）就补回一部分；"
                   f"② 没用上近期数据，近似「老化」，重训比只删危机还好的年份有 {'、'.join(R['years_rolling_better_than_no_crisis'])}。两者哪个大，每年不一样。"),
    ("所以呢？", f"建议每年重训，训练时删掉危机年份。LightGBM 趋势一样，但重训的好处小，只有 {len(S['rules']['lightgbm']['years_rolling_better'])} 年明显更好。"),
), h="h2")

# ---------- 2 疫情年份 ----------
g2 = bars("2018–2021 年两种口径的 AUC：不剔除宽限贷款时更高，但 2019–2020 年仍低于 2017 年",
          [yr(y) for y in YEARS],
          [("剔除宽限贷款", C["fixed"], [fx[y] for y in YEARS]),
           ("不剔除", C["muted"], [a(y, SC, "incl_assisted", "auc") for y in YEARS])],
          0.60, 0.85, [0.60, 0.65, 0.70, 0.75, 0.80, 0.85], lambda v: f"{v:.2f}")
br = lambda y: (A[y]["sample"]["bad_rate"], A[y]["sample"]["bad_rate_incl_assisted"])
s2 = sec("s2", "二　2019–2020 年为什么掉得最多？一部分是口径问题", [fig(g2)], QA(
    ("宽限贷款是什么？", "疫情期间借款人可以申请暂停还款。Freddie Mac 的数据里，暂停期间也会显示「逾期」，所以这些贷款算不算坏客户很难说。"),
    ("两种口径差多少？", f"2019 年发放的贷款，剔除宽限贷款后坏账率 {pct(br('2019')[0], 2)}，不剔除是 {pct(br('2019')[1], 2)}，差了约 {br('2019')[1] / br('2019')[0]:.0f} 倍。"
                       "剔除会偏低（拿掉了不少真有问题的人），不剔除会偏高（把暂停还款也当成违约）。"),
    ("看到了什么？", f"不剔除时 2019、2020 年 AUC 是 {a('2019', SC, 'incl_assisted', 'auc'):.2f}、{a('2020', SC, 'incl_assisted', 'auc'):.2f}，"
                   f"比剔除口径高，但仍低于 2017 年的 {a('2017', SC, 'incl_assisted', 'auc'):.2f}。"),
    ("所以呢？", "这两年的下跌一部分是口径造成的，一部分是模型真的变弱了，两者没法定量拆开。看这两年的结论要两种口径一起看。"),
), h="h2")

# ---------- 3 概率准不准 ----------
g3 = chart(f"预测违约率和实际违约率：固定模型越来越高估，2023 年高估约 {OVER:.0f} 倍；删掉危机年份能减半",
           [yr(y) for y in YEARS],
           [("实际违约率", C["actual"], [A[y]["sample"]["bad_rate"] for y in YEARS], False),
            ("固定模型预测", C["fixed"], [a(y, SC, "fixed", "mean_pred") for y in YEARS], False),
            ("删掉危机后预测", C["no_crisis"], [a(y, SC, "no_crisis", "mean_pred") for y in YEARS], True),
            ("按规则校准后", C["policy"], [a(y, SC, "policy", "mean_pred") for y in YEARS], True)],
           0, 0.05, [0, 0.01, 0.02, 0.03, 0.04, 0.05], lambda v: pct(v, 0))
p23 = a("2023", SC, "fixed", "mean_pred")
s3 = sec("s3", "三　模型给的违约概率准吗？明显偏高", [fig(g3)], QA(
    ("这张图看什么？", "黑线是每年实际违约率，其他线是模型平均预测的违约率。线越贴近黑线越准。"),
    ("偏高多少？", f"2023 年实际 {pct(A['2023']['sample']['bad_rate'], 2)}，固定模型预测 {pct(p23, 2)}，约 {p23 / A['2023']['sample']['bad_rate']:.0f} 倍。"
                 f"删掉危机年份后降到 {pct(a('2023', SC, 'no_crisis', 'mean_pred'), 2)}。"),
    ("校准能修好吗？", f"按真实时间（年初只能用 3 年前的结果）每年校准一次，8 年里只有 {'、'.join(R['years_policy_in_band'])} 年校到可接受范围"
                     f"（O/E 在 {BAND}）。2021–2023 年参照年里疫情宽限贷款太多，规则主动不校、交人工，所以紫线和红线在这几年重合。"),
    ("所以呢？", "违约水平跟着利率、房价、政策逐年变，3 年前的数据追不上。光靠校准不够，要重训，再加人工对宏观形势的判断。"),
), h="h2")

# ---------- 4 监控 ----------
qs = [r for r in Q if r["segment"] == "test"]
fl = lambda r, k: float(r[k]) if r[k] else None
g4 = chart(f"每季度新客户的分数变化（PSI）：和 2005–2010 训练期比几乎一直超线；和上线前 4 个季度比只在 {len(R['ref_rolling_fired_act'])} 个季度超线",
           [r["quarter"] for r in qs],
           [("和训练期比", C["train"], [fl(r, "psi_scorecard") for r in qs], False),
            ("和上线前 4 季比", C["ref"], [fl(r, "psi_ref_rolling_scorecard") for r in qs], False)],
           0, 1.6, [0, 0.4, 0.8, 1.2, 1.6], lambda v: f"{v:.1f}",
           hlines=[(0.1, "0.1 观察线", C["muted"]), (0.25, "0.25 行动线", C["fixed"])], every=4,
           bands=[(next(i for i, r in enumerate(qs) if r["quarter"] == "2020Q1"),
                   next(i for i, r in enumerate(qs) if r["quarter"] == "2021Q4"), "疫情")])
s4 = sec("s4", "四　什么时候该报警？要和最近的客户比", [fig(g4)], QA(
    ("这张图看什么？", "每个点是一个季度的新客户，纵轴是 PSI：新客户分数分布和参照期差多少。超过 0.25 那条线就该查原因。"),
    ("为什么红线一直超线？", f"和 2005–2010 年训练期比，{R['test_quarters_act']}/{S['quarters']['test']} 个季度都超行动线，等于一直亮灯。"
                         f"主要因为利率：2012–2021 年利率低，和训练期差太远。"),
    ("绿线说明什么？", f"改成和「上线前最近 4 个季度」比，只有 {'、'.join(R['ref_rolling_fired_act'])} 连续两季超线，"
                     "正好是疫情前后的再融资潮和 2022 年加息，都是真实的客户变化。"),
    ("绿线为什么每年初掉回 0 附近？", "每年年初模型重训一次，参照也换成最近 4 个季度，所以新一年的第一季度和参照很像，之后逐季拉开。"),
    ("报警了做什么？", "先查是哪个变量变了（利率、贷款用途、DTI），判断是市场变了还是客户变了，不直接重训。"),
), h="h2")

# ---------- 5 规则 ----------
g5 = svg(760, 250, "三条规则：每季度看客户变化，每年看概率准不准和排序能力，触发后分别查原因、校准或重训",
         box(10, 20, 220, 60, "st lf", "每季度：新客户打分", "不需要等结果") +
         box(270, 20, 220, 60, "card", "R1 客户变了吗", "PSI vs 上线前 4 季") +
         box(530, 20, 220, 60, "pp", "查原因", "利率？客群？编码改了？") +
         box(10, 140, 220, 60, "st mc", "每年年初：3 年前的结果", "那批贷款 24 个月已到期") +
         box(270, 110, 220, 54, "card", "R2 概率准吗", f"O/E 出了 {BAND}") +
         box(270, 180, 220, 54, "card", "R3 排序变差了吗", "AUC 低于验证期") +
         box(530, 110, 220, 54, "pp", "校准概率", "疫情宽限多的年份改交人工") +
         box(530, 180, 220, 54, "st bad", "重训 + 人工判断", "校准两次仍不准也到这") +
         arrow(230, 50, 268, 50) + arrow(490, 50, 528, 50) + arrow(230, 160, 268, 140) + arrow(230, 175, 268, 205) +
         arrow(490, 137, 528, 137) + arrow(490, 207, 528, 207) + arrow(640, 164, 640, 178, "dash"))
s5 = sec("s5", "五　建议的监控规则", [fig(g5)], QA(
    ("为什么分「每季度」和「每年」？", "违约要 24 个月后才知道，年初最多只能看到 3 年前那批贷款的结果。只有 R1 不需要等，当季就能报。"),
    ("回测里触发了多少次？", f"R1 评分卡 {len(R['ref_rolling_fired_act'])} 个季度；R2 每年都触发；R3 有 {len(R['years_retrain'])}/{len(YEARS)} 年触发。"
                         "说明「训练一次一直用」行不通。"),
    ("阈值从哪来？", f"PSI 的 0.1 / 0.25 是行业惯例；O/E 容忍带 {BAND}、连续 {CFG['rules']['consecutive_quarters']} 季、宽限占比 2% 是本项目定的，其中 2% 是看了回测才定的，没独立验证。"),
), h="h2")

# ---------- 6 局限 ----------
s6 = sec("s6", "六　还没解决的", [], QA(
    ("样本小", "每年抽 5 万笔，2019–2021 年剔除宽限后坏客户只有一两百个，逐年结论误差大。2020、2021 年全量数据已下载，还没用来复核。"),
    ("没有独立复跑", "代码被另一家模型（Gemini）和顾问审过，但都只读了代码，没人拿原始数据重跑一遍核对结果。"),
    ("渠道变量", "2009 年起渠道的编码改了，评分卡把编码变化当成了风险信号，模型里还没合并。"),
), h="h2")

nav = [("s0", "指标"), ("s1", "排序"), ("s2", "疫情"), ("s3", "概率"), ("s4", "监控"), ("s5", "规则"), ("s6", "局限")]
out = ROOT / "reports" / "stability_overview.html"
page(str(out), "跨期稳定性总览", "credit-risk-stability", nav,
     "贷前风控模型：放到新年份还靠谱吗？",
     f"结论：会变差。排序能力在 2019–2020 年跌到最低，违约概率越估越高（2023 年高估约 {OVER:.0f} 倍）。"
     "原因有两个：训练数据里含 2008 危机，以及没用上近期数据，哪个影响大每年不一样。建议每年重训、删掉危机年份，并按客户变化、概率、排序三条规则监控。",
     "原文（数字和方法细节）：reports/stability.md；数字全部来自 outputs/stability.json。总览只做缩写和画图，冲突以原文为准。",
     [s0, s1, s2, s3, s4, s5, s6], [],
     "MULT-416 · 数据：Freddie Mac Single-Family Loan-Level Dataset（抽样版）")
