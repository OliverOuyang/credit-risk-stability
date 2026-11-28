# HTML 总览模板：样式（style.css，原样取自 11-23 总结 artifact）+ 画图函数 + 「每节图 + 3–4 问 + 原则标签」骨架。用法见同目录 README.md。
import html
import os

E = html.escape
CSS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css"), encoding="utf-8").read()
MK = '<defs><marker id="kk" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1L9 5L1 9z" fill="var(--muted)"/></marker></defs>'

# ---------- 画图：坐标按 760 宽的画布写；颜色只用 class ----------
# 方框 class：pp=人做的事  "st mc"=平台/Multica  "st lf"=已通/存证  loc=本机  wash=底色说明  "st bad"=退回/卡住
#            card=普通卡片  "card todo"=虚线、还没做


def svg(w, h, label, body):
    """一张图。label 是给读屏和没看图的人的一句话总结，必须写全。"""
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{E(label)}" class="dg">{MK}{body}</svg>'


def box(x, y, w, h, cls, t, s=None, s2=None):
    """方框：标题 t，最多两行小字 s、s2，自动垂直居中。"""
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/>'
    cx = x + w / 2
    lines = [l for l in (s, s2) if l]
    if not lines:
        return out + f'<text x="{cx}" y="{y + h / 2 + 4.5}" class="c t">{E(t)}</text>'
    top = y + h / 2 - 8.5 * len(lines) + 4.5
    out += f'<text x="{cx}" y="{top}" class="c t">{E(t)}</text>'
    for i, l in enumerate(lines):
        out += f'<text x="{cx}" y="{top + 17 * (i + 1)}" class="c s">{E(l)}</text>'
    return out


def num(cx, cy, n):
    """圆形编号。"""
    return (f'<circle cx="{cx}" cy="{cy}" r="12" fill="var(--accent)"/>'
            f'<text x="{cx}" y="{cy + 4}" class="c" style="fill:#fff;font-weight:700">{n}</text>')


def arrow(x1, y1, x2, y2, extra=""):
    """直线箭头；extra="dash" 是虚线。折线箭头直接写 <path d="M.. H.. V.." class="ar" marker-end="url(#kk)"/>。"""
    return f'<path d="M{x1} {y1}L{x2} {y2}" class="ar {extra}" fill="none" marker-end="url(#kk)"/>'


def lanes(cols, y=6, h=28):
    """泳道表头：cols = [(标题, x, 宽, class), ...]。"""
    return "".join(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" class="{c}"/><text x="{x + w / 2}" y="{y + 19}" class="c b">{E(t)}</text>'
                   for t, x, w, c in cols)


def fig(s, hint=False):
    """把 svg 包成图；宽图 hint=True，手机上提示左右滑。"""
    h = '<div class="scroll-hint">← 左右滑动看全图</div>' if hint else ""
    return f'<figure class="fg fx" style="--mw:760px">{h}<div class="sv">{s}</div></figure>'


# ---------- 骨架：每节 = 图 + 3–4 问 + 原则标签 ----------
def QA(*items):
    """问答：每项 (问, 答)，一节 3–4 个。答里可以用 <code>。"""
    return '<div class="qa">' + "".join(f"<div><b>{q}</b>{a}</div>" for q, a in items) + "</div>"


def TAG(t):
    return f'<p class="tag">{t}</p>'


def sec(id_, head, figs, qa, tag=None, h="h3"):
    """一节：先图后问答，最后一行「对应原则 N：……」。大章节用 h="h2"。"""
    return (f'<section class="wrap blk" id="{id_}">\n<{h}>{head}</{h}>\n' + "".join(figs) + "\n" + qa + "\n" + (TAG(tag) if tag else "") + "\n</section>\n")


def part(title):
    """分部大标题（不带图和问答）。"""
    return f'<h2 class="wrap part2">{title}</h2>\n'


def page(out, title, brand, nav, h1, lede, note, body, appendix, footer):
    """整页。nav = [(锚点 id, 文字), ...]；appendix = [(标题, [条目 html, ...])] 放在页尾折叠区。"""
    links = "".join(f'<a href="#{i}">{t}</a>' for i, t in nav)
    app = "".join(f'\n<details class="wrap prin"><summary>{t}</summary>\n<ol>\n' + "".join(f"<li>{x}</li>\n" for x in items) + "</ol></details>\n"
                  for t, items in appendix)
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<nav class="top" aria-label="页内导航"><div class="wrap"><span class="brand">{brand}</span>{links}</div></nav>

<header class="wrap hero2">
<h1>{h1}</h1>
<p class="lede">{lede}</p>
<p class="note">{note}</p>
</header>

{"".join(body)}{app}
<footer class="wrap">{footer}</footer>
</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "w", encoding="utf-8").write(doc)
    print(out)
