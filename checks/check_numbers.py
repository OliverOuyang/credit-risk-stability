"""Number provenance: every number in finished reports must come from metrics.json, config/ or the
constants allowlist. Reports should use placeholders ({{metrics.x}}, {{config.x}}) rendered from those."""
import json
import re
from pathlib import Path

import yaml

from checks.common import ROOT, main

SCAN = ["reports/**/*.md", "docs/site/**/*.md", "docs/site/**/*.html", "app/**/*.md"]
NUM = re.compile(r"(?<![A-Za-z0-9_.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%?)")
YEAR_AFTER = re.compile(r"\s*[年Q\-–—/~]")
YEAR_BEFORE = re.compile(r"[\-–—/~至到]\s*$")
HTML = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.S | re.I)
STRIP = [re.compile(p) for p in (r"\{\{.*?\}\}", r"`[^`]*`", r"\]\([^)]*\)", r"https?://\S+")]


def source_values(root):
    vals = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            vals.add(abs(float(x)))
        elif isinstance(x, str):
            vals.update(float(m) for m in re.findall(r"\d+(?:\.\d+)?", x))
        elif isinstance(x, dict):
            [walk(i) for i in x.values()]
        elif isinstance(x, list):
            [walk(i) for i in x]

    m = Path(root) / "outputs" / "metrics.json"
    if m.exists():
        walk(json.loads(m.read_text()))
    for p in sorted((Path(root) / "config").glob("*.yaml")):
        walk(yaml.safe_load(p.read_text()))
    return vals


def matches(value, decimals, percent, vals):
    cands = [(value, decimals)] + ([(value / 100, decimals + 2)] if percent else [])
    return any(abs(round(s, k) - c) < 1e-9 for c, k in cands for s in vals)


def scan_text(text, rel, vals):
    v, fence = [], False
    for no, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        line = re.sub(r"^\s*#+\s*[\d.]*\s*", "", line)      # heading numbers
        line = re.sub(r"^\s*\d+[.)、]\s+", "", line)           # list numbers
        for p in STRIP:
            line = p.sub(" ", line)
        for m in NUM.finditer(line):
            whole, frac, pct = m.group(1).replace(",", ""), m.group(2) or "", m.group(3)
            if not frac and len(whole) == 4 and 1900 <= int(whole) <= 2099 and (
                    YEAR_AFTER.match(line, m.end()) or YEAR_BEFORE.search(line[:m.start()])):
                continue
            if not matches(float(whole + frac), max(len(frac) - 1, 0), bool(pct), vals):
                v.append(f"{rel}:{no}: {m.group(0)} has no source (metrics.json / config / constants_allowlist)")
    return v


def run(root=ROOT):
    root = Path(root)
    vals = source_values(root)
    files = sorted({p for g in SCAN for p in root.glob(g) if p.is_file()})
    v = []
    for p in files:
        text = p.read_text()
        if p.suffix == ".html":  # drop tags/scripts, keep line numbers
            text = HTML.sub(lambda m: "\n" * m.group(0).count("\n"), text)
        v += scan_text(text, p.relative_to(root), vals)
    return v


if __name__ == "__main__":
    main("numbers", run())
