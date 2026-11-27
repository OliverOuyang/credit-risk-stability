"""Render reports/*.tmpl.md -> reports/*.md. Placeholders: {{metrics.a.b:.3f}} reads outputs/metrics.json, {{stability.a.b}} outputs/stability.json,
{{config.file.a.b}} reads config/file.yaml; the part after ':' is a Python format spec."""
import json
import re

from crs.target import ROOT, load_cfg

PH = re.compile(r"\{\{\s*(metrics|stability|config)\.([^}:]+?)(?::([^}]*))?\s*\}\}")


def lookup(src, path, metrics):
    keys = path.split(".")
    x = metrics[src] if src in metrics else load_cfg(keys.pop(0))
    for k in keys:
        x = x[int(k)] if isinstance(x, list) else x[k]  # KeyError = template refers to a missing number
    return x


def fmt(x, spec):
    if isinstance(x, list):  # e.g. years a rule fired in
        return "、".join(format(i, spec) for i in x) or "无"
    return format(x, spec)


def render(text, metrics):
    return PH.sub(lambda m: fmt(lookup(m[1], m[2], metrics), m[3] or ""), text)


if __name__ == "__main__":
    metrics = {k: json.loads((ROOT / "outputs" / f"{k}.json").read_text()) for k in ("metrics", "stability")}
    for t in sorted((ROOT / "reports").glob("*.tmpl.md")):
        t.with_name(t.name.replace(".tmpl", "")).write_text(render(t.read_text(), metrics))
