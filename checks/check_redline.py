import subprocess
from pathlib import Path

from checks.common import ROOT, main


def run(root, files, read=None):
    return []  # red step: not implemented yet


def repo_files(root):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True).stdout.decode()
    return [f for f in out.split("\0") if f and (Path(root) / f).is_file()]


if __name__ == "__main__":
    main("redline", run(ROOT, []))
