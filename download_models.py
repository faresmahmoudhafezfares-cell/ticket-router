"""Download the trained model files (they are too big to keep in the Git repository).

Usage:  python download_models.py            (deployed + small models, needed for the app)
        python download_models.py --teacher  (also the big 12-layer RoBERTa teacher)
"""
import argparse
import re
import subprocess
import shutil
import urllib.request
import zipfile
from pathlib import Path

# The model files are attached to the "v1.0" GitHub Release of this repository.
# The repository address is read from `git remote` (or set REPO below by hand).
REPO = "OWNER/REPO"


def repo_slug():
    try:
        url = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
        m = re.search(r"github\.com[:/]([^/]+/[^/.]+)", url)
        if m:
            return m.group(1)
    except (OSError, subprocess.CalledProcessError):
        pass
    return REPO


def fetch(url, name):
    print(f"downloading {name} ...")
    tmp = Path(name)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(".")
    tmp.unlink()
    print(f"{name} unpacked into models/")


ap = argparse.ArgumentParser()
ap.add_argument("--teacher", action="store_true")
args = ap.parse_args()
base = f"https://github.com/{repo_slug()}/releases/download/v1.0"
fetch(f"{base}/models.zip", "models.zip")
if args.teacher:
    fetch(f"{base}/teacher.zip", "teacher.zip")
