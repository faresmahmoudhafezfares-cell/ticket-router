"""Download the trained model files. They are not stored in the Git repository (too big);
they are attached to the GitHub Release "v1.1" of this repository.

Usage:
  python download_models.py         the four models of the ladder, needed by the app (~540 MB):
                                    tf-idf + LR, word2vec + FFN, RoBERTa-base FP32, distilled RoBERTa INT8
  python download_models.py --all   also the two extra rows of the results table:
                                    4a distilled RoBERTa FP32 (downloaded, ~245 MB)
                                    3b RoBERTa-base INT8 (built locally by quantizing model 3)
"""
import argparse
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = "faresmahmoudhafezfares-cell/ticket-router"
TAG = "v1.1"


def repo_slug():
    """Use the address of the cloned repository if there is one, else REPO above."""
    try:
        url = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        m = re.search(r"github\.com[:/]([^/]+/[^/.]+)", url)
        if m:
            return m.group(1)
    except (OSError, subprocess.CalledProcessError):
        pass
    return REPO


def fetch(url, name):
    print(f"downloading {name} ... (this can take a few minutes)")
    tmp = Path(name)
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Could not download {url} ({e.code}). Is {name} attached to release {TAG}?")
    with zipfile.ZipFile(tmp) as z:
        z.extractall(".")
    tmp.unlink()
    print(f"{name} unpacked into models/")


ap = argparse.ArgumentParser()
ap.add_argument("--all", action="store_true", help="also get the extra models 3b and 4a")
args = ap.parse_args()
base = f"https://github.com/{repo_slug()}/releases/download/{TAG}"
fetch(f"{base}/models.zip", "models.zip")
if args.all:
    fetch(f"{base}/extras.zip", "extras.zip")
    if not Path("models/roberta_teacher_int8").exists():
        print("building model 3b (RoBERTa-base INT8) by quantizing model 3 ...")
        from router.models import quantize_and_save
        quantize_and_save("models/roberta_teacher", "models/roberta_teacher_int8")
print("done. Start the app with:  python app.py")
