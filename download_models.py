"""Download the trained model files. They are not stored in the Git repository (too big);
they are attached to the GitHub Release "v1.1" of this repository.

Usage:
  python download_models.py         the four models of the ladder, needed by the app (~530 MB):
                                    tf-idf + LR, word2vec + FFN, RoBERTa-base FP32, distilled RoBERTa INT8
  python download_models.py --all   also the two extra rows of the results table:
                                    4a distilled RoBERTa FP32 (downloaded, ~240 MB)
                                    3b RoBERTa-base INT8 (built locally by quantizing model 3)
"""
import argparse
import os
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


def fetch(base, names, zip_name, have=None):
    """Download one zip (given as one file, or as several parts that are joined) and unpack it.

    `have` is a path that already exists when this zip has been unpacked before; it is skipped."""
    if have and Path(have).exists():
        print(f"{zip_name}: already in models/, skipping")
        return
    tmp = Path(zip_name)
    with open(tmp, "wb") as out:
        for name in names:
            print(f"downloading {name} ... (this can take a few minutes)", flush=True)
            try:
                with urllib.request.urlopen(f"{base}/{name}") as r:
                    shutil.copyfileobj(r, out)
            except urllib.error.HTTPError as e:
                out.close()
                tmp.unlink(missing_ok=True)
                raise SystemExit(f"Could not download {base}/{name} ({e.code}). "
                                 f"Is it attached to release {TAG}?")
    with zipfile.ZipFile(tmp) as z:
        z.extractall(".")
    tmp.unlink()
    print(f"{zip_name} unpacked into models/")


ap = argparse.ArgumentParser()
ap.add_argument("--all", action="store_true", help="also get the extra models 3b and 4a")
args = ap.parse_args()
# TICKET_ROUTER_MODELS_URL can point to a mirror of the release files.
base = os.environ.get("TICKET_ROUTER_MODELS_URL") or f"https://github.com/{repo_slug()}/releases/download/{TAG}"

fetch(base, ["ticket-router-models-1.zip"], "models-1.zip",                      # models 1, 2 and 4
      have="models/roberta_student_int8")
fetch(base, ["ticket-router-models-2.zip.part0", "ticket-router-models-2.zip.part1"],
      "models-2.zip", have="models/roberta_teacher")                            # model 3, in two parts
if args.all:
    fetch(base, ["ticket-router-extras.zip"], "extras.zip",                      # model 4a
          have="models/roberta_student")
    if not Path("models/roberta_teacher_int8").exists():
        print("building model 3b (RoBERTa-base INT8) by quantizing model 3 ...")
        from router.models import quantize_and_save
        quantize_and_save("models/roberta_teacher", "models/roberta_teacher_int8")
print("done. Start the app with:  python app.py")
