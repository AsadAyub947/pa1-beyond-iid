"""
Download PACS (original JPEG files) and arrange it as <out>/<domain>/<class>/*.jpg.

    python shared/download_pacs.py --out /content/pacs_data

Source: the public GitHub copy used for Task 3
(https://github.com/MachineLearning2020/Homework3-PACS, folder PACS/). Files are
moved, never re-encoded, so Task 2 and Task 3 see byte-identical images. The
script verifies the standard PACS counts (9,991 images) before returning.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

REPO = "https://github.com/MachineLearning2020/Homework3-PACS.git"
DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
EXPECTED = {  # per-domain, per-class image counts of the standard PACS release
    "photo": [189, 202, 182, 186, 199, 280, 432],
    "art_painting": [379, 255, 285, 184, 201, 295, 449],
    "cartoon": [389, 457, 346, 135, 324, 288, 405],
    "sketch": [772, 740, 753, 608, 816, 80, 160],
}
IMG_EXT = (".jpg", ".jpeg", ".png")


def count(out):
    res = {}
    for d in DOMAINS:
        res[d] = [len([f for f in os.listdir(os.path.join(out, d, c)) if f.lower().endswith(IMG_EXT)])
                  if os.path.isdir(os.path.join(out, d, c)) else 0 for c in CLASSES]
    return res


def verify(out) -> bool:
    got = count(out)
    ok = True
    for d in DOMAINS:
        flag = "OK " if got[d] == EXPECTED[d] else "BAD"
        ok &= got[d] == EXPECTED[d]
        print(f"  [{flag}] {d:13s} total={sum(got[d]):5d} per class={got[d]}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/pacs_data")
    args = ap.parse_args()
    out = args.out
    if all(os.path.isdir(os.path.join(out, d)) for d in DOMAINS) and verify(out):
        print(f"PACS already present and verified at {out}")
        return
    os.makedirs(out, exist_ok=True)
    tmp = tempfile.mkdtemp()
    print(f"Cloning {REPO} ...")
    subprocess.run(["git", "clone", "--depth", "1", "-q", REPO, tmp], check=True)
    src_root = None
    for root, dirs, _ in os.walk(tmp):
        if all(d in dirs for d in DOMAINS):
            src_root = root
            break
    if src_root is None:
        sys.exit("Could not find the PACS domain folders in the cloned repository.")
    for d in DOMAINS:
        dst = os.path.join(out, d)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(os.path.join(src_root, d), dst)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"PACS moved to {out}")
    if not verify(out):
        sys.exit("PACS image counts do not match the standard release -- check the download.")
    print("All 4 domains x 7 classes verified (9,991 images).")


if __name__ == "__main__":
    main()
