"""Put the downloaded data bundle into data/ with the layout DATA.md expects.

    python -m possession.unpack <zip or folder> [<zip or folder> ...]

Google Drive splits large folders into several zips and may or may not keep the tactical/ and
broadcast/ subfolders, so every file is placed by its name: BAR-ATH_tactical_* -> data/tactical/,
BAR-ATH_broadcast_* -> data/broadcast/. Prints file names and sizes only, never content.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")


def target_for(name):
    base = os.path.basename(name)
    if not base.startswith("BAR-ATH_"):
        return None
    # Drive names big single-file downloads with a part number: BAR-ATH_tactical-002.mp4
    stem, ext = os.path.splitext(base)
    base = re.sub(r"-\d{3}$", "", stem) + ext
    sub = "broadcast" if "_broadcast" in base else "tactical"
    return os.path.join(DATA, sub, base)


def place_zip(path):
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            target = target_for(info.filename)
            if target is None:
                print("  skip", info.filename)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            print(f"  {os.path.relpath(target, ROOT)}  {info.file_size / 1e6:.1f} MB")


def place_file(path):
    target = target_for(path)
    if target is None:
        print("  skip", path)
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.move(path, target)
    print(f"  {os.path.relpath(target, ROOT)}  {os.path.getsize(target) / 1e6:.1f} MB")


def main(args):
    for arg in args:
        if os.path.isdir(arg):
            for dirpath, _, files in os.walk(arg):
                for f in files:
                    p = os.path.join(dirpath, f)
                    place_zip(p) if f.endswith(".zip") else place_file(p)
        elif arg.endswith(".zip"):
            print(arg)
            place_zip(arg)
        else:
            place_file(arg)
    expected = ["tactical/BAR-ATH_tactical_atd.xml", "tactical/BAR-ATH_tactical_atd.txt",
                "tactical/BAR-ATH_tactical_homography.csv",
                "tactical/BAR-ATH_tactical_smart_tagging.xml", "tactical/BAR-ATH_tactical.mp4",
                "broadcast/BAR-ATH_broadcast_atd.xml", "broadcast/BAR-ATH_broadcast_atd.txt",
                "broadcast/BAR-ATH_broadcast_homography.csv", "broadcast/BAR-ATH_broadcast.mp4"]
    print("\ncheck:")
    for rel in expected:
        ok = os.path.exists(os.path.join(DATA, rel))
        print(f"  {'ok     ' if ok else 'MISSING'} {rel}")


if __name__ == "__main__":
    main(sys.argv[1:])
