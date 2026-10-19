"""Shared argument handling for the nanomotion scripts."""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def args(needs_mat=True):
    ap = argparse.ArgumentParser()
    if needs_mat:
        ap.add_argument("mat", help="allDat.mat from doi:10.4121/384b7cf5-94da-4275-a6d6-a868eca8fec2")
    ap.add_argument("-o", "--out", default="build", help="output directory (default: build)")
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    return a
