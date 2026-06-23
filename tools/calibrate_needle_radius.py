"""Estimate the canonical needle arc radius (mm) from existing keypoint sidecars.

Step (a) of the model-based pose plan (docs/NEEDLE_POSE_REGISTRATION.md) needs ONE
number: the true radius of the rigid needle arc. Rather than read it off a spec
sheet, we recover it from the frames where the current free-radius fit was most
trustworthy: well-seen, high-confidence, tip/tail-resolved needles. The robust
median of those per-frame radii is the model radius; we write it to a small
needle_model.json that the pose pipeline loads via --needle-model.

The sidecars are produced by needle_keypoints.py and live at
    <root>/<dataset>/keypoints/<key>/.../<stem>.json
with needle.model.radius_mm and needle.conf per frame.

Example:
    python tools/calibrate_needle_radius.py \
        --root /root/autodl-tmp/data/surgical_seg \
        --datasets march_1 march_2 \
        --min-conf 0.5 --out tools/needle_model.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def iter_sidecars(root, datasets, subdir):
    root = Path(root)
    ds_list = datasets or [p.name for p in root.iterdir()
                           if (p / subdir).is_dir()]
    for d in ds_list:
        for p in (root / d / subdir).rglob("*.json"):
            yield d, p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="dataset subdirs to scan (default: all with a keypoints/ dir)")
    ap.add_argument("--subdir", default="keypoints")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="only use frames with needle.conf >= this")
    ap.add_argument("--require-tip-tail", action="store_true",
                    help="only use frames where tip/tail orientation was resolved")
    ap.add_argument("--rmin", type=float, default=1.0, help="discard radii below this (mm)")
    ap.add_argument("--rmax", type=float, default=50.0, help="discard radii above this (mm)")
    ap.add_argument("--out", default="needle_model.json")
    args = ap.parse_args()

    radii, used, total = [], 0, 0
    for _d, p in iter_sidecars(args.root, args.datasets, args.subdir):
        total += 1
        try:
            nd = json.loads(p.read_text(encoding="utf-8")).get("needle")
        except Exception:
            continue
        if not nd:
            continue
        r = (nd.get("model") or {}).get("radius_mm")
        conf = float(nd.get("conf", 0.0))
        if r is None or conf < args.min_conf:
            continue
        if args.require_tip_tail and not nd.get("tip_tail_known", False):
            continue
        if not (args.rmin <= float(r) <= args.rmax):
            continue
        radii.append(float(r)); used += 1

    if not radii:
        raise SystemExit(f"[calib] no usable radii (scanned {total} sidecars). "
                         "Lower --min-conf or check --root/--subdir/--datasets.")

    r = np.asarray(radii)
    # robust median + IQR-trimmed mean for reporting
    q1, med, q3 = np.percentile(r, [25, 50, 75])
    iqr = q3 - q1
    keep = r[(r >= q1 - 1.5 * iqr) & (r <= q3 + 1.5 * iqr)]
    model = {
        "radius_mm": float(med),
        "radius_mean_trimmed_mm": float(keep.mean()),
        "radius_std_mm": float(keep.std()),
        "n_frames_used": int(used),
        "n_sidecars_scanned": int(total),
        "min_conf": args.min_conf,
        "units": "mm",
        "note": "Canonical needle arc radius for fixed-radius 3D registration "
                "(see docs/NEEDLE_POSE_REGISTRATION.md). Loaded via --needle-model.",
    }
    Path(args.out).write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"[calib] radius_mm (median) = {med:.3f}  "
          f"trimmed-mean = {keep.mean():.3f} +/- {keep.std():.3f}  "
          f"(used {used}/{total} sidecars)")
    print(f"[calib] wrote -> {args.out}")


if __name__ == "__main__":
    main()
