"""Dual-eye annotator: TWO fully-editable AnnotatorGUI panes (left | right) in ONE
window. Each pane edits masks/points and propagates independently (identical UI).
The Sync button applies the 3D rigid-arc constraint and maps the needle from the
ACTIVE pane to the INACTIVE one (keypoints become consistent in both; a synthetic
needle mask is reprojected into the inactive eye).

This is a CORRECTION tool, not an inference tool: it assumes the segmentation
model was ALREADY run (predicted masks + keypoint sidecars exist) and lets you
fix those predictions. The GUI never runs the model — its only "automatic" step
is Sync, which is pure stereo geometry (process_frame), not a network.
Prerequisite: run `UniMatch-V2_local/infer_keypoints_seg.py --save-masks` and
`tools/prep_right_for_annotation.py` once before launching.

Usage:
  python demo/app_gui_dual.py \
    --left-image-dir  /root/autodl-tmp/data/surgical_seg/march_1/images/1_01/part_000 \
    --right-image-dir /root/autodl-tmp/data/surgical_seg/march_1/stereo_right/1_01 \
    --num-classes 3 --calib ../tools/needle_calib.json
"""
import argparse
import os
import sys
import tkinter as tk
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app_gui import AnnotatorGUI, discover_parts, build_predictor
from engine import overlay_index_mask  # noqa: F401  (kept import parity)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
try:
    from needle_keypoints import process_frame, load_calib
except Exception as _e:                       # skimage/scipy/calib missing -> sync off
    process_frame = load_calib = None
    print(f"[warn] stereo sync disabled: {_e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left-image-dir", default=None,
                    help="left-eye images/<key>; if omitted built from --root/--dataset/--key")
    ap.add_argument("--right-image-dir", default=None,
                    help="right-eye stereo_right/<key>; if omitted built from --root/--dataset/--key")
    ap.add_argument("--root", default="/root/autodl-tmp/data/surgical_seg",
                    help="dataset root, used with --dataset/--key when dirs are omitted")
    ap.add_argument("--dataset", default=None, help="video name under --root (e.g. march_1)")
    ap.add_argument("--key", default=None, help="left-eye frame key (e.g. 1_01)")
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--kp-subdir", default="keypoints_pred")
    ap.add_argument("--calib", default="../tools/needle_calib.json")
    ap.add_argument("--needle-class", type=int, default=1)
    ap.add_argument("--thread-class", type=int, default=2)
    ap.add_argument("--num-keypoints", type=int, default=5)
    ap.add_argument("--needle-px", type=int, default=12)
    a = ap.parse_args()
    if not (a.left_image_dir and a.right_image_dir):
        assert a.dataset and a.key, \
            "give --left-image-dir and --right-image-dir, or both --dataset and --key"
        a.left_image_dir = a.left_image_dir or os.path.join(a.root, a.dataset, "images", a.key)
        a.right_image_dir = a.right_image_dir or os.path.join(a.root, a.dataset, "stereo_right", a.key)

    predictor = build_predictor(a.cfg, a.ckpt)           # one model, two states
    lparts = discover_parts(a.left_image_dir)
    rparts = discover_parts(a.right_image_dir)
    calib = load_calib(a.calib) if (load_calib and os.path.exists(a.calib)) else None

    root = tk.Tk()
    root.title("SAM-2-Plus Dual Annotator  (LEFT | RIGHT)")
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{sw}x{sh}+0+0")
    half = max(500, sw // 2 - 30)

    bar = tk.Frame(root, bg="#dbe5ee")
    bar.grid(row=0, column=0, columnspan=2, sticky="we")
    panes = tk.Frame(root)
    panes.grid(row=1, column=0, columnspan=2, sticky="nw")
    lf = tk.Frame(panes); lf.grid(row=0, column=0, sticky="nw", padx=4)
    rf = tk.Frame(panes); rf.grid(row=0, column=1, sticky="nw", padx=4)

    appL = AnnotatorGUI(lf, lparts, predictor, num_classes=a.num_classes,
                        max_disp=half, kp_subdir=a.kp_subdir, embedded=True)
    appR = AnnotatorGUI(rf, rparts, predictor, num_classes=a.num_classes,
                        max_disp=half, kp_subdir=a.kp_subdir, embedded=True)

    # ---- linked navigation: both eyes always show the SAME frame index ----
    _busy = {"on": False}
    appL._seek0, appR._seek0 = appL.seek, appR.seek

    def _linked(src, dst, idx):
        # sync by STEM (filename) so left/right correspond even with different counts
        if _busy["on"]:
            return
        _busy["on"] = True
        try:
            src._seek0(idx)
            stem = Path(src.sess.frames[src.cur]).stem
            dmap = {Path(f).stem: k for k, f in enumerate(dst.sess.frames)}
            if stem in dmap:
                dst._seek0(dmap[stem])
        finally:
            _busy["on"] = False
    appL.seek = lambda idx: _linked(appL, appR, idx)
    appR.seek = lambda idx: _linked(appR, appL, idx)

    st = {"active": appL, "inactive": appR}

    def set_active(which):
        if which is not st["active"]:
            st["active"], st["inactive"] = which, st["active"]
        lbl.config(text=f"active = {'LEFT' if st['active'] is appL else 'RIGHT'} "
                        f"→ sync maps to {'RIGHT' if st['inactive'] is appR else 'LEFT'}")
    appL.canvas.bind("<Button-1>", lambda e: set_active(appL), add="+")
    appR.canvas.bind("<Button-1>", lambda e: set_active(appR), add="+")

    names = ["tip"] + [f"k{i}" for i in range(1, a.num_keypoints - 1)] + ["tail"]

    def _write_pane_kps(pane, idx, pts2d, out):
        kps = pane.load_kps(idx)
        kps[:] = []
        for i in range(a.num_keypoints):
            xyz = out["xyz_mm"][i]
            kps.append({"id": i, "name": names[i],
                        "x": float(pts2d[i][0]), "y": float(pts2d[i][1]),
                        "xyz_mm": ([float(v) for v in xyz] if xyz is not None else None),
                        "visible": int(out["visible"][i])})
        pane.save_kps(idx)

    def _reproject_mask(pane, idx, pts2d):
        H, W = pane.sess.H, pane.sess.W
        band = np.zeros((H, W), np.uint8)
        poly = np.asarray([p for p in pts2d if p is not None], np.float32)
        if len(poly) >= 2:
            dense = []
            for i in range(len(poly) - 1):
                for t in np.linspace(0, 1, 20):
                    dense.append(poly[i] * (1 - t) + poly[i + 1] * t)
            cv2.polylines(band, [np.asarray(dense, np.int32).reshape(-1, 1, 2)],
                          False, 1, a.needle_px)
        cm = pane.sess.class_masks.setdefault(idx, {})
        cm[a.needle_class] = band.astype(bool)
        try:
            pane.sess._save(idx)
        except Exception:
            pass

    def do_sync():
        if process_frame is None or calib is None:
            lbl.config(text="Sync: unavailable (need calib + skimage/scipy)."); return
        iL, iR = appL.cur, appR.cur
        nL = appL.sess.class_masks.get(iL, {}).get(a.needle_class)
        nR = appR.sess.class_masks.get(iR, {}).get(a.needle_class)
        tL = appL.sess.class_masks.get(iL, {}).get(a.thread_class)
        if nL is None or not nL.any() or nR is None or not nR.any():
            lbl.config(text="Sync: both eyes need a needle (class 1) mask on the current frame.")
            return
        try:
            out, status = process_frame(nL, nR, tL if tL is not None else np.zeros_like(nL),
                                        calib, a.num_keypoints)
        except Exception as e:
            lbl.config(text=f"Sync failed: {e}"); return
        if out is None:
            lbl.config(text=f"Sync: no stereo fit ({status})."); return
        # consistent keypoints into BOTH panes (each its own view's points)
        _write_pane_kps(appL, iL, out["left"], out)
        _write_pane_kps(appR, iR, out["right"], out)
        # synthetic needle mask reprojected into the INACTIVE pane only
        inactive = st["inactive"]
        if inactive is appR:
            _reproject_mask(appR, iR, out["right"])
        else:
            _reproject_mask(appL, iL, out["left"])
        appL.redraw(); appR.redraw()
        lbl.config(text="Sync: mapped active → inactive (keypoints both; mask to inactive).")

    tk.Button(bar, text="Sync  active ▶ inactive", bg="#2e7d57", fg="white",
              relief="flat", padx=14, pady=5, font=("TkDefaultFont", 11, "bold"),
              command=do_sync).pack(side="left", padx=8, pady=4)
    lbl = tk.Label(bar, text="active = LEFT → sync maps to RIGHT", bg="#dbe5ee",
                   fg="#1a3c5a", font=("TkDefaultFont", 10, "bold"))
    lbl.pack(side="left", padx=8)

    root.mainloop()


if __name__ == "__main__":
    main()
