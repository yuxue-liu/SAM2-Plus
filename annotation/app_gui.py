"""
SAM-2-Plus multi-class interactive video annotator.

Annotates ONE part-folder (<=200 frames) of the extracted dataset.
Each class is a SAM object (obj_id 1..N); background = 0. Masks are auto-saved
as single-channel index PNGs mirrored into masks/ (UniMatch-V2 convention).

Features
  * Point mode  : left-click = positive, right-click (or Ctrl+left) = negative
                  -> real-time SAM selection for the active class on this frame
  * Brush mode  : drag with left/right button to lay many positive/negative
                  points along the stroke, SAM runs on release
  * Multi-class : pick the active class (1..N); each gets its own SAM object
  * Propagate   : forward video tracking from the current frame
  * Pause/Resume: pause anytime, edit frames, resume -> re-propagates from the
                  current frame using the latest corrections
  * Rewind/seek : -10 / -1 / +1 / +10 and a slider to revisit & fix any frame
  * Auto-save   : every edited or propagated frame is written immediately

Run (inside your SAM-2 env, with the checkpoint present):
  cd demo
  python app_gui.py --image_dir D:/study/code/ab_dataset/images/a/part_000 \
                    --num-classes 3
"""
import os
# must be set BEFORE torch initializes CUDA -> reduces fragmentation OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import cv2
import torch
import argparse
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import json
import numpy as np
from PIL import Image, ImageTk
import warnings
import json
from pathlib import Path

from natsort import natsorted

# make the SAM2-Plus repo root importable regardless of the launch cwd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2_plus.build_sam import build_sam2_video_predictor_plus
from point_manager import PointManager
from app_core import AnnotationSession
from engine import overlay_index_mask, PALETTE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
try:
    from needle_keypoints import order_skeleton, fit_arc_2d, extend_to_mask
except Exception as _cl_e:                     # skimage/scipy missing -> centerline off
    order_skeleton = fit_arc_2d = extend_to_mask = None
    print(f"[warn] centerline disabled: {_cl_e}")

# ----------------------------------------------------------------- UI theme
_PANEL_BG = "#eef1f4"
_FONT = ("TkDefaultFont", 10, "bold")
_BTN = {  # kind -> (bg, fg, active-bg)
    "primary": ("#2e7d57", "#ffffff", "#24613f"),
    "action":  ("#3270b3", "#ffffff", "#255a91"),
    "danger":  ("#b3433b", "#ffffff", "#8f332c"),
    "neutral": ("#566573", "#ffffff", "#3f4a55"),
}


def mkbtn(parent, text, cmd, kind="action", **kw):
    bg, fg, ab = _BTN[kind]
    return tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                     activebackground=ab, activeforeground="#ffffff",
                     relief="flat", bd=0, padx=10, pady=4, font=_FONT,
                     cursor="hand2", highlightthickness=0, **kw)


def mklf(parent, text):
    return tk.LabelFrame(parent, text=text, font=("TkDefaultFont", 9, "bold"),
                         fg="#2c3e50", bg=_PANEL_BG, padx=5, pady=3)

warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'")
os.environ["TQDM_DISABLE"] = "1"

# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default=None, type=str,
                    help="a video folder (images/<key>) or single part-folder. "
                         "If omitted, built from --root/--dataset/--key")
parser.add_argument("--root", default="/root/autodl-tmp/data/surgical_seg",
                    help="dataset root, used with --dataset/--key when --image_dir is omitted")
parser.add_argument("--dataset", default=None, help="video name under --root (e.g. march_1)")
parser.add_argument("--key", default=None, help="left-eye frame key (e.g. 1_01); = images/<key>")
parser.add_argument("--num-classes", type=int, default=3)
parser.add_argument("--part-index", type=int, default=0,
                    help="which part folder to start on (0-based); use the Part ◀ ▶ buttons to switch")
parser.add_argument("--cfg", type=str,
                    default="configs/sam2.1/sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml",
                    help="hydra model config (package-relative)")
parser.add_argument("--ckpt", type=str, default=None,
                    help="checkpoint path; default: <SAM2-Plus>/checkpoints/checkpoint_phase123.pt")
parser.add_argument("--max-disp", type=int, default=1100,
                    help="max canvas width in px (height scales to keep aspect)")
# frames are kept on CPU by default (200 frames on GPU is what OOMs cuDNN);
# pass --gpu-video only on a large, dedicated GPU for a small speed gain.
parser.add_argument("--gpu-video", dest="offload_video", action="store_false",
                    help="keep video frames on GPU (default: offload to CPU)")
parser.add_argument("--offload-state", action="store_true",
                    help="also offload inference state to CPU (slower, least VRAM)")
parser.add_argument("--kp-subdir", default="keypoints_pred",
                    help="sidecar dir for needle keypoints (Keypt mode loads/saves here)")
parser.set_defaults(offload_video=True)

# ---------------- DISCOVER PART FOLDERS ----------------
# --image_dir may be a single part folder (has images) OR a video folder whose
# children are part_000/part_001/... . In the latter case we list all parts and
# let the GUI switch between them (the SAM2 predictor is reused, no model reload).
def _has_images(d):
    return any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(d))

def discover_parts(image_dir):
    if _has_images(image_dir):
        return [image_dir]
    subs = sorted(os.path.join(image_dir, d) for d in os.listdir(image_dir)
                  if os.path.isdir(os.path.join(image_dir, d)))
    parts = [s for s in subs if _has_images(s)]
    return parts or [image_dir]

def build_predictor(cfg, ckpt=None):
    """Build the SAM-2-Plus video predictor (reused across parts AND panes)."""
    ckpt = ckpt or os.path.join(os.path.dirname(__file__), "..", "checkpoints",
                                "checkpoint_phase123.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = build_sam2_video_predictor_plus(cfg, ckpt, device=device)
    # Treat a correction on ANY frame (incl. already-propagated ones) as a
    # conditioning frame, so edits after rewinding take full effect immediately
    # AND survive a later re-propagation.
    p.add_all_frames_to_correct_as_cond = True
    print("[OK] SAM-2-Plus loaded on", device)
    return p


# ====================================================================
# GUI
# ====================================================================
class AnnotatorGUI:
    def __init__(self, root, parts, predictor, num_classes=3, part_index=0,
                 max_disp=1100, offload_video=True, offload_state=False,
                 kp_subdir="keypoints_pred", embedded=False):
        self.root = root
        self.parts = parts
        self._predictor = predictor
        self.num_classes = num_classes
        self.max_disp = max_disp
        self.offload_video = offload_video
        self.offload_state = offload_state
        self.kp_subdir = kp_subdir
        self.embedded = embedded
        self.part_idx = min(max(part_index, 0), len(self.parts) - 1)
        self.cur = 0                       # current frame index
        self.active_cls = 1                # active class (obj_id)
        self.mode = "point"               # "point" | "brush"
        self.propagating = False
        self._load_part(self.part_idx)     # builds self.sess for the current part
        self._init_curation()             # excluded-frames + augmentation specs
        self.brush_spacing = 12            # px between sampled brush points (native)
        self.propagating = False
        self.gen = None
        self._brush_last = None            # last sampled brush point (native coords)
        self._stroke_active = False         # paint-mode stroke in progress

        # ---- display / zoom-pan state ----
        # Everything (mask + points) is rendered at NATIVE resolution, then a
        # viewport is cropped (zoom + offset) and resized to the fixed canvas.
        self.nW, self.nH = self.sess.W, self.sess.H
        self.base_scale = min(self.max_disp / self.nW, 1.0)   # fit-to-canvas at zoom=1
        self.dW = int(self.nW * self.base_scale)
        self.dH = int(self.nH * self.base_scale)
        self.zoom = 1.0                    # >=1.0; 1.0 = whole frame fits canvas
        self.min_zoom, self.max_zoom = 1.0, 8.0
        self.off_x, self.off_y = 0.0, 0.0  # native coord of top-left visible pixel
        self._pan_last = None              # last middle-drag pos (canvas coords)
        self.tk_img = None
        self.centerline = {}               # frame idx -> (mask-sum signature, polyline)
        self._undo = []                    # Ctrl+Z stack: (frame, (masks_copy, kps_copy))

        if not self.embedded:
            root.title("SAM-2-Plus Annotator")
        self._build_ui()
        self.redraw()

    # ------------------------------------------------------------ part switching
    def _load_part(self, idx):
        """(Re)build the SAM2 state + session for part `idx`, reusing the model."""
        idx = max(0, min(idx, len(self.parts) - 1))
        self.part_idx = idx
        part_dir = self.parts[idx]
        frames = [os.path.join(part_dir, f) for f in natsorted(os.listdir(part_dir))
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        # release previous SAM2 state before loading the next part
        if getattr(self, "sess", None) is not None:
            self.sess = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        state = self._predictor.init_state(part_dir,
                                           offload_video_to_cpu=self.offload_video,
                                           offload_state_to_cpu=self.offload_state)
        self.sess = AnnotationSession(self._predictor, state, frames, self.num_classes, PointManager())
        self.cur = 0
        self.propagating = False
        if not getattr(self, "embedded", False):
            self.root.title(f"SAM-2-Plus Annotator  -  part {idx + 1}/{len(self.parts)}: "
                            f"{os.path.basename(part_dir)}")
        if hasattr(self, "slider"):
            self.slider.config(to=max(0, self.sess.num_frames - 1))
            self.slider.set(0)
        if hasattr(self, "part_lbl"):
            self.part_lbl.config(text=f"part {idx + 1}/{len(self.parts)}")
        # keypoint-editing state is per-part (frames change) -> reset
        self.kp_sel = None                 # index of selected keypoint on cur frame
        self.kp_add = False                # next left-click adds a new keypoint
        self.kp_cache = {}                 # frame idx -> keypoints list (refs into kp_full)
        self.kp_full = {}                  # frame idx -> full sidecar dict
        if hasattr(self, "canvas"):
            self.redraw()

    def _switch_part(self, delta):
        if self.propagating:
            return
        self._load_part(self.part_idx + delta)

    def done_next_part(self):
        """Mark current part done and advance; notify when all parts are finished.
        (Masks auto-save on every edit/propagation, so no extra save is needed.)"""
        if self.propagating:
            return
        if self.part_idx < len(self.parts) - 1:
            self._load_part(self.part_idx + 1)
            messagebox.showinfo("Next part",
                                f"Now on part {self.part_idx + 1}/{len(self.parts)}: "
                                f"{os.path.basename(self.parts[self.part_idx])}")
        else:
            messagebox.showinfo(
                "All parts done",
                "All parts of this video are annotated.\n\n"
                "Close this window, then run:\n"
                "bash tools/build_one_dataset.sh finalize <name> <right_mp4> <key>")

    # ------------------------------------------------------------ UI layout
    def _build_ui(self):
        self.canvas = tk.Canvas(self.root, width=self.dW, height=self.dH, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=12, padx=6, pady=6)
        self.canvas.bind("<Button-1>", self.on_left_down)
        self.canvas.bind("<Button-3>", self.on_right_down)
        self.canvas.bind("<Control-Button-1>", self.on_right_down)
        self.canvas.bind("<B1-Motion>", lambda e: self.on_drag(e, 1))
        self.canvas.bind("<B3-Motion>", lambda e: self.on_drag(e, 0))
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonRelease-3>", self.on_release)
        # zoom: mouse wheel (Windows/mac) + Button-4/5 (Linux/X11, e.g. autodl)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.on_wheel(e, delta=1))
        self.canvas.bind("<Button-5>", lambda e: self.on_wheel(e, delta=-1))
        # pan: middle-button drag
        self.canvas.bind("<Button-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", lambda e: setattr(self, "_pan_last", None))
        # keypoint shortcuts — bound to THIS pane's canvas so two panes don't clash
        self.canvas.config(takefocus=1)
        self.canvas.bind("<Button-1>", lambda e: self.canvas.focus_set(), add="+")
        self.canvas.bind("a", lambda e: self.kp_add_toggle())
        self.canvas.bind("<Delete>", lambda e: self.kp_delete_sel())
        self.canvas.bind("<Control-z>", lambda e: self.undo())
        self.canvas.bind("<Control-Z>", lambda e: self.undo())

        self.root.configure(bg=_PANEL_BG)
        _names = {1: "needle", 2: "thread", 3: "clamps"}
        # ---- STEP 1: pick the object (class) ----
        cls_frame = mklf(self.root, "2 · Object")
        cls_frame.grid(row=1, column=4, columnspan=3, sticky="w", pady=2)
        self.cls_var = tk.IntVar(value=1)
        for c in range(1, self.num_classes + 1):
            r, g, b = PALETTE.get(c, (200, 200, 200))[::-1]  # BGR->RGB
            col = f"#{r:02x}{g:02x}{b:02x}"
            tk.Radiobutton(cls_frame, text=f"{c} {_names.get(c, '')}".strip(),
                           variable=self.cls_var, value=c, indicatoron=False,
                           width=9, font=_FONT, fg=col, bg="#ffffff",
                           selectcolor="#cfe0f0", relief="groove", bd=2,
                           command=self.on_class_change).pack(side="left", padx=2)

        # ---- STEP 2: choose WHAT you operate (Mask vs Points), then the tools ----
        op_frame = mklf(self.root, "1 · Operate")
        op_frame.grid(row=1, column=0, columnspan=4, sticky="w", padx=6, pady=2)
        self.target_var = tk.StringVar(value="mask")
        for _v, _l in (("mask", "Mask"), ("points", "Points")):
            tk.Radiobutton(op_frame, text=_l, variable=self.target_var, value=_v,
                           indicatoron=False, width=7, font=_FONT, bg="#ffffff",
                           selectcolor="#ffd9a6", relief="groove", bd=2,
                           command=self.on_target_change).pack(side="left", padx=1)
        tk.Label(op_frame, text="tool:", bg=_PANEL_BG, font=_FONT).pack(side="left", padx=(8, 0))
        self.mode_var = tk.StringVar(value="point")
        self._mask_tools = []
        for _val, _lab in (("point", "Point"), ("brush", "Brush"), ("paint", "Paint")):
            rb = tk.Radiobutton(op_frame, text=_lab, variable=self.mode_var, value=_val,
                                indicatoron=False, width=5, font=_FONT, bg="#ffffff",
                                selectcolor="#f0d2cf", relief="groove", bd=2,
                                command=self.on_mode_change)
            rb.pack(side="left", padx=1)
            self._mask_tools.append(rb)
        self.brush_scale = tk.Scale(op_frame, from_=4, to=40, orient="horizontal",
                                    length=70, showvalue=True, bg=_PANEL_BG,
                                    command=self.on_brush_size)
        self.brush_scale.set(self.brush_spacing)
        self.brush_scale.pack(side="left")

        # ---- frame navigation ----
        nav = mklf(self.root, "Frame")
        nav.grid(row=1, column=7, columnspan=5, sticky="we", pady=2)
        mkbtn(nav, "⏮10", lambda: self.seek(self.cur - 10)).pack(side="left", padx=1)
        mkbtn(nav, "◀1", lambda: self.seek(self.cur - 1)).pack(side="left", padx=1)
        mkbtn(nav, "1▶", lambda: self.seek(self.cur + 1)).pack(side="left", padx=1)
        mkbtn(nav, "10⏭", lambda: self.seek(self.cur + 10)).pack(side="left", padx=1)
        self.slider = tk.Scale(nav, from_=0, to=self.sess.num_frames - 1,
                               orient="horizontal", length=220, bg=_PANEL_BG,
                               command=self.on_slider, showvalue=False)
        self.slider.pack(side="left", padx=4)

        # ---- STEP 3: act on the current selection ----
        act = tk.Frame(self.root, bg=_PANEL_BG)
        act.grid(row=2, column=0, columnspan=12, sticky="we", pady=4)

        mask_f = mklf(act, "Mask"); mask_f.pack(side="left", padx=6); self.mask_f = mask_f
        mkbtn(mask_f, "Clear class", self.clear_class, "danger").pack(side="left", padx=2)
        mkbtn(mask_f, "Clear all", self.clear_frame, "danger").pack(side="left", padx=2)
        mkbtn(mask_f, "Use saved", self.use_saved_mask, "neutral").pack(side="left", padx=2)
        self.prop_btn = mkbtn(mask_f, "Propagate ▶", self.toggle_propagate, "primary", width=12)
        self.prop_btn.pack(side="left", padx=4)
        mkbtn(mask_f, "Reset part", self.reset_all, "danger").pack(side="left", padx=2)

        kp_frame = mklf(act, "Keypoints"); kp_frame.pack(side="left", padx=6); self.kp_frame = kp_frame
        mkbtn(kp_frame, "Add (a)", self.kp_add_toggle, "action").pack(side="left", padx=2)
        mkbtn(kp_frame, "Del (Del)", self.kp_delete_sel, "danger").pack(side="left", padx=2)

        part_frame = mklf(act, "Part"); part_frame.pack(side="left", padx=6)
        mkbtn(part_frame, "◀", lambda: self._switch_part(-1)).pack(side="left", padx=1)
        self.part_lbl = tk.Label(part_frame, text=f"{self.part_idx + 1}/{len(self.parts)}",
                                 width=6, bg="#ffffff", relief="groove", font=_FONT)
        self.part_lbl.pack(side="left", padx=2)
        mkbtn(part_frame, "▶", lambda: self._switch_part(1)).pack(side="left", padx=1)
        mkbtn(part_frame, "Done ✓", self.done_next_part, "primary").pack(side="left", padx=4)

        cur_frame = mklf(act, "Curate"); cur_frame.pack(side="left", padx=6)
        mkbtn(cur_frame, "Del range", self.delete_range, "danger").pack(side="left", padx=2)
        mkbtn(cur_frame, "Restore", self.restore_range, "neutral").pack(side="left", padx=2)
        mkbtn(cur_frame, "Augment", self.augment_settings, "neutral").pack(side="left", padx=2)

        zoom_frame = mklf(act, "Zoom"); zoom_frame.pack(side="left", padx=6)
        mkbtn(zoom_frame, "−", lambda: self.zoom_at(1 / 1.25, self.dW / 2, self.dH / 2), "neutral", width=2).pack(side="left", padx=1)
        mkbtn(zoom_frame, "+", lambda: self.zoom_at(1.25, self.dW / 2, self.dH / 2), "neutral", width=2).pack(side="left", padx=1)
        mkbtn(zoom_frame, "Fit", self.zoom_fit, "neutral").pack(side="left", padx=2)
        self.zoom_lbl = tk.Label(zoom_frame, text="100%", width=5, bg="#ffffff",
                                 relief="groove", font=_FONT)
        self.zoom_lbl.pack(side="left", padx=2)

        self.status = tk.Label(self.root, text="", anchor="w", fg="#1a3c5a",
                               bg="#dbe5ee", font=_FONT, padx=8, pady=3)
        self.status.grid(row=3, column=0, columnspan=12, sticky="we", padx=6, pady=(2, 4))
        self.on_target_change()        # apply initial Mask/Points gating

    # ----------------------------------------------------- coord mapping
    def _eff(self):
        """Effective px-per-native-px at the current zoom."""
        return self.base_scale * self.zoom

    def _clamp_offsets(self):
        eff = self._eff()
        view_w, view_h = self.dW / eff, self.dH / eff
        self.off_x = min(max(self.off_x, 0.0), max(0.0, self.nW - view_w))
        self.off_y = min(max(self.off_y, 0.0), max(0.0, self.nH - view_h))

    def to_native(self, ex, ey):
        """Canvas pixel -> native image pixel (accounts for zoom + pan)."""
        eff = self._eff()
        nx = self.off_x + min(max(ex, 0), self.dW - 1) / eff
        ny = self.off_y + min(max(ey, 0), self.dH - 1) / eff
        return int(min(max(nx, 0), self.nW - 1)), int(min(max(ny, 0), self.nH - 1))

    # ----------------------------------------------------- zoom / pan
    def zoom_at(self, factor, cx, cy):
        """Zoom by `factor`, keeping the native point under (cx,cy) fixed."""
        eff_old = self._eff()
        nx = self.off_x + cx / eff_old
        ny = self.off_y + cy / eff_old
        self.zoom = min(max(self.zoom * factor, self.min_zoom), self.max_zoom)
        eff_new = self._eff()
        self.off_x = nx - cx / eff_new
        self.off_y = ny - cy / eff_new
        self._clamp_offsets()
        self.redraw()

    def zoom_fit(self):
        self.zoom = 1.0
        self.off_x = self.off_y = 0.0
        self.redraw()

    def on_wheel(self, e, delta=None):
        if delta is None:
            delta = 1 if getattr(e, "delta", 0) > 0 else -1
        self.zoom_at(1.25 if delta > 0 else 1 / 1.25, e.x, e.y)

    def on_pan_start(self, e):
        self._pan_last = (e.x, e.y)

    def on_pan_move(self, e):
        if self._pan_last is None:
            return
        eff = self._eff()
        self.off_x -= (e.x - self._pan_last[0]) / eff
        self.off_y -= (e.y - self._pan_last[1]) / eff
        self._pan_last = (e.x, e.y)
        self._clamp_offsets()
        self.redraw()

    def on_brush_size(self, v):
        self.brush_spacing = max(1, int(float(v)))

    # ----------------------------------------------------- drawing
    def _centerline_of(self, m):
        """Extended (reaches tip/tail) needle centerline polyline of a mask, or None."""
        if fit_arc_2d is None or m is None or not m.any():
            return None
        try:
            poly = fit_arc_2d(m)
            if poly is None:
                poly = order_skeleton(m)
        except Exception:
            poly = None
        if poly is not None and len(poly) >= 2 and extend_to_mask is not None:
            poly = extend_to_mask(np.asarray(poly, float), m)
        return poly

    def _needle_centerline(self, idx):
        """Needle (class 1) centerline for display; cached, recomputed on mask change."""
        m = self.sess.class_masks.get(idx, {}).get(1)
        if m is None or not m.any():
            self.centerline.pop(idx, None)
            return None
        sig = int(m.sum())
        cached = self.centerline.get(idx)
        if cached and cached[0] == sig:
            return cached[1]
        poly = self._centerline_of(m)
        self.centerline[idx] = (sig, poly)
        return poly

    def redraw(self):
        img = cv2.imread(self.sess.frames[self.cur])          # native nH x nW
        idx = self.sess.get_mask(self.cur)
        img = overlay_index_mask(img, idx)   # low-saturation, semi-transparent
        poly = self._needle_centerline(self.cur)              # needle centerline (magenta)
        if poly is not None and len(poly) >= 2:
            cv2.polylines(img, [np.asarray(poly, np.int32).reshape(-1, 1, 2)],
                          False, (255, 0, 255), 2, cv2.LINE_AA)
        # draw active-class points on this frame (at native coords)
        pts, labels = self.sess.pm.get(self.cur, self.active_cls)
        for (x, y), lb in zip(pts, labels):
            color = (0, 255, 0) if lb == 1 else (0, 0, 255)
            cv2.circle(img, (int(x), int(y)), 5, color, -1)
            cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 1)
        # draw needle keypoints (Keypt mode loads/edits these); selected = cyan
        for j, kp in enumerate(self.load_kps(self.cur)):
            if kp.get("x") is None:
                continue
            kx, ky = int(kp["x"]), int(kp["y"])
            sel = (self.mode == "keypt" and j == self.kp_sel)
            col = (0, 255, 255) if sel else (0, 165, 255)
            cv2.circle(img, (kx, ky), 6, col, -1)
            cv2.circle(img, (kx, ky), 6, (255, 255, 255), 1)
            cv2.putText(img, str(kp.get("name", j)), (kx + 7, ky - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        # crop the zoom/pan viewport, then scale to the fixed canvas
        eff = self._eff()
        self._clamp_offsets()
        x0, y0 = int(round(self.off_x)), int(round(self.off_y))
        x1 = min(self.nW, int(round(x0 + self.dW / eff)))
        y1 = min(self.nH, int(round(y0 + self.dH / eff)))
        crop = img[y0:y1, x0:x1]
        disp = cv2.resize(crop, (self.dW, self.dH), interpolation=cv2.INTER_LINEAR)
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(disp))
        self.canvas.delete("frame_img")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img,
                                 tags="frame_img")
        self._set_status()
        if hasattr(self, "zoom_lbl"):
            self.zoom_lbl.config(text=f"{int(self.zoom * 100)}%")
        self.canvas.update_idletasks()   # force immediate repaint within handlers

    def _set_status(self):
        objs = self.sess.pm.objs_on_frame(self.cur)
        saved = "saved" if os.path.exists(self.sess.mask_path(self.cur)) else "-"
        excl = "  [EXCLUDED]" if self.sess.frames[self.cur] in self.excluded else ""
        self.status.config(
            fg="red" if excl else "navy",
            text=f"frame {self.cur+1}/{self.sess.num_frames}  |  "
                 f"class={self.active_cls}  mode={self.mode}  |  "
                 f"prompts on frame: {objs}  |  mask:{saved}  |  "
                 f"excluded={len(self.excluded)} augspecs={len(self.augment_specs)}  |  "
                 f"{'PROPAGATING...' if self.propagating else 'idle'}{excl}")

    # ---------------------------------------- curation: delete-range / augment
    def _init_curation(self):
        p = os.path.abspath(self.sess.frames[0]).replace("\\", "/")
        self.droot = p.split("/images/")[0] if "/images/" in p else os.path.dirname(p)
        self.excluded = set()
        self.augment_specs = []
        exc, aug = self._curation_paths()
        if os.path.exists(exc):
            for line in open(exc, encoding="utf-8"):
                line = line.strip()
                if line:
                    self.excluded.add(line)
        if os.path.exists(aug):
            try:
                self.augment_specs = json.load(open(aug, encoding="utf-8"))
            except Exception:
                self.augment_specs = []

    def _curation_paths(self):
        return (os.path.join(self.droot, "excluded_frames.txt"),
                os.path.join(self.droot, "augment_config.json"))

    def _save_curation(self):
        exc, aug = self._curation_paths()
        with open(exc, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(self.excluded)) + ("\n" if self.excluded else ""))
        with open(aug, "w", encoding="utf-8") as f:
            json.dump(self.augment_specs, f, indent=2)

    def _ask_range(self, title, extra_hint=""):
        s = simpledialog.askstring(
            title, f"frames: start end {extra_hint}\n"
                   f"(1..{self.sess.num_frames}; current = {self.cur + 1})", parent=self.root)
        if not s:
            return None
        parts = s.replace(",", " ").split()
        if len(parts) < 2:
            messagebox.showwarning(title, "need at least: start end")
            return None
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            messagebox.showwarning(title, "start/end must be integers")
            return None
        n = self.sess.num_frames
        a = max(1, min(a, n)); b = max(1, min(b, n))
        if a > b:
            a, b = b, a
        return a - 1, b - 1, parts[2:]

    def delete_range(self):
        r = self._ask_range("Delete frames")
        if not r:
            return
        a, b, _ = r
        for i in range(a, b + 1):
            self.excluded.add(self.sess.frames[i])
        self._save_curation(); self.redraw()
        messagebox.showinfo("Delete", f"excluded {b - a + 1} frames "
                            f"({a + 1}..{b + 1}); total excluded = {len(self.excluded)}")

    def restore_range(self):
        r = self._ask_range("Restore frames")
        if not r:
            return
        a, b, _ = r
        for i in range(a, b + 1):
            self.excluded.discard(self.sess.frames[i])
        self._save_curation(); self.redraw()

    def augment_settings(self):
        r = self._ask_range("Augment frames", extra_hint="[rotate_deg translate_frac]")
        if not r:
            return
        a, b, extra = r
        angle = float(extra[0]) if len(extra) > 0 else 10.0
        shift = float(extra[1]) if len(extra) > 1 else 0.05
        self.augment_specs.append({
            "frames": [self.sess.frames[i] for i in range(a, b + 1)],
            "rotate_deg": angle, "translate_frac": shift, "hflip": True})
        self._save_curation(); self.redraw()
        messagebox.showinfo("Augment", f"spec for frames {a + 1}..{b + 1}: "
                            f"rotate +/-{angle}, translate +/-{shift}; "
                            f"total specs = {len(self.augment_specs)}")

    # ----------------------------------------------------- events: selectors
    def on_class_change(self):
        self.active_cls = self.cls_var.get()
        self.redraw()

    def on_mode_change(self):
        self.mode = self.mode_var.get()
        self.redraw()

    def on_target_change(self):
        """Two-step gating: pick Mask or Points, then only that group is active."""
        mask_on = (self.target_var.get() == "mask")
        for rb in getattr(self, "_mask_tools", []):
            rb.config(state=("normal" if mask_on else "disabled"))
        if hasattr(self, "mask_f"):
            for w in self.mask_f.winfo_children():
                w.config(state=("normal" if mask_on else "disabled"))
        if hasattr(self, "kp_frame"):
            for w in self.kp_frame.winfo_children():
                w.config(state=("disabled" if mask_on else "normal"))
        if mask_on:
            if self.mode_var.get() == "keypt":
                self.mode_var.set("point")
        else:
            self.mode_var.set("keypt")
        self.on_mode_change()

    def on_slider(self, v):
        if not self.propagating:
            self.seek(int(v))

    def seek(self, idx):
        idx = max(0, min(self.sess.num_frames - 1, idx))
        self.cur = idx
        self.kp_sel = None
        self.slider.set(idx)
        self.redraw()

    # ----------------------------------------------------- events: mouse
    def _add_and_apply(self, ex, ey, label):
        x, y = self.to_native(ex, ey)
        self.sess.pm.add_point(self.cur, self.active_cls, x, y, label)
        self.sess.apply_points(self.cur, self.active_cls)
        self.redraw()

    def _down(self, e, label):
        if self.propagating:
            return
        self._push_undo()
        if self.mode == "keypt":
            self._kp_down(e, label)
            return
        if self.mode == "point":
            self._add_and_apply(e.x, e.y, label)
        elif self.mode == "brush":
            self._brush_last = self.to_native(e.x, e.y)
            self.sess.pm.add_point(self.cur, self.active_cls, *self._brush_last, label)
            self.redraw()
        else:  # paint: directly write pixels, bypassing SAM
            self._stroke_active = True
            x, y = self.to_native(e.x, e.y)
            self.sess.paint(self.cur, self.active_cls, x, y,
                            self.brush_spacing, erase=(label == 0))
            self.redraw()

    def on_left_down(self, e):
        self._down(e, 1)

    def on_right_down(self, e):
        self._down(e, 0)

    def on_drag(self, e, label):
        if self.propagating:
            return
        if self.mode == "keypt":
            if self.kp_sel is not None:
                kps = self.load_kps(self.cur)
                if 0 <= self.kp_sel < len(kps):
                    x, y = self.to_native(e.x, e.y)
                    kps[self.kp_sel]["x"] = float(x)
                    kps[self.kp_sel]["y"] = float(y)
                    self.redraw()
            return
        if self.mode == "brush" and self._brush_last is not None:
            x, y = self.to_native(e.x, e.y)
            lx, ly = self._brush_last
            if (x - lx) ** 2 + (y - ly) ** 2 >= self.brush_spacing ** 2:
                self.sess.pm.add_point(self.cur, self.active_cls, x, y, label)
                self._brush_last = (x, y)
                self.redraw()        # live feedback (no SAM until release)
        elif self.mode == "paint" and self._stroke_active:
            x, y = self.to_native(e.x, e.y)
            self.sess.paint(self.cur, self.active_cls, x, y,
                            self.brush_spacing, erase=(label == 0))
            self.redraw()

    def on_release(self, e):
        if self.propagating:
            return
        if self.mode == "keypt":
            if self.kp_sel is not None:
                self.save_kps(self.cur)
            return
        if self.mode == "brush" and self._brush_last is not None:
            self._brush_last = None
            self.sess.apply_points(self.cur, self.active_cls)   # run SAM once
            self.redraw()
        elif self.mode == "paint" and self._stroke_active:
            self._stroke_active = False
            self.sess.commit_paint(self.cur, self.active_cls)   # save + anchor SAM
            self.redraw()

    # ----------------------------------------------------- keypoints (Keypt mode)
    def kp_sidecar(self, idx):
        p = self.sess.frames[idx].replace("\\", "/")
        p = p.replace("/images/", f"/{self.kp_subdir}/")
        return Path(p).with_suffix(".json")

    def load_kps(self, idx):
        """Return (cached) list of needle keypoints for frame `idx`, loaded from
        the sidecar. Edits to the returned list are persisted by save_kps()."""
        if idx in self.kp_cache:
            return self.kp_cache[idx]
        path = self.kp_sidecar(idx)
        full = None
        if path.exists():
            try:
                full = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                full = None
        if full is None:
            rel = self.sess.frames[idx].replace("\\", "/")
            i = rel.find("/images/")
            rel = rel[i + 1:] if i >= 0 else os.path.basename(rel)
            full = {"image": rel, "needle": {"keypoints": []}, "instruments": []}
        nd = full.get("needle")
        if nd is None:
            nd = {"keypoints": []}
            full["needle"] = nd
        kps = nd.setdefault("keypoints", [])
        self.kp_full[idx] = full
        self.kp_cache[idx] = kps
        return kps

    def save_kps(self, idx):
        full = self.kp_full.get(idx)
        if full is None:
            return
        if full.get("needle") is not None:
            full["needle"]["keypoints"] = self.kp_cache.get(idx, [])
        path = self.kp_sidecar(idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(full, indent=2), encoding="utf-8")

    def _kp_pick(self, x, y, kps, thr=12):
        best, bd = None, float(thr) ** 2
        for j, kp in enumerate(kps):
            if kp.get("x") is None:
                continue
            d = (kp["x"] - x) ** 2 + (kp["y"] - y) ** 2
            if d <= bd:
                bd, best = d, j
        return best

    def _kp_down(self, e, label):
        x, y = self.to_native(e.x, e.y)
        kps = self.load_kps(self.cur)
        if label == 0:                          # right-click = delete nearest
            j = self._kp_pick(x, y, kps)
            if j is not None:
                kps.pop(j)
                self.kp_sel = None
                self.save_kps(self.cur)
                self.redraw()
            return
        if self.kp_add:                         # add a new keypoint
            kps.append({"id": len(kps), "name": f"kp{len(kps)}",
                        "x": float(x), "y": float(y), "visible": 1})
            self.kp_sel = len(kps) - 1
            self.kp_add = False
            self.save_kps(self.cur)
            self.redraw()
            return
        self.kp_sel = self._kp_pick(x, y, kps)  # select for move (drag)
        self.redraw()

    def kp_add_toggle(self):
        self.kp_add = not self.kp_add
        if self.kp_add and self.mode != "keypt":
            self.mode_var.set("keypt")
            self.mode = "keypt"
        self.redraw()

    def kp_delete_sel(self):
        if getattr(self, "sess", None) is None:
            return
        kps = self.load_kps(self.cur)
        if self.kp_sel is not None and 0 <= self.kp_sel < len(kps):
            self._push_undo()
            kps.pop(self.kp_sel)
            self.kp_sel = None
            self.save_kps(self.cur)
            self.redraw()
        else:
            self.status.config(text="Keypt: no keypoint selected — left-click a dot first, or 'Add KP (a)'.")

    # ----------------------------------------------------- undo (Ctrl+Z)
    def _push_undo(self):
        """Snapshot the current frame's masks + keypoints before a mutating edit."""
        import copy
        if getattr(self, "sess", None) is None:
            return
        cm = self.sess.class_masks.get(self.cur, {})
        snap = ({c: m.copy() for c, m in cm.items()},
                copy.deepcopy(self.load_kps(self.cur)))
        self._undo.append((self.cur, snap))
        if len(self._undo) > 30:
            self._undo.pop(0)

    def undo(self):
        if not getattr(self, "_undo", None):
            self.status.config(text="Nothing to undo.")
            return "break"
        idx, (masks_copy, kps_copy) = self._undo.pop()
        self.sess.class_masks[idx] = {c: m.copy() for c, m in masks_copy.items()}
        try:
            self.sess._save(idx)
        except Exception:
            pass
        kps = self.load_kps(idx)
        kps[:] = kps_copy
        self.save_kps(idx)
        self.centerline.pop(idx, None)
        self.kp_sel = None
        if idx != self.cur:
            self.seek(idx)
        else:
            self.redraw()
        self.status.config(text=f"Undo (Ctrl+Z) — {len(self._undo)} step(s) left.")
        return "break"

    # ----------------------------------------------------- events: actions
    def clear_class(self):
        if self.propagating:
            return
        self._push_undo()
        self.sess.clear_frame_obj(self.cur, self.active_cls)
        self.redraw()

    def clear_frame(self):
        if self.propagating:
            return
        self._push_undo()
        self.sess.clear_frame(self.cur)
        self.redraw()

    def reset_all(self):
        if self.propagating:
            return
        self.sess.predictor.reset_state(self.sess.state)
        self.sess.pm = PointManager()
        self.sess.class_masks.clear()
        self.redraw()

    # ----------------------------------------------------- propagation
    def toggle_propagate(self):
        if self.propagating:
            self.pause()
        else:
            self.start()

    def start(self):
        # Anchor on the current frame's saved/loaded annotation so we can
        # propagate from previously-saved labels (e.g. a part reopened with only
        # preloaded prediction PNGs, which are NOT yet SAM inputs).
        self.sess.seed_from_mask(self.cur)
        # Drop any SAM object that has no input on any frame; a dangling object
        # (e.g. a class whose only prompt got cleared on an edit) otherwise makes
        # propagate's preflight crash with "No input points or masks for object id N".
        self.sess.prune_dangling_objects()
        if not self.sess.has_inputs():
            self.status.config(
                text="Nothing to propagate: brush/paint on this frame (or open one "
                     "with a saved mask) so a class is seeded, then press Propagate.")
            return
        # (re)build generator from current frame so latest edits are honored
        self.gen = self.sess.make_propagator(self.cur)
        self.propagating = True
        self.prop_btn.config(text="Pause ⏸")
        self._tick()

    def use_saved_mask(self):
        """Seed SAM with the current frame's saved/loaded mask as an anchor."""
        if self.propagating:
            return
        n = self.sess.seed_from_mask(self.cur)
        self.status.config(
            text=(f"Seeded {n} class(es) from saved mask on frame {self.cur+1} "
                  f"- now press Propagate ▶" if n else
                  f"No saved mask on frame {self.cur+1} to seed from."))
        self.redraw()

    def pause(self):
        self.propagating = False
        self.gen = None                 # resume rebuilds from current frame
        self.prop_btn.config(text="Propagate ▶")
        self._set_status()

    def _tick(self):
        if not self.propagating or self.gen is None:
            return
        try:
            fidx, _ = next(self.gen)
        except StopIteration:
            self.propagating = False
            self.gen = None
            self.prop_btn.config(text="Propagate ▶")
            self.redraw()
            return
        self.cur = fidx
        self.slider.set(fidx)
        self.redraw()
        self.root.after(1, self._tick)


def main():
    args = parser.parse_args()
    if not args.image_dir:
        assert args.dataset and args.key, \
            "give --image_dir, or both --dataset and --key (joined with --root)"
        args.image_dir = os.path.join(args.root, args.dataset, "images", args.key)
    parts = discover_parts(args.image_dir)
    assert parts, f"No images / part folders under {args.image_dir}"
    print(f"[parts] {len(parts)} part(s): {[os.path.basename(p) for p in parts]}")
    predictor = build_predictor(args.cfg, args.ckpt)
    root = tk.Tk()
    AnnotatorGUI(root, parts, predictor, num_classes=args.num_classes,
                 part_index=args.part_index, max_disp=args.max_disp,
                 offload_video=args.offload_video, offload_state=args.offload_state,
                 kp_subdir=args.kp_subdir)
    root.mainloop()


if __name__ == "__main__":
    main()
