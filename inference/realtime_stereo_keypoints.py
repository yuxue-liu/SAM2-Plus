"""Real-time stereo needle keypoint + 6-DoF pose from a streaming/capture-card source.

Per frame:
  left/right image -> segmentation model (fp16, low VRAM) -> needle/thread masks
  -> needle_keypoints.process_frame -> N equally-spaced 3D keypoints (xyz_mm)
  -> 6-DoF pose of the rigid needle (R, t, rvec)  [from the fitted arc]
Shows the LEFT|RIGHT overlay with live FPS, prints/returns the 3D points + pose.

Sources:
  --left A --right B          two streams (video files OR camera indices, e.g. 0 1)
  --capture 0 --layout sbs    one device carrying side-by-side stereo (split L|R)
                  layout tb    top-bottom stereo

Example (two video files):
  python realtime_stereo_keypoints.py --config configs/surgical_combined.yaml \
    --checkpoint exp/combined_r100_base/best.pth \
    --calib /root/autodl-tmp/code/SAM2-Plus/tools/needle_calib.json \
    --left left.mp4 --right right.mp4 --num-keypoints 5 --seg-size 640 --show

Example (capture card, side-by-side):
  python realtime_stereo_keypoints.py --config ... --checkpoint ... --calib ... \
    --capture 0 --layout sbs --num-keypoints 5 --show
"""
import argparse
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
import torchvision.transforms as T

from test import build_inference_model, infer_pred
from util.utils import intersectionAndUnion
from util.classes import CLASSES

_NORM = T.Compose([T.ToTensor(),
                   T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_nk(path):
    sys.path.insert(0, path)
    import needle_keypoints as nk
    return nk


# ----------------------------------------------------------------- source
class StereoSource:
    def __init__(self, args):
        self.mode = None
        if args.dataset:                       # replay a stored stereo sequence as a stream
            import json
            self.ds = args.root.resolve() / args.dataset
            self.key = args.key
            meta = json.loads((self.ds / 'meta.json').read_text(encoding='utf-8'))
            self.recs = sorted(meta['videos'][args.key], key=lambda r: r['ordinal'])
            if args.limit:
                self.recs = self.recs[:args.limit]
            self.i = 0
            self.mode = 'dataset'
        elif args.capture is not None:
            self.cap = cv2.VideoCapture(int(args.capture))
            self.layout = args.layout
            self.mode = 'split'
        else:
            def _open(s):
                return cv2.VideoCapture(int(s) if str(s).isdigit() else s)
            self.capL = _open(args.left)
            self.capR = _open(args.right)
            self.mode = 'pair'

    def read(self):
        if self.mode == 'dataset':
            if self.i >= len(self.recs):
                return None, None, None
            r = self.recs[self.i]; self.i += 1
            stem = os.path.splitext(os.path.basename(r['image']))[0]
            L = cv2.imread(str(self.ds / r['image']))
            R = cv2.imread(str(self.ds / 'stereo_right' / self.key / f'{stem}.jpg'))
            if L is None or R is None:
                return self.read()             # skip missing, advance
            return L, R, stem
        if self.mode == 'split':
            ok, fr = self.cap.read()
            if not ok:
                return None, None, None
            h, w = fr.shape[:2]
            if self.layout == 'sbs':
                return fr[:, :w // 2], fr[:, w // 2:], None
            return fr[:h // 2], fr[h // 2:], None
        okL, L = self.capL.read(); okR, R = self.capR.read()
        if not okL or not okR:
            return None, None, None
        return L, R, None

    def release(self):
        for c in ('cap', 'capL', 'capR'):
            if hasattr(self, c):
                getattr(self, c).release()


# ----------------------------------------------------------------- seg
def seg_mask(bundle, bgr, seg_size, device, use_amp):
    H, W = bgr.shape[:2]
    if seg_size and max(H, W) > seg_size:
        s = seg_size / float(max(H, W))
        small = cv2.resize(bgr, (int(round(W * s)), int(round(H * s))))
    else:
        small = bgr
    chw = _NORM(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()
    with torch.inference_mode(), ctx:
        pred = infer_pred(bundle, chw, device)            # HxW uint8 at `small` size
    pred = np.asarray(pred)
    if pred.shape[:2] != (H, W):
        pred = cv2.resize(pred.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return pred


def seg_masks_batch(bundle, bgrs, seg_size, device, use_amp):
    """Segment BOTH eyes in ONE forward (2x speedup). Falls back to per-image
    infer_pred when the model has affinity/edge branches or the two frames differ
    in size."""
    if bundle['affinity_side'] is not None or bundle['use_edge_enhance'] \
            or len({b.shape[:2] for b in bgrs}) != 1:
        return [seg_mask(bundle, b, seg_size, device, use_amp) for b in bgrs]
    model = bundle['model']; patch = bundle['patch_size']
    H, W = bgrs[0].shape[:2]
    if seg_size and max(H, W) > seg_size:
        s = seg_size / float(max(H, W)); sh, sw = int(round(H * s)), int(round(W * s))
    else:
        sh, sw = H, W
    new_h = max(patch, int(round(sh / patch)) * patch)
    new_w = max(patch, int(round(sw / patch)) * patch)
    ts = [_NORM(Image.fromarray(cv2.cvtColor(cv2.resize(b, (sw, sh)), cv2.COLOR_BGR2RGB)))
          for b in bgrs]
    x = torch.stack(ts).to(device)
    if (new_h, new_w) != (sh, sw):
        x = F.interpolate(x, (new_h, new_w), mode='bilinear', align_corners=True)
    ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()
    with torch.inference_mode(), ctx:
        logits = model(x)                                   # (B, nclass, mh, mw)
        lab = logits.argmax(1)                              # argmax at MODEL res (cheap)
        if lab.shape[-2:] != (H, W):                        # nearest-upscale LABELS only
            lab = F.interpolate(lab[:, None].float(), (H, W), mode='nearest')[:, 0]
        preds = lab.to(torch.uint8).cpu().numpy()
    return [preds[i] for i in range(len(bgrs))]


class PoseKalman:
    """Constant-velocity Kalman on the 6-vector [tx,ty,tz, rx,ry,rz] to de-jitter
    the 6-DoF pose. Coasts (predict-only) on frames with no detection."""
    def __init__(self, meas_noise=1.0, proc_noise=1e-2):
        kf = cv2.KalmanFilter(12, 6)
        Ft = np.eye(12, dtype=np.float32)
        for i in range(6):
            Ft[i, i + 6] = 1.0
        kf.transitionMatrix = Ft
        Hm = np.zeros((6, 12), np.float32); Hm[:6, :6] = np.eye(6)
        kf.measurementMatrix = Hm
        kf.processNoiseCov = np.eye(12, dtype=np.float32) * proc_noise
        kf.measurementNoiseCov = np.eye(6, dtype=np.float32) * meas_noise
        self.kf = kf; self.inited = False; self.prev_rvec = None

    def _align(self, rvec):                  # keep rvec continuous (avoid sign flips)
        rvec = np.asarray(rvec, float)
        if self.prev_rvec is not None and np.dot(rvec, self.prev_rvec) < 0:
            rvec = -rvec
        self.prev_rvec = rvec
        return rvec

    def update(self, t, rvec):
        rvec = self._align(rvec)
        m = np.asarray(list(t) + list(rvec), np.float32).reshape(6, 1)
        if not self.inited:
            self.kf.statePost = np.vstack([m, np.zeros((6, 1), np.float32)])
            self.inited = True
            return np.asarray(t, float), np.asarray(rvec, float)
        self.kf.predict()
        est = self.kf.correct(m)[:6, 0]
        return est[:3], est[3:6]

    def coast(self):
        if not self.inited:
            return None
        est = self.kf.predict()[:6, 0]
        return est[:3], est[3:6]


def screen_size(default=(1920, 1080)):
    """Best-effort desktop resolution (for fitting the window). Falls back gracefully."""
    try:
        import tkinter as tk
        r = tk.Tk(); r.withdraw()
        wh = (r.winfo_screenwidth(), r.winfo_screenheight())
        r.destroy()
        return wh
    except Exception:
        return default


def fit_window(win, canvas_w, canvas_h, frac=0.9):
    """Resize a WINDOW_NORMAL window to fit `frac` of the desktop, keeping aspect."""
    sw, sh = screen_size()
    s = min(sw * frac / canvas_w, sh * frac / canvas_h, 1.0)
    cv2.resizeWindow(win, max(1, int(canvas_w * s)), max(1, int(canvas_h * s)))


def hstack_same_h(a, b):
    if a.shape[0] != b.shape[0]:
        s = a.shape[0] / b.shape[0]
        b = cv2.resize(b, (int(b.shape[1] * s), a.shape[0]))
    return cv2.hconcat([a, b])


def rot_to_euler_deg(R):
    R = np.asarray(R, float)
    sy = (R[0, 0] ** 2 + R[1, 0] ** 2) ** 0.5
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2]); y = np.arctan2(-R[2, 0], sy); z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1]); y = np.arctan2(-R[2, 0], sy); z = 0.0
    return np.degrees([x, y, z])


def draw_pose_axes(img, R, t, K, D, rvec_cam, tvec_cam, axis_mm=20.0):
    """Project the object frame axes (X red, Y green, Z blue) into a view — for
    visual pose verification (back-projection)."""
    R = np.asarray(R, float); t = np.asarray(t, float)
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
        return                                  # degenerate/NaN pose -> skip drawing
    pts = np.stack([t, t + axis_mm * R[:, 0], t + axis_mm * R[:, 1], t + axis_mm * R[:, 2]])
    pr, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.asarray(rvec_cam, float),
                              np.asarray(tvec_cam, float), np.asarray(K, float), np.asarray(D, float))
    pr = pr.reshape(-1, 2)
    if not np.all(np.isfinite(pr)):
        return
    o, ax, ay, az = [tuple(np.int32(np.round(q))) for q in pr]
    cv2.line(img, o, ax, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(img, o, ay, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(img, o, az, (255, 0, 0), 2, cv2.LINE_AA)


def draw_reproj(img, xyz, K, D, rvec_cam, tvec_cam):
    """Re-project the 3D keypoints back into a view (white rings) — they should
    sit on the detected keypoints if the triangulation/pose is consistent."""
    pts = np.asarray([q for q in xyz if q is not None and np.all(np.isfinite(q))], float)
    if len(pts) == 0:
        return
    pr, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.asarray(rvec_cam, float),
                              np.asarray(tvec_cam, float), np.asarray(K, float), np.asarray(D, float))
    for (x, y) in pr.reshape(-1, 2):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue                            # NaN/inf projection -> skip (avoids cv2 crash)
        cv2.circle(img, (int(round(x)), int(round(y))), 9, (255, 255, 255), 1, cv2.LINE_AA)


def project_xyz(xyz, K, D, rvec_cam, tvec_cam):
    pts = np.asarray([q for q in xyz if q is not None], float)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)
    pr, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.asarray(rvec_cam, float),
                              np.asarray(tvec_cam, float), np.asarray(K, float), np.asarray(D, float))
    return pr.reshape(-1, 2)


def reprojection_error(out, calib, rvecR, tvecR):
    if out is None:
        return None
    xyz = out.get('xyz_mm') or []
    valid = [i for i, q in enumerate(xyz) if q is not None]
    if not valid:
        return None
    pL = project_xyz([xyz[i] for i in valid], calib['K1'], calib['D1'], np.zeros(3), np.zeros(3))
    pR = project_xyz([xyz[i] for i in valid], calib['K2'], calib['D2'], rvecR, tvecR)
    qL = np.asarray([out['left'][i] for i in valid], float)
    qR = np.asarray([out['right'][i] for i in valid], float)
    eL = np.linalg.norm(pL - qL, axis=1) if len(pL) else np.array([])
    eR = np.linalg.norm(pR - qR, axis=1) if len(pR) else np.array([])
    both = np.concatenate([eL, eR]) if len(eL) or len(eR) else np.array([])
    if len(both) == 0:
        return None
    return {
        'mean': float(both.mean()),
        'max': float(both.max()),
        'left_mean': float(eL.mean()) if len(eL) else float('nan'),
        'right_mean': float(eR.mean()) if len(eR) else float('nan'),
    }


def overlay_segmentation(bgr, mask, needle_class=1, thread_class=2, alpha=0.38):
    color = bgr.copy()
    color[mask == needle_class] = (0, 0, 255)
    color[mask == thread_class] = (0, 220, 0)
    color[mask == 3] = (255, 120, 0)
    return cv2.addWeighted(bgr, 1.0 - alpha, color, alpha, 0)


def put_panel_line(panel, text, y, color=(235, 235, 235), scale=0.48):
    cv2.putText(panel, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)
    return y + int(22 * max(scale / 0.48, 1.0))


def build_info_panel(out, fps, reproj, kp_names, width=680, height=720):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    y = 28
    y = put_panel_line(panel, 'Stereo keypoints + 6-DoF pose', y, (0, 255, 255), 0.62)
    y = put_panel_line(panel, f'FPS: {fps:.1f}', y + 4, (0, 255, 0), 0.58)
    if reproj is None:
        y = put_panel_line(panel, 'Reprojection error: N/A', y, (180, 180, 180), 0.5)
    else:
        y = put_panel_line(panel, f"Reprojection mean/max: {reproj['mean']:.2f} / {reproj['max']:.2f} px", y,
                           (255, 255, 255), 0.5)
        y = put_panel_line(panel, f"L/R mean: {reproj['left_mean']:.2f} / {reproj['right_mean']:.2f} px", y,
                           (210, 210, 210), 0.48)
    if out is None:
        put_panel_line(panel, 'Detection: no valid stereo needle', y + 10, (80, 180, 255), 0.55)
        return panel

    y = put_panel_line(panel, f"Confidence: {out.get('conf', 0.0):.3f}", y + 6, (255, 255, 255), 0.5)
    y = put_panel_line(panel, f"Visible: {out.get('visible')}", y, (210, 210, 210), 0.46)
    y = put_panel_line(panel, '3D keypoints (mm):', y + 8, (0, 255, 255), 0.52)
    for i, xyz in enumerate(out['xyz_mm']):
        name = kp_names[i] if i < len(kp_names) else f'kp{i}'
        if xyz is None:
            line = f'{name:<4}: None'
        else:
            line = f'{name:<4}: [{xyz[0]:7.1f}, {xyz[1]:7.1f}, {xyz[2]:7.1f}]'
        y = put_panel_line(panel, line, y, (235, 235, 235), 0.45)

    pose = out['pose']
    R = np.asarray(pose['R'], float)
    t = np.asarray(pose['t'], float)
    y = put_panel_line(panel, 'Pose R (3x3):', y + 8, (0, 255, 255), 0.52)
    for row in R:
        y = put_panel_line(panel, f'[{row[0]: .4f} {row[1]: .4f} {row[2]: .4f}]', y,
                           (235, 235, 235), 0.45)
    y = put_panel_line(panel, 'Pose t (3x1, mm):', y + 6, (0, 255, 255), 0.52)
    for val in t:
        y = put_panel_line(panel, f'[{val: .3f}]', y, (235, 235, 235), 0.45)
    rvec = np.asarray(pose['rvec'], float)
    eu = rot_to_euler_deg(R)
    y = put_panel_line(panel, f'rvec: [{rvec[0]:.4f}, {rvec[1]:.4f}, {rvec[2]:.4f}]', y + 6,
                       (210, 210, 210), 0.44)
    put_panel_line(panel, f'euler XYZ: [{eu[0]:.1f}, {eu[1]:.1f}, {eu[2]:.1f}] deg', y,
                   (210, 210, 210), 0.44)
    return panel


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--calib', required=True)
    p.add_argument('--left'); p.add_argument('--right')
    p.add_argument('--capture'); p.add_argument('--layout', choices=['sbs', 'tb'], default='sbs')
    # dataset replay + ground-truth eval
    from pathlib import Path as _P
    p.add_argument('--root', type=_P, help='data_root (dataset replay mode)')
    p.add_argument('--dataset', help='sequence subdir, e.g. march_1 (dataset replay mode)')
    p.add_argument('--key', help='video key, e.g. 1_01')
    p.add_argument('--gt-subdir', default='keypoints',
                   help='sidecar dir to use as ground truth for metrics (set "" to disable)')
    p.add_argument('--pck-thresh', type=float, default=10.0, help='PCK pixel threshold (left view)')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--num-keypoints', type=int, default=5)
    p.add_argument('--needle-class', type=int, default=1)
    p.add_argument('--thread-class', type=int, default=2)
    p.add_argument('--seg-size', type=int, default=512, help='seg inference long-side (small=faster/less VRAM)')
    p.add_argument('--view-height', type=int, default=720,
                   help='downscale the L|R|panel canvas to this height for display+video '
                        '(big speed win on draw/encode; 0=full res). Window also auto-fits the desktop.')
    p.add_argument('--sam2-tools', default=os.path.join(os.path.dirname(__file__), '..', 'SAM2-Plus', 'tools'),
                   help='path to SAM2-Plus/tools (for needle_keypoints)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--no-smooth', action='store_true', help='disable Kalman pose smoothing')
    p.add_argument('--no-batch', action='store_true', help='segment eyes separately (default: one batched forward)')
    p.add_argument('--no-reproject', action='store_true',
                   help='disable pose-axes + keypoint reprojection overlay (verification)')
    p.add_argument('--show', action='store_true')
    p.add_argument('--save-video', default=None)
    p.add_argument('--save-poses', default=None, help='CSV: frame,kp0x,kp0y,kp0z,...,tx,ty,tz,rx,ry,rz')
    p.add_argument('--save-results', default=None,
                   help='JSONL: one record per frame (works for ALL input modes) with predicted '
                        'keypoints + pose + frame id/stem, for offline metric computation')
    p.add_argument('--print', dest='do_print', action='store_true', help='print xyz+pose per frame')
    args = p.parse_args()

    cfg = yaml.load(open(args.config, encoding='utf-8'), Loader=yaml.Loader)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda') and not args.no_amp
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    bundle = build_inference_model(cfg, args.checkpoint, device, visual_adapter=False)
    nk = load_nk(os.path.abspath(args.sam2_tools))
    calib = nk.load_calib(args.calib)
    # camera extrinsics for reprojection: left=cam1 (identity), right=cam2 (R,t)
    rvecL = np.zeros(3); tvecL = np.zeros(3)
    rvecR = cv2.Rodrigues(np.asarray(calib['R'], float))[0].ravel()
    tvecR = np.asarray(calib['t'], float).ravel()
    src = StereoSource(args)
    kp_names = (["tip"] + [f"k{i}" for i in range(1, args.num_keypoints - 1)] + ["tail"])
    pk = None if args.no_smooth else PoseKalman()

    # ground-truth (previously generated sidecars) for metrics
    import json
    gt_on = bool(args.dataset and args.gt_subdir)
    gt_dir = (args.root.resolve() / args.dataset / args.gt_subdir / args.key) if gt_on else None
    gt_index = {p.stem: p for p in gt_dir.rglob('*.json')} if (gt_on and gt_dir.is_dir()) else {}
    err_px = [[] for _ in range(args.num_keypoints)]      # per-kp left-view px error
    err_mm = [[] for _ in range(args.num_keypoints)]      # per-kp 3D error
    n_eval = 0

    # optional SEGMENTATION GT (dataset replay): per-frame left mask vs GT mask
    nclass = int(cfg['nclass'])
    seg_names = CLASSES.get(cfg.get('dataset'), [str(i) for i in range(nclass)])
    seg_gt_dir = (args.root.resolve() / args.dataset / 'masks' / args.key) if args.dataset else None
    seg_index = {p.stem: p for p in seg_gt_dir.rglob('*.png')} if (seg_gt_dir and seg_gt_dir.is_dir()) else {}
    seg_inter = np.zeros(nclass); seg_union = np.zeros(nclass); seg_tgt = np.zeros(nclass)
    n_seg = 0

    def load_gt(stem):
        p = gt_index.get(stem)
        if p is None:
            return None
        nd = json.loads(p.read_text(encoding='utf-8')).get('needle')
        return nd['keypoints'] if nd else None

    writer = None
    WIN = 'stereo keypoints (q=quit)'
    win_ready = False
    pcsv = open(args.save_poses, 'w') if args.save_poses else None
    if pcsv:
        hdr = (['frame'] + [f'kp{i}_{a}' for i in range(args.num_keypoints) for a in 'xyz']
               + ['tx', 'ty', 'tz', 'rvx', 'rvy', 'rvz']
               + [f'R{r}{c}' for r in range(3) for c in range(3)]
               + ['eul_x', 'eul_y', 'eul_z'])
        pcsv.write(','.join(hdr) + '\n')
    rjsonl = open(args.save_results, 'w', encoding='utf-8') if args.save_results else None
    fps = 0.0
    fi = 0
    print('[realtime] started — press q to quit' + ('' if use_amp else '  (amp off)'))
    while True:
        t0 = time.perf_counter()
        L, R, stem = src.read()
        if L is None:
            break
        if args.no_batch:
            ml = seg_mask(bundle, L, args.seg_size, device, use_amp)
            mr = seg_mask(bundle, R, args.seg_size, device, use_amp)
        else:
            ml, mr = seg_masks_batch(bundle, [L, R], args.seg_size, device, use_amp)
        needleL = ml == args.needle_class; needleR = mr == args.needle_class
        threadL = ml == args.thread_class
        out = None
        if needleL.sum() >= 20 and needleR.sum() >= 20:
            try:
                out, _ = nk.process_frame(needleL, needleR, threadL, calib, args.num_keypoints)
            except Exception:
                out = None
        # ---- pose smoothing (Kalman) ----
        if pk is not None:
            if out is not None:
                ts, rs = pk.update(out['pose']['t'], out['pose']['rvec'])
                out['pose']['t'] = list(map(float, ts))
                out['pose']['rvec'] = list(map(float, rs))
                out['pose']['R'] = cv2.Rodrigues(np.asarray(rs, float))[0].tolist()
            else:
                pk.coast()
        dt = time.perf_counter() - t0
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6)) if fps else 1.0 / max(dt, 1e-6)

        # ---- metrics vs ground-truth sidecar ----
        gt_kps = load_gt(stem) if gt_on else None
        if out is not None and gt_kps and len(gt_kps) >= args.num_keypoints:
            n_eval += 1
            for i in range(args.num_keypoints):
                g = gt_kps[i]
                px = ((out['left'][i][0] - g['x']) ** 2 + (out['left'][i][1] - g['y']) ** 2) ** 0.5
                err_px[i].append(px)
                if out['xyz_mm'][i] is not None and g.get('xyz_mm'):
                    mm = float(np.linalg.norm(np.asarray(out['xyz_mm'][i]) - np.asarray(g['xyz_mm'])))
                    err_mm[i].append(mm)

        # ---- segmentation metrics vs GT mask (dataset replay only) ----
        if stem is not None and stem in seg_index:
            gtm = cv2.imread(str(seg_index[stem]), cv2.IMREAD_UNCHANGED)
            if gtm is not None:
                if gtm.ndim == 3:
                    gtm = gtm[:, :, 0]
                if gtm.shape[:2] == ml.shape[:2]:
                    i_, u_, t_ = intersectionAndUnion(
                        ml.astype(np.int64), gtm.astype(np.int64), nclass, 255)
                    seg_inter += i_; seg_union += u_; seg_tgt += t_; n_seg += 1

        # ---- outputs ----
        if out is not None:
            xyz = out['xyz_mm']; pose = out['pose']
            if args.do_print:
                tt = pose['t']; rv = pose['rvec']
                print(f'[{fi}] tip_xyz={np.round(xyz[0],1).tolist()}mm  '
                      f'pose_t={np.round(tt,1).tolist()}mm rvec={np.round(rv,3).tolist()}  fps={fps:.1f}',
                      flush=True)
            if pcsv:
                Rflat = [v for rowR in pose['R'] for v in rowR]      # 9 rotation-matrix elements
                eu = list(rot_to_euler_deg(pose['R']))               # euler XYZ (deg)
                row = ([fi] + [v for pt in xyz for v in pt] + list(pose['t'])
                       + list(pose['rvec']) + Rflat + eu)
                pcsv.write(','.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in row) + '\n')

        # ---- unified per-frame result record (ALL input modes) ----
        if rjsonl is not None:
            if out is not None:
                needle = {"keypoints": [
                    {"name": kp_names[i], "x": out['left'][i][0], "y": out['left'][i][1],
                     "x_right": out['right'][i][0], "y_right": out['right'][i][1],
                     "xyz_mm": out['xyz_mm'][i], "visible": int(out['visible'][i])}
                    for i in range(args.num_keypoints)],
                    "pose": out['pose'], "conf": out['conf']}
            else:
                needle = None
            rec = {"frame": fi, "stem": stem, "fps": round(fps, 2), "needle": needle}
            rjsonl.write(json.dumps(rec) + '\n')

        # ---- visualization ----
        if args.show or args.save_video:
            visL = overlay_segmentation(L, ml, args.needle_class, args.thread_class)
            visR = overlay_segmentation(R, mr, args.needle_class, args.thread_class)
            rep_err = reprojection_error(out, calib, rvecR, tvecR)
            if out is not None:
                nk.draw_debug(visL, out['left'], kp_names, out['visible'], tag=None)
                nk.draw_debug(visR, out['right'], kp_names, out['visible'], tag=None)
                if not args.no_reproject:      # pose-axes + keypoint reprojection (verification)
                    Ro = out['pose']['R']; to = out['pose']['t']
                    draw_pose_axes(visL, Ro, to, calib['K1'], calib['D1'], rvecL, tvecL)
                    draw_pose_axes(visR, Ro, to, calib['K2'], calib['D2'], rvecR, tvecR)
                    draw_reproj(visL, out['xyz_mm'], calib['K1'], calib['D1'], rvecL, tvecL)
                    draw_reproj(visR, out['xyz_mm'], calib['K2'], calib['D2'], rvecR, tvecR)
            canvas = hstack_same_h(visL, visR)   # right info panel removed (text too small to read)
            cv2.putText(canvas, f'FPS {fps:5.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, f'FPS {fps:5.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            if out is not None:
                t = out['pose']['t']; eu = rot_to_euler_deg(out['pose']['R'])
                line1 = f"t=({t[0]:.0f},{t[1]:.0f},{t[2]:.0f})mm"
                line2 = f"rot=({eu[0]:.0f},{eu[1]:.0f},{eu[2]:.0f})deg"
                line3 = "reproj=N/A" if rep_err is None else f"reproj={rep_err['mean']:.2f}px"
                for k, s in enumerate((line1, line2, line3)):
                    yk = 64 + 28 * k
                    cv2.putText(canvas, s, (10, yk), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(canvas, s, (10, yk), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
            # downscale the (wide) L|R|panel canvas -> faster encode/display + fits desktop
            if args.view_height and canvas.shape[0] > args.view_height:
                s = args.view_height / canvas.shape[0]
                canvas = cv2.resize(canvas, (int(round(canvas.shape[1] * s)), args.view_height),
                                    interpolation=cv2.INTER_AREA)
            if args.save_video:
                if writer is None:
                    h, w = canvas.shape[:2]
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*'mp4v'),
                                             20.0, (w, h))
                writer.write(canvas)
            if args.show:
                if not win_ready:
                    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    fit_window(WIN, canvas.shape[1], canvas.shape[0])
                    win_ready = True
                cv2.imshow(WIN, canvas)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
        fi += 1

    src.release()
    if writer is not None:
        writer.release()
    if pcsv:
        pcsv.close()
    if rjsonl is not None:
        rjsonl.close()
        print(f'[results] per-frame predictions saved -> {args.save_results}')
    cv2.destroyAllWindows()
    print(f'[realtime] done — {fi} frames, ~{fps:.1f} fps')

    # ---- segmentation metrics summary (left eye vs GT masks) ----
    if n_seg > 0:
        iou = seg_inter / np.maximum(seg_union, 1e-9) * 100
        dice = 2 * seg_inter / np.maximum(seg_inter + seg_union, 1e-9) * 100
        pres = seg_tgt > 0
        print(f'\n[metrics] segmentation (left eye vs GT mask)  frames evaluated = {n_seg}')
        print(f'   mIoU={np.mean(iou[pres]):.2f}  mDice={np.mean(dice[pres]):.2f}  (present-only)')
        for c in range(nclass):
            if pres[c]:
                print(f'   {seg_names[c]:<10} IoU={iou[c]:6.2f}  Dice={dice[c]:6.2f}')

    # ---- keypoint metrics summary (dataset replay + GT sidecars) ----
    if gt_on and n_eval > 0:
        print(f'\n[metrics] vs GT sidecars ({args.gt_subdir})  frames evaluated = {n_eval}')
        all_px = []
        for i in range(args.num_keypoints):
            if err_px[i]:
                m = float(np.mean(err_px[i])); all_px += err_px[i]
                mm = (f'  3D={np.mean(err_mm[i]):.1f}mm' if err_mm[i] else '')
                print(f'   {kp_names[i]:<4} mean px err = {m:6.2f}{mm}')
        if all_px:
            arr = np.asarray(all_px)
            pck = float((arr <= args.pck_thresh).mean()) * 100
            print(f'   OVERALL mean px err = {arr.mean():.2f}  '
                  f'PCK@{args.pck_thresh:.0f}px = {pck:.1f}%  '
                  f'(median {np.median(arr):.2f})')
    elif gt_on:
        print('[metrics] no overlapping GT frames found (check --gt-subdir / sidecars exist)')


if __name__ == '__main__':
    main()
