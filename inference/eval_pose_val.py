"""Evaluate seg-model needle keypoints + 6-DoF pose on the VAL stereo videos.

For every stereo video (key) that contributes frames to the val split, this replays
the COMPLETE sequence through:

    segmentation model -> needle/thread masks -> needle_keypoints.process_frame
    -> stereo triangulation -> N 3D keypoints (xyz_mm) + 6-DoF pose (R,t,rvec)

and for each frame:
  * reprojects the triangulated 3D keypoints back into BOTH views and measures the
    reprojection error in px  -> accuracy of triangulation/pose, needs NO ground truth;
  * where GT keypoint sidecars exist (e.g. march_*), also reports 2D px + 3D mm error;
  * saves a per-key inference video with seg overlay, reprojected keypoints + pose
    axes, live FPS and reprojection error burned in;
  * tracks peak VRAM (kept under --vram-cap-gb) and processing FPS.

Efficiency knobs (VRAM bounded by --seg-size): fp16 autocast + BOTH eyes in one
batched forward + cudnn.benchmark + inference_mode. Pick the largest --seg-size
whose reported peak VRAM stays under the cap.

Run (all val stereo videos):
  python eval_pose_val.py \
    --config configs/surgical_combined_base.yaml \
    --checkpoint exp/combined_r100_base/best.pth \
    --calib /root/autodl-tmp/code/SAM2-Plus/tools/needle_calib.json \
    --root /root/autodl-tmp/data/surgical_seg \
    --val-split /root/autodl-tmp/data/surgical_seg/combined/splits/r100/val.txt \
    --out-dir exp/pose_val --seg-size 512 --num-keypoints 5
"""
import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2
import torch

from test import build_inference_model
from realtime_stereo_keypoints import (
    StereoSource, seg_masks_batch, seg_mask, reprojection_error,
    draw_pose_axes, draw_reproj, build_info_panel, overlay_segmentation,
    hstack_same_h, rot_to_euler_deg, PoseKalman, load_nk,
)
import yaml


def parse_val_keys(val_split, root):
    """From a val.txt (frame list), return ordered (dataset, key) pairs that have a
    stereo_right/ dir, plus the set of val stems per key (for val-only GT metrics)."""
    pairs, val_stems = [], {}
    seen = set()
    for line in Path(val_split).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        img = line.split('\t')[0].split(' ')[0]
        parts = img.split('/')
        if len(parts) < 4 or parts[1] != 'images':
            continue
        dataset, key = parts[0], parts[2]
        stem = os.path.splitext(parts[-1])[0]
        val_stems.setdefault((dataset, key), set()).add(stem)
        if (dataset, key) in seen:
            continue
        if (Path(root) / dataset / 'stereo_right').is_dir():
            pairs.append((dataset, key))
            seen.add((dataset, key))
    return pairs, val_stems


def run_key(dataset, key, bundle, nk, calib, rvecR, tvecR, kp_names, args, device, use_amp):
    """Replay one COMPLETE stereo video; return per-frame stats and write a video."""
    src = StereoSource(SimpleNamespace(
        dataset=dataset, key=key, root=Path(args.root), limit=args.limit,
        capture=None, left=None, right=None))
    gt_dir = (Path(args.root) / dataset / args.gt_subdir / key) if args.gt_subdir else None
    gt_index = {p.stem: p for p in gt_dir.rglob('*.json')} if (gt_dir and gt_dir.is_dir()) else {}
    val_stems = args._val_stems.get((dataset, key))

    pk = None if args.no_smooth else PoseKalman()
    out_path = os.path.join(args.out_dir, f'{dataset}__{key}.mp4')
    writer = None
    K = args.num_keypoints

    reproj_all, fps_all = [], []
    err_px = [[] for _ in range(K)]
    err_mm = [[] for _ in range(K)]
    n_frames = n_det = n_gt = 0
    fps = 0.0

    while True:
        t0 = time.perf_counter()
        L, R, stem = src.read()
        if L is None:
            break
        n_frames += 1
        ml, mr = seg_masks_batch(bundle, [L, R], args.seg_size, device, use_amp)
        needleL = ml == args.needle_class
        needleR = mr == args.needle_class
        threadL = ml == args.thread_class
        out = None
        if needleL.sum() >= 20 and needleR.sum() >= 20:
            try:
                out, _ = nk.process_frame(needleL, needleR, threadL, calib, K)
            except Exception:
                out = None
        if pk is not None and out is not None:
            ts, rs = pk.update(out['pose']['t'], out['pose']['rvec'])
            out['pose']['t'] = list(map(float, ts))
            out['pose']['rvec'] = list(map(float, rs))
            out['pose']['R'] = cv2.Rodrigues(np.asarray(rs, float))[0].tolist()
        elif pk is not None:
            pk.coast()
        dt = time.perf_counter() - t0
        fps = (0.9 * fps + 0.1 / max(dt, 1e-6)) if fps else 1.0 / max(dt, 1e-6)
        fps_all.append(1.0 / max(dt, 1e-6))

        rep = reprojection_error(out, calib, rvecR, tvecR)
        if out is not None:
            n_det += 1
        if rep is not None:
            reproj_all.append(rep['mean'])

        # ---- GT accuracy (only on annotated frames; optionally val-only) ----
        use_for_gt = (stem in gt_index) and (val_stems is None or stem in val_stems)
        if out is not None and use_for_gt:
            nd = json.loads(gt_index[stem].read_text(encoding='utf-8')).get('needle')
            gt_kps = nd['keypoints'] if nd else None
            if gt_kps and len(gt_kps) >= K:
                n_gt += 1
                for i in range(K):
                    g = gt_kps[i]
                    px = float(np.hypot(out['left'][i][0] - g['x'], out['left'][i][1] - g['y']))
                    err_px[i].append(px)
                    if out['xyz_mm'][i] is not None and g.get('xyz_mm'):
                        err_mm[i].append(float(np.linalg.norm(
                            np.asarray(out['xyz_mm'][i]) - np.asarray(g['xyz_mm']))))

        # ---- visualization / video ----
        visL = overlay_segmentation(L, ml, args.needle_class, args.thread_class)
        visR = overlay_segmentation(R, mr, args.needle_class, args.thread_class)
        if out is not None:
            nk.draw_debug(visL, out['left'], kp_names, out['visible'], tag=None)
            nk.draw_debug(visR, out['right'], kp_names, out['visible'], tag=None)
            if not args.no_reproject:
                Ro, to = out['pose']['R'], out['pose']['t']
                draw_pose_axes(visL, Ro, to, calib['K1'], calib['D1'], np.zeros(3), np.zeros(3))
                draw_pose_axes(visR, Ro, to, calib['K2'], calib['D2'], rvecR, tvecR)
                draw_reproj(visL, out['xyz_mm'], calib['K1'], calib['D1'], np.zeros(3), np.zeros(3))
                draw_reproj(visR, out['xyz_mm'], calib['K2'], calib['D2'], rvecR, tvecR)
        canvas = hstack_same_h(visL, visR)   # right info panel removed (text too small to read)
        tag = f'{dataset}/{key}  frame {n_frames}'
        for s, yk, col in ((f'FPS {fps:5.1f}', 30, (0, 255, 0)),
                           (tag, 58, (0, 255, 255)),
                           ('reproj=N/A' if rep is None else f"reproj={rep['mean']:.2f}px (max {rep['max']:.1f})",
                            84, (0, 255, 255))):
            cv2.putText(canvas, s, (10, yk), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, s, (10, yk), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 1, cv2.LINE_AA)
        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                     float(args.fps_out), (w, h))
        writer.write(canvas)

    if writer is not None:
        writer.release()
    src.release()
    return dict(dataset=dataset, key=key, video=out_path, n_frames=n_frames,
                n_det=n_det, n_gt=n_gt, reproj_all=reproj_all, fps_all=fps_all,
                err_px=err_px, err_mm=err_mm)


def summarize(stats, K, kp_names, pck_thresh):
    rep = np.concatenate([np.asarray(s['reproj_all']) for s in stats]) if stats else np.array([])
    fpsv = np.concatenate([np.asarray(s['fps_all']) for s in stats]) if stats else np.array([])
    px_all = [np.concatenate([np.asarray(s['err_px'][i]) for s in stats]) if stats else np.array([])
              for i in range(K)]
    mm_all = [np.concatenate([np.asarray(s['err_mm'][i]) for s in stats]) if stats else np.array([])
              for i in range(K)]
    flat_px = np.concatenate(px_all) if any(len(a) for a in px_all) else np.array([])
    out = {
        'frames': int(sum(s['n_frames'] for s in stats)),
        'detected': int(sum(s['n_det'] for s in stats)),
        'gt_frames': int(sum(s['n_gt'] for s in stats)),
        'fps_mean': float(fpsv.mean()) if len(fpsv) else 0.0,
        'fps_median': float(np.median(fpsv)) if len(fpsv) else 0.0,
        'reproj_mean_px': float(rep.mean()) if len(rep) else float('nan'),
        'reproj_median_px': float(np.median(rep)) if len(rep) else float('nan'),
        'per_kp': {kp_names[i]: {
            'px_mean': float(px_all[i].mean()) if len(px_all[i]) else float('nan'),
            'mm_mean': float(mm_all[i].mean()) if len(mm_all[i]) else float('nan')}
            for i in range(K)},
        'gt_px_mean': float(flat_px.mean()) if len(flat_px) else float('nan'),
        'gt_px_median': float(np.median(flat_px)) if len(flat_px) else float('nan'),
        f'PCK@{pck_thresh:g}px': float((flat_px <= pck_thresh).mean() * 100) if len(flat_px) else float('nan'),
    }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--calib', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--val-split', default=None,
                   help='val.txt; keys with stereo are auto-selected. Omit to use --datasets.')
    p.add_argument('--datasets', nargs='*', default=None,
                   help='explicit dataset list (each: all keys). Overrides --val-split key discovery.')
    p.add_argument('--out-dir', default='exp/pose_val')
    p.add_argument('--num-keypoints', type=int, default=5)
    p.add_argument('--needle-class', type=int, default=1)
    p.add_argument('--thread-class', type=int, default=2)
    p.add_argument('--seg-size', type=int, default=512,
                   help='seg inference long-side. Larger=more accurate+more VRAM. VRAM cap knob.')
    p.add_argument('--vram-cap-gb', type=float, default=4.0)
    p.add_argument('--gt-subdir', default='keypoints', help='GT sidecar dir ("" to disable GT metrics)')
    p.add_argument('--pck-thresh', type=float, default=10.0)
    p.add_argument('--limit', type=int, default=0, help='cap frames per key (0=all=complete video)')
    p.add_argument('--fps-out', type=float, default=20.0, help='output video playback fps')
    p.add_argument('--sam2-tools', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        '..', 'SAM2-Plus', 'tools'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--no-smooth', action='store_true')
    p.add_argument('--no-reproject', action='store_true')
    args = p.parse_args()

    cfg = yaml.load(open(args.config, encoding='utf-8'), Loader=yaml.Loader)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda') and not args.no_amp
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    os.makedirs(args.out_dir, exist_ok=True)

    bundle = build_inference_model(cfg, args.checkpoint, device, visual_adapter=False)
    nk = load_nk(os.path.abspath(args.sam2_tools))
    calib = nk.load_calib(args.calib)
    rvecR = cv2.Rodrigues(np.asarray(calib['R'], float))[0].ravel()
    tvecR = np.asarray(calib['t'], float).ravel()
    kp_names = (['tip'] + [f'k{i}' for i in range(1, args.num_keypoints - 1)] + ['tail'])

    # ---- decide which (dataset, key) complete videos to run ----
    args._val_stems = {}
    if args.datasets:
        keys = []
        for d in args.datasets:
            meta = json.loads((Path(args.root) / d / 'meta.json').read_text(encoding='utf-8'))
            keys += [(d, k) for k in meta['videos'].keys()]
    else:
        assert args.val_split, 'provide --val-split or --datasets'
        keys, args._val_stems = parse_val_keys(args.val_split, args.root)
    print(f'[eval] {len(keys)} stereo video(s) to run: '
          + ', '.join(f'{d}/{k}' for d, k in keys))

    stats = []
    t_start = time.perf_counter()
    for d, k in keys:
        print(f'[eval] >>> {d}/{k} ...', flush=True)
        s = run_key(d, k, bundle, nk, calib, rvecR, tvecR, kp_names, args, device, use_amp)
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0
        rep = np.asarray(s['reproj_all'])
        print(f'    frames={s["n_frames"]} det={s["n_det"]} gt={s["n_gt"]}  '
              f'reproj_mean={rep.mean():.2f}px  ' if len(rep) else
              f'    frames={s["n_frames"]} det={s["n_det"]}  reproj=N/A  ', flush=True)
        print(f'    peak VRAM so far = {peak:.2f} GB  -> {s["video"]}', flush=True)
        stats.append(s)

    summary = summarize(stats, args.num_keypoints, kp_names, args.pck_thresh)
    summary['peak_vram_gb'] = (torch.cuda.max_memory_allocated() / 1e9) if device.type == 'cuda' else 0.0
    summary['wall_seconds'] = time.perf_counter() - t_start
    summary['seg_size'] = args.seg_size
    summary['videos'] = [s['video'] for s in stats]
    summary['per_key'] = [{
        'dataset': s['dataset'], 'key': s['key'], 'frames': s['n_frames'], 'det': s['n_det'],
        'gt': s['n_gt'],
        'reproj_mean_px': float(np.mean(s['reproj_all'])) if s['reproj_all'] else float('nan'),
        'fps_mean': float(np.mean(s['fps_all'])) if s['fps_all'] else 0.0,
    } for s in stats]

    out_json = os.path.join(args.out_dir, 'summary.json')
    Path(out_json).write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('\n================ VAL POSE / KEYPOINT SUMMARY ================')
    print(f'videos={len(stats)}  frames={summary["frames"]}  detected={summary["detected"]}'
          f'  gt_frames={summary["gt_frames"]}')
    print(f'seg_size={args.seg_size}  PEAK VRAM={summary["peak_vram_gb"]:.2f} GB'
          f'  (cap {args.vram_cap_gb:.1f} GB) -> {"OK" if summary["peak_vram_gb"] <= args.vram_cap_gb else "OVER CAP!"}')
    print(f'FPS mean/median = {summary["fps_mean"]:.1f} / {summary["fps_median"]:.1f}')
    print(f'Reprojection error (3D->2D, both views): mean={summary["reproj_mean_px"]:.2f}px '
          f'median={summary["reproj_median_px"]:.2f}px')
    if summary['gt_frames']:
        print(f'GT 2D error: mean={summary["gt_px_mean"]:.2f}px median={summary["gt_px_median"]:.2f}px '
              f'PCK@{args.pck_thresh:g}px={summary[f"PCK@{args.pck_thresh:g}px"]:.1f}%')
        for n, v in summary['per_kp'].items():
            print(f'   {n:<4} px={v["px_mean"]:6.2f}  mm={v["mm_mean"]:6.2f}')
    print(f'summary -> {out_json}')


if __name__ == '__main__':
    main()
