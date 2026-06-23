"""Export the segmentation model to an accelerated engine (TorchScript / TensorRT).

Step 6 of the inference-time acceleration plan. Produces an engine whose
forward(x: (B,3,H,W) fp16) -> logits (B, nclass, mh, mw) is a drop-in replacement
for bundle['model'], loaded at runtime by infer_accel.load_seg_engine and used
by the v3 accelerated drivers via `--seg-engine`.

The engine is FIXED-SHAPE and HARDWARE-SPECIFIC: it is built for one input size
(B, 3, infer_h, infer_w). That size is exactly what seg_masks_batch feeds the
model, derived from --seg-size + the source frame size + the backbone patch:
    s = seg_size / max(H, W)         (only if max(H,W) > seg_size)
    sh, sw   = round(H*s), round(W*s)
    infer_h  = max(patch, round(sh/patch)*patch)
    infer_w  = max(patch, round(sw/patch)*patch)
So pass the SAME --seg-size and the source frame --src-h/--src-w you will run
with, and BUILD ON THE TARGET GPU. Batch is 2 (both eyes in one forward).

Requires (TensorRT path): `pip install torch-tensorrt` matching your torch/CUDA.
If torch_tensorrt is missing or compile fails, falls back to a TorchScript engine
(still removes Python/kernel-launch overhead, just no TRT kernel fusion).

Example (1080x1920 source, seg-size 640, both eyes):
    python export_seg_engine.py \
        --config configs/surgical_combined_base.yaml \
        --checkpoint exp/combined_r100_base/best.pth \
        --src-h 1080 --src-w 1920 --seg-size 640 \
        --out exp/combined_r100_base/seg_engine_640.ts
Then run inference with:
    python realtime_stereo_keypoints_v3_accel.py ... --seg-size 640 \
        --seg-engine exp/combined_r100_base/seg_engine_640.ts
"""
import argparse
import os

import torch
import yaml

from test import build_inference_model


def infer_size(src_h, src_w, seg_size, patch):
    if seg_size and max(src_h, src_w) > seg_size:
        s = seg_size / float(max(src_h, src_w))
        sh, sw = int(round(src_h * s)), int(round(src_w * s))
    else:
        sh, sw = src_h, src_w
    ih = max(patch, int(round(sh / patch)) * patch)
    iw = max(patch, int(round(sw / patch)) * patch)
    return ih, iw


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True, help='output engine path (.ts)')
    p.add_argument('--src-h', type=int, default=1080, help='source frame height (left eye)')
    p.add_argument('--src-w', type=int, default=1920, help='source frame width (left eye)')
    p.add_argument('--seg-size', type=int, default=640, help='MUST match the run-time --seg-size')
    p.add_argument('--batch', type=int, default=2, help='2 = both eyes in one forward')
    p.add_argument('--format', choices=['tensorrt', 'torchscript', 'auto'], default='auto',
                   help='auto = try TensorRT, fall back to TorchScript')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    # xformers' memory_efficient_attention runs through a custom autograd Function
    # (_Unbind) that TorchScript/TensorRT cannot serialize -> torch.jit.save fails
    # with "Could not export Python function call '_Unbind'". DINOv2's MemEffAttention
    # has a built-in fallback to the equivalent PyTorch SDPA path (Attention.forward,
    # same math, identical fp16 result) when XFORMERS_AVAILABLE is False. Force that
    # path for export so the graph is traceable/serializable; runtime accuracy is
    # unchanged. The exported engine therefore uses SDPA attention.
    try:
        from model.backbone.dinov2_layers import attention as _attn
        if getattr(_attn, 'XFORMERS_AVAILABLE', False):
            _attn.XFORMERS_AVAILABLE = False
            print('[export] xformers attention disabled for export -> PyTorch SDPA '
                  '(same math, fp16-identical); the exported engine is serializable.')
    except Exception as e:  # noqa
        print(f'[export] note: could not toggle xformers attention flag ({e}); '
              'export may fail on _Unbind if xformers is active.')

    cfg = yaml.load(open(args.config, encoding='utf-8'), Loader=yaml.Loader)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    bundle = build_inference_model(cfg, args.checkpoint, device, visual_adapter=False)
    if bundle.get('affinity_side') is not None or bundle.get('use_edge_enhance'):
        raise SystemExit('[export] model has an affinity/edge head -> the batched forward '
                         'is bypassed at runtime, so an exported engine would never be used. '
                         'Export a plain segmentation model instead.')
    model = bundle['model'].eval()
    patch = bundle['patch_size']
    ih, iw = infer_size(args.src_h, args.src_w, args.seg_size, patch)
    shape = (args.batch, 3, ih, iw)
    print(f'[export] patch={patch}  src={args.src_h}x{args.src_w}  seg_size={args.seg_size}'
          f'  -> engine input shape {shape}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)

    def export_torchscript():
        m = model.half()
        ex = torch.randn(*shape, device=device, dtype=torch.float16)
        with torch.inference_mode():
            ts = torch.jit.trace(m, ex, check_trace=False)
            ts = torch.jit.freeze(ts)
        torch.jit.save(ts, args.out)
        print(f'[export] TorchScript engine -> {args.out}')

    def export_tensorrt():
        import torch_tensorrt
        m = model.half()                          # match fp16 input (conv bias must be half too)
        ex_in = torch_tensorrt.Input(shape, dtype=torch.half)
        with torch.inference_mode():
            trt = torch_tensorrt.compile(
                m, inputs=[ex_in], enabled_precisions={torch.half})
        torch.jit.save(trt, args.out)
        print(f'[export] Torch-TensorRT FP16 engine -> {args.out}')

    if args.format == 'torchscript':
        export_torchscript()
    elif args.format == 'tensorrt':
        export_tensorrt()
    else:  # auto
        try:
            export_tensorrt()
        except Exception as e:  # noqa
            print(f'[export] TensorRT export failed ({e})\n[export] falling back to TorchScript')
            export_torchscript()

    # sanity: reload + forward once
    from infer_accel import load_seg_engine
    eng = load_seg_engine(args.out, device)
    with torch.inference_mode():
        y = eng(torch.randn(*shape, device=device))
    print(f'[export] OK — reload forward produced logits {tuple(y.shape)}')


if __name__ == '__main__':
    main()
