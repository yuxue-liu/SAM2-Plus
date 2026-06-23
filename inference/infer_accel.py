"""Inference-time acceleration helpers for the needle keypoint/pose pipeline.

This module is SHARED by the v3 accelerated drivers
(`realtime_stereo_keypoints_v3_accel.py`, `eval_pose_val_v3_accel.py`). It does
NOT change the segmentation logic or the pose method — only how the SAME forward
pass is executed and how frame I/O / video encoding overlap with the GPU.

Three independent, opt-in levers (steps 4-6 of docs/NEEDLE_POSE_REGISTRATION.md
acceleration notes):

  4. ASYNC PIPELINE (`PrefetchReader`, `AsyncVideoWriter`): read the next stereo
     pair on a background thread and encode the previous canvas on another, so
     disk decode + mp4 encode overlap with the GPU forward instead of serializing.
     Pure throughput win, zero effect on numbers.

  5. torch.compile + channels_last (`accelerate_model`): fuse the graph and cut
     kernel-launch overhead; channels_last helps the DPT conv head. No accuracy
     change (fp16 math identical). Needs a fixed input shape to stay fast — the
     pipeline already pads to a patch multiple, so shape is stable per seg-size.

  6. TENSORRT / TORCHSCRIPT BACKEND (`load_seg_engine`): load a pre-exported
     engine (built by export_seg_engine.py) whose forward(x)->logits matches the
     PyTorch model, and drop it into the bundle. FP16 TRT is the biggest single
     forward-pass speedup. The engine is hardware/shape specific: build it ON the
     target GPU at the seg-size you will run.

All three degrade gracefully: missing torch_tensorrt / compile errors fall back
to plain eager execution with a printed warning, so a v3 run never hard-fails
just because an accel backend is unavailable.
"""
import os
import queue
import threading

import torch


# --------------------------------------------------------------------------- #
# step 6: pre-exported accelerated segmentation backend
# --------------------------------------------------------------------------- #
class _HalfInputEngine(torch.nn.Module):
    """Cast the input to fp16 before calling the engine. Both export formats
    (TorchScript .half() and Torch-TensorRT FP16) declare half inputs, but the
    caller (seg_masks_batch) builds a float32 tensor, so we cast here. Keeps the
    drop-in model(x)->logits contract."""
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def forward(self, x):
        return self.engine(x.half())


def load_seg_engine(path, device):
    """Load a TorchScript / Torch-TensorRT engine saved by export_seg_engine.py.

    Returns an nn.Module-like object whose forward(x: (B,3,H,W)) -> logits
    (B, nclass, mh, mw), i.e. a drop-in replacement for bundle['model']."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"seg engine not found: {path}")
    # torch_tensorrt engines deserialize through torch.jit and need the runtime
    # registered; import it if present (harmless if the engine is plain TS).
    try:
        import torch_tensorrt  # noqa: F401
    except Exception:
        pass
    m = torch.jit.load(path, map_location=device).eval()
    return _HalfInputEngine(m)


def attach_engine_to_bundle(bundle, engine_path, device):
    """Swap a pre-exported engine in for bundle['model'] so seg_masks_batch uses
    it transparently. Requires the plain batched forward (no affinity/edge head)."""
    if bundle.get('affinity_side') is not None or bundle.get('use_edge_enhance'):
        print('[accel] WARNING: model has affinity/edge head -> batched forward is '
              'bypassed; the engine will NOT be used. Export/run a plain seg model.')
        return bundle
    bundle['model'] = load_seg_engine(engine_path, device)
    print(f'[accel] segmentation backend = pre-exported engine: {engine_path}')
    return bundle


# --------------------------------------------------------------------------- #
# step 5: torch.compile + channels_last
# --------------------------------------------------------------------------- #
class _ChannelsLast(torch.nn.Module):
    """Wrap a model so inputs are converted to channels_last before forward."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x.contiguous(memory_format=torch.channels_last))


def accelerate_model(model, compile=False, channels_last=False,
                     mode='reduce-overhead'):
    """Optionally wrap bundle['model'] with channels_last + torch.compile.
    Falls back to the original model on any failure (prints a warning)."""
    if channels_last:
        try:
            model = _ChannelsLast(model.to(memory_format=torch.channels_last))
            print('[accel] channels_last memory format enabled')
        except Exception as e:  # noqa
            print(f'[accel] channels_last failed ({e}); skipping')
    if compile:
        try:
            model = torch.compile(model, mode=mode)
            print(f'[accel] torch.compile enabled (mode={mode}); '
                  'first 1-2 frames are slow while it traces/optimizes')
        except Exception as e:  # noqa
            print(f'[accel] torch.compile failed ({e}); running eager')
    return model


def accelerate_bundle(bundle, compile=False, channels_last=False,
                      mode='reduce-overhead'):
    """In-place: apply compile/channels_last to bundle['model'] if it is plainly
    forward-callable (no affinity/edge branch)."""
    if not (compile or channels_last):
        return bundle
    if bundle.get('affinity_side') is not None or bundle.get('use_edge_enhance'):
        print('[accel] WARNING: model has affinity/edge head -> batched forward is '
              'bypassed; compile/channels_last will NOT take effect on it.')
        return bundle
    bundle['model'] = accelerate_model(bundle['model'], compile=compile,
                                       channels_last=channels_last, mode=mode)
    return bundle


# --------------------------------------------------------------------------- #
# step 4: async I/O pipeline
# --------------------------------------------------------------------------- #
class PrefetchReader:
    """Background-thread wrapper around a source with .read()->(L,R,stem) and
    .release(). Reads the next stereo pair while the main thread runs the GPU
    forward, hiding disk decode latency. End of stream yields (None,None,None)."""
    def __init__(self, src, queue_size=4):
        self.src = src
        self.q = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self._stop.is_set():
            item = self.src.read()
            self.q.put(item)
            if item[0] is None:        # sentinel: end of stream
                break

    def read(self):
        return self.q.get()

    def release(self):
        self._stop.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.src.release()


class AsyncVideoWriter:
    """Background-thread cv2.VideoWriter so mp4 encoding overlaps the GPU forward.
    Same constructor signature as cv2.VideoWriter; .write(frame) is non-blocking
    until the queue fills."""
    def __init__(self, path, fourcc, fps, size, queue_size=8):
        import cv2
        self.w = cv2.VideoWriter(path, fourcc, fps, size)
        self.q = queue.Queue(maxsize=max(1, queue_size))
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            frame = self.q.get()
            if frame is None:
                break
            self.w.write(frame)

    def write(self, frame):
        self.q.put(frame)

    def release(self):
        self.q.put(None)
        self.t.join()
        self.w.release()
