# 缝针分割与 6-DoF 位姿估计

双目内镜下，对手术缝针做**语义分割**（针 / 缝线 / 持针钳）、**关键点定位**与 **6-DoF 位姿估计**。
分割用 DINOv2-base + DPT；位姿用"分割 → 中心线 → 立体三角化 → 圆弧/模型配准"的几何方法。
支持验证集回放、双路视频流、双目采集卡三种输入，并提供 TensorRT 加速推理。

> 上传到本仓库时，请把本文件作为仓库根 `README.md`。

## 文档导航
- 📖 **操作手册** [`OPERATION_MANUAL.md`](OPERATION_MANUAL.md) — 推理三版本用法、数据全流程、方法简介（**先看这个**）。
- 🎯 位姿配准算法 [`NEEDLE_POSE_REGISTRATION.md`](NEEDLE_POSE_REGISTRATION.md) — v2/v3 的模型配准与加速细节。
- 📦 上传清单 [`NECESSARY_FILES.md`](NECESSARY_FILES.md) — 哪些文件需要上传 / 哪些保持私有。

## 环境配置与安装

推荐 conda + Python 3.10、CUDA 12.1、PyTorch 2.5.1。

```bash
# 1. 创建环境
conda create -n unimatchv2 python=3.10 -y
conda activate unimatchv2

# 2. 安装 PyTorch（按你的 CUDA 选版本，下面是 cu121）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装推理依赖
pip install -r docs/requirements-inference.txt

# 4.（可选）TensorRT 加速：匹配 torch/CUDA 的 torch-tensorrt
pip install torch-tensorrt        # 缺它时导出会自动退回 TorchScript，不影响功能
```

预训练骨干：`pretrained/dinov2_vitb14_pretrain.pth`（DINOv2-base，配置中 `backbone_checkpoint` 指向它）。
仅做**推理**时不需要训练，但仍需该骨干结构对应的权重已在 `CKPT` 内（best.pth 已包含完整模型）。

> 标注器 `SAM2-Plus/demo/app_gui.py` 另需 SAM-2 环境与其 checkpoint，且 `transformers<4.49`；详见 SAM2-Plus 自带说明。

## 操作文件位置一览

| 用途 | 文件 |
|---|---|
| **推理 · v1**（自由半径） | `realtime_stereo_keypoints.py`、`eval_pose_val.py`、`SAM2-Plus/tools/needle_keypoints.py` |
| **推理 · v2**（模型配准） | `realtime_stereo_keypoints_v2.py`、`eval_pose_val_v2.py`、`SAM2-Plus/tools/needle_keypoints_v2.py` |
| **推理 · v3**（v2+加速） | `realtime_stereo_keypoints_v3_accel.py`、`eval_pose_val_v3_accel.py`、`infer_accel.py` |
| 引擎导出（TensorRT/TS） | `export_seg_engine.py` |
| **封装部署**（不含模型源码） | `infer_engine_only.py`（只加载引擎，无 `test.py`/`model/` 依赖） |
| 针半径标定 | `SAM2-Plus/tools/calibrate_needle_radius.py` → `needle_model.json` |
| 双目标定文件 | `SAM2-Plus/tools/needle_calib.json` |
| 训练（全监督 / 半监督） | `train_supervised_basic.py` / `unimatch_v2.py` |
| 分割指标评测 | `test.py` |
| 数据集合并+划分 | `SAM2-Plus/tools/build_combined_trainset.py` |
| 右目预测+校正 | `build_stereo_id_path.py`、`prep_right_for_annotation.py` |
| 交互标注器 | `SAM2-Plus/demo/app_gui.py`（+ `app_gui_stereo.py` / `app_gui_dual.py`） |
| 模型配置 | `configs/surgical_combined_base.yaml` |
| 权重 | `exp/combined_r100_base/best.pth` |

## 快速上手（验证集，加速版）

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate unimatchv2
cd /root/autodl-tmp/code/UniMatch-V2_local
python eval_pose_val_v3_accel.py \
  --config configs/surgical_combined_base.yaml \
  --checkpoint exp/combined_r100_base/best.pth \
  --calib /root/autodl-tmp/code/SAM2-Plus/tools/needle_calib.json \
  --needle-model /root/autodl-tmp/code/SAM2-Plus/tools/needle_model.json \
  --root /root/autodl-tmp/data/surgical_seg \
  --val-split /root/autodl-tmp/data/surgical_seg/combined/splits/r100/val.txt \
  --out-dir exp/pose_val_v3 --seg-size 512 --num-keypoints 5 --compile --no-video
```
完整用法见 [`OPERATION_MANUAL.md`](OPERATION_MANUAL.md)。
