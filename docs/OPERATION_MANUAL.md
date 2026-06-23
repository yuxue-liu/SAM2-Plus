# 缝针分割与 6-DoF 位姿估计 

- **第一部分 · 推理使用**：三个版本的推理代码，以及验证集 / 视频流 / 双目采集卡三种输入的参数替换方法。
- **第二部分 · 全流程**：数据集构建与标注 → 训练 → 推理 → 校正 → 数据集扩充，每一步用到的文件、指令与可替换变量。
- **第三部分 · 方法简介**：标注器、分割模型、关键点与位姿模块。

> 环境配置、安装、文件位置见 [`README.md`](README.md)。位姿配准方法的算法细节见 [`NEEDLE_POSE_REGISTRATION.md`](NEEDLE_POSE_REGISTRATION.md)。

约定：下文 `<尖括号>` 为需要替换的变量；命令默认在服务器上、已 `conda activate unimatchv2`、工作目录 `UniMatch-V2_local/`。

---

## 公共变量

环境变量，后面命令直接引用：

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate unimatchv2
cd /root/autodl-tmp/code/UniMatch-V2_local

CFG=configs/surgical_combined_base.yaml                              # 模型配置（DINOv2-base + DPT, 4 类）
CKPT=exp/combined_r100_base/best.pth                                 # 训练好的权重
CALIB=/root/autodl-tmp/code/SAM2-Plus/tools/needle_calib.json        # 双目标定
MODEL=/root/autodl-tmp/code/SAM2-Plus/tools/needle_model.json        # 针的规范半径（v2/v3 需要）
ROOT=/root/autodl-tmp/data/surgical_seg                              # 统一数据根目录
```

| 变量 | 含义 | 何时替换 |
|---|---|---|
| `CFG` | 分割模型配置 yaml | 换骨干/类别数时 |
| `CKPT` | 模型权重 | 换训练结果时 |
| `CALIB` | 双目内外参标定 json | 换相机/采集卡时**必须重标定** |
| `MODEL` | `{"radius_mm": ...}` | 换针型号时重标定（见第二部分） |
| `ROOT` | 数据根目录 | 换数据存放位置时 |

---

# 第一部分 · 推理使用

## 1.1 三个版本怎么选

| 版本 | 位姿原理                 | 主要文件 | 何时用 |
|---|----------------------|---|---|
| **v1** | 自由半径圆弧拟合（每帧重估半径）     | `realtime_stereo_keypoints.py`、`eval_pose_val.py`、`tools/needle_keypoints.py`(SAM2-Plus) | 基线 / 对照 |
| **v2** | 已知半径的模型配准（遮挡下更准）     | `realtime_stereo_keypoints_v2.py`、`eval_pose_val_v2.py`、`needle_keypoints_v2.py` | **推荐的精度版** |
| **v3** | = v2 方法 + 推理加速（精度不变） | `realtime_stereo_keypoints_v3_accel.py`、`eval_pose_val_v3_accel.py`、`infer_accel.py`、`export_seg_engine.py` | **推荐的速度版** |

> 三个版本互不覆盖，可逐版本对比。v2/v3 比 v1 多需要一个 `MODEL`（针半径），用 `SAM2-Plus/tools/calibrate_needle_radius.py` 标定一次（见第二部分 §2.4）。

三个版本的**输入方式与参数完全一致**，只是脚本名不同。下面以 v3（最常用）举例，换成 v2/v1 只需换脚本名并去掉加速开关。

## 1.2 三种输入源的参数替换

输入源由一组互斥参数决定（三选一）：

| 输入源 | 选择参数 | 说明 |
|---|---|---|
| **验证集（已存的双目序列回放）** | `--root <ROOT> --dataset <数据集> --key <视频key>` | 单视频回放，有 GT 可出指标 |
| **验证集（整个 val 划分批量评测）** | 用 `eval_pose_val_*` + `--val-split <val.txt>` | 汇总 PCK / 3D 误差 |
| **视频流（两路立体视频/相机）** | `--left <左.mp4> --right <右.mp4>` | 也可填相机索引 `--left 0 --right 1` |
| **双目采集卡（单设备含左右目）** | `--capture <设备号> --layout sbs\|tb` | sbs=左右拼接，tb=上下拼接，自动切分 L/R |

### A. 验证集 · 单视频可视化推理

```bash
python realtime_stereo_keypoints_v3_accel.py --config $CFG --checkpoint $CKPT \
  --calib $CALIB --needle-model $MODEL \
  --root $ROOT --dataset march_1 --key 1_01 \              # ← 替换数据集/视频
  --num-keypoints 5 --seg-size 640 --device cuda:0 \
  --gt-subdir keypoints --pck-thresh 10 \                  # 有 GT 时出指标；无 GT 删这两行
  --save-video out.mp4 --save-results out.jsonl --save-poses out.csv
```
- 换视频：改 `--dataset` / `--key`。
- 不要指标：删 `--gt-subdir/--pck-thresh`。
- 实时窗口预览：删三个 `--save-*`，加 `--show`（需要 DISPLAY）。

### B. 验证集 · 整个 val 划分批量评测

```bash
python eval_pose_val_v3_accel.py --config $CFG --checkpoint $CKPT \
  --calib $CALIB --needle-model $MODEL --root $ROOT \
  --val-split $ROOT/combined/splits/r100/val.txt \         # ← 替换划分（r100/r50/r30/r10）
  --out-dir exp/pose_val_v3 --seg-size 512 --num-keypoints 5 \
  --no-video                                               # 只要数字最快；要每个 key 的 mp4 就删掉
```
- 换比例划分：改 `--val-split` 里的 `r100`。
- 指定具体数据集而非 val 划分：把 `--val-split ...` 换成 `--datasets march_1 march_2`。
- 结果汇总在 `exp/pose_val_v3/summary.json`（PCK、各点 px/mm 误差、fps、峰值显存）。

### C. 视频流（两路立体视频文件 / 相机）

```bash
python realtime_stereo_keypoints_v3_accel.py --config $CFG --checkpoint $CKPT \
  --calib $CALIB --needle-model $MODEL \
  --left <左.mp4> --right <右.mp4> \                        # ← 替换为的两路源（或相机索引 0 1）
  --num-keypoints 5 --seg-size 640 --device cuda:0 \
  --save-video out_stream.mp4 --save-results out_stream.jsonl
```
- 视频流没有 GT，不出指标，只产生可视化 + 结果文件。
- **注意**：单个"左右拼接"的视频**文件**不被直接支持（`--capture` 只接设备号）；请提供两路独立文件，或用采集卡模式。

### D. 双目采集卡

先探测采集卡分辨率：
```bash
python -c "import cv2;c=cv2.VideoCapture(0);print(int(c.get(3)),'x',int(c.get(4)));c.release()"
```
```bash
python realtime_stereo_keypoints_v3_accel.py --config $CFG --checkpoint $CKPT \
  --calib $CALIB --needle-model $MODEL \
  --capture 0 --layout sbs \                               # ← 设备号；sbs 左右 / tb 上下
  --num-keypoints 5 --seg-size 640 --device cuda:0 \
  --show --save-video out_card.mp4
```
- `sbs`：每只眼宽 = 设备宽 / 2；`tb`：每只眼高 = 设备高 / 2。
- 采集卡/视频流换相机后，`CALIB` 标定必须对应该相机，否则 3D/位姿不准。

## 1.3 如何开到最快（推理加速）

加速三层，可叠加，**只影响速度不影响数值**（降 seg-size 除外）：

| 开关（v3 专有） | 作用 | 依赖 |
|---|---|---|
| `--seg-engine <engine.ts>` | 用预导出的 TensorRT/TorchScript 引擎，前向最快(2–4×) | 需先导出引擎 |
| `--compile`(+`--channels-last`) | torch.compile 融合前向(~1.2–1.6×) | 无 |
| 异步流水线（默认开） | 读帧/编码与 GPU 重叠 | 无，`--no-async` 关 |
| `--no-video`（仅 eval） | 跳过画面绘制+编码 | 无 |

**最快配置 = 导出 TensorRT 引擎 + `--seg-engine` + (评测加 `--no-video`)**。引擎按"每眼分辨率 + seg-size"定死，要与运行时一致；本项目数据均为 1920×1080：

```bash
# 一次性导出引擎（在目标 GPU 上）：realtime 用 640、eval 用 512
python export_seg_engine.py --config $CFG --checkpoint $CKPT --format tensorrt \
  --src-h 1080 --src-w 1920 --seg-size 640 --out exp/combined_r100_base/seg_trt_s640.ts
python export_seg_engine.py --config $CFG --checkpoint $CKPT --format tensorrt \
  --src-h 1080 --src-w 1920 --seg-size 512 --out exp/combined_r100_base/seg_trt_s512.ts
```
运行时加 `--seg-engine exp/combined_r100_base/seg_trt_s640.ts`（realtime）或 `..._s512.ts --no-video`（eval）。
未装 `torch-tensorrt` 时导出自动退回 TorchScript；也可直接用零依赖的 `--compile`。生效时日志会打印 `[accel] segmentation backend = pre-exported engine: ...`。

## 1.4 推理产生的文件与顺序

**单视频脚本** `realtime_stereo_keypoints_v*`（逐帧流式写，结束统一收尾）：
1. `--save-video` → `*.mp4`：左|右叠加可视化（关键点/位姿轴/重投影）。
2. `--save-results` → `*.jsonl`：每帧一行，最全的结构化结果（5 个关键点的左右 2D + `xyz_mm` + `visible`、`pose`(R/t/rvec)、`conf`；无检测则 `needle:null`）。
3. `--save-poses` → `*.csv`：扁平数值表（`frame, kp0x..kp4z, tx,ty,tz, rvx,rvy,rvz, R00..R22, eul_xyz`），仅有检测的帧。

**批量评测脚本** `eval_pose_val_v*`：
1. 每个 key 一个 `<dataset>__<key>.mp4`（`--no-video` 时不产生）。
2. 所有视频跑完后写一次 `summary.json`（总指标）。

---

# 第二部分 · 全流程（构建标注 → 训练 → 推理 → 校正 → 扩充）

数据组织（统一根 `ROOT`，每个视频一个子目录）：
```
surgical_seg/
  <dataset>/  (如 march_1, july_1)
     meta.json                          每帧记录 {image, mask, ordinal}
     images/<key>/part_xxx/<stem>.jpg   左目帧
     masks /<key>/part_xxx/<stem>.png   左目分割（标注/预测，索引PNG）
     stereo_right/<key>/<stem>.jpg      右目帧
     keypoints/<key>/.../<stem>.json    关键点+位姿 sidecar（自动生成）
  combined/splits/r{100,50,30,10}/{labeled,unlabeled,val}.txt   训练/验证划分
```

## 2.1 数据集构建与标注

| 步骤 | 文件 | 指令（替换 `<...>`） |
|---|---|---|
| ① 抽帧（单目/双目） | `SAM2-Plus/tools/extract_frames.py` / `extract_stereo_frames.py` | `python tools/extract_frames.py --video <raw.mp4> --out <ROOT>/<dataset> ...` |
| ② 交互标注左目 | `SAM2-Plus/demo/app_gui.py` | `cd demo && python app_gui.py --image_dir <ROOT>/<dataset>/images/<key>/part_000` |
| ③ 合并为统一训练集 + 划分 | `SAM2-Plus/tools/build_combined_trainset.py` | 见下 |

标注器（②）：点/笔刷多类标注 + SAM2 视频传播 + 暂停-续传，自动把索引 PNG 存到 `masks/`。每个 part 文件夹 ≤200 帧。

合并 + 划分（③）：
```bash
python SAM2-Plus/tools/build_combined_trainset.py \
  --root $ROOT \
  --datasets march_1 july_1 <新增数据集> \                 # ← 列出要纳入的视频
  --test-ratio 0.2 --ratios 1.0 0.5 0.3 0.1
# 产出 $ROOT/combined/splits/r{100,50,30,10}/{labeled,unlabeled,val}.txt
```
- 每个视频按 ordinal 排序，**最后 20% 作 val**（无时间泄漏）；其余按比例 r 均匀抽为有标签，其余为无标签（半监督用）。

## 2.2 训练

| 方式 | 文件 | 何时用 |
|---|---|---|
| **全监督基线** | `train_supervised_basic.py` | 用 r100 的全部标签 |
| 半监督（UniMatch-V2） | `unimatch_v2.py` | 标签少时(r10/r30/r50)，用 labeled+unlabeled |

全监督（最常用）：
```bash
python train_supervised_basic.py --config $CFG \
  --labeled-id-path $ROOT/combined/splits/r100/labeled.txt \
  --val-id-path     $ROOT/combined/splits/r100/val.txt \
  --save-path exp/combined_r100_base                       # ← 输出权重目录，得到 best.pth
```
- 换标签比例：把 `r100` 改成 `r50/r30/r10`，并把 `--save-path` 改成对应名字。
- 配置里可改 `epochs / batch_size / lr / backbone`（`CFG` 文件）。

## 2.3 推理（左目分割 / 指标）

- **纯分割指标**：`python test.py --config $CFG --checkpoint $CKPT --id-path <val.txt> --save-csv <out.csv>`
- **关键点+位姿**：见第一部分（realtime / eval 脚本）。

## 2.4 校正（右目分割 → 关键点 → 针半径标定）

立体三角化需要右目分割。流程：

```bash
# ① 给右目帧生成 id-path
python SAM2-Plus/tools/build_stereo_id_path.py --root $ROOT --dataset march_1 --key 1_01 \
  --out $ROOT/march_1/stereo_right_ids.txt
# ② 用模型预测右目 mask
python test.py --config $CFG --checkpoint $CKPT \
  --id-path $ROOT/march_1/stereo_right_ids.txt --no-bias \
  --save-preds /root/autodl-tmp/exp/right_pred/march_1
# ③ 把"右目帧+预测mask"摆进标注工作区，供人工校正
python SAM2-Plus/tools/prep_right_for_annotation.py --root $ROOT --dataset march_1 --key 1_01 \
  --right-pred-dir /root/autodl-tmp/exp/right_pred/march_1
#    然后用 app_gui.py 打开 right_annot/ 校正
# ④ 由左右目 mask 生成关键点+位姿 sidecar（v1 几何法，写 keypoints/）
python SAM2-Plus/tools/needle_keypoints.py --root $ROOT --dataset march_1 --key 1_01 \
  --right-pred-dir <校正后的右目mask目录> --calib $CALIB --num-keypoints 5
# ⑤ 由 sidecar 标定针的规范半径 -> needle_model.json（v2/v3 用）
python SAM2-Plus/tools/calibrate_needle_radius.py --root $ROOT --datasets march_1 \
  --min-conf 0.0 --rmin 2 --rmax 20 --out SAM2-Plus/tools/needle_model.json
```
替换：`--dataset/--key` 换视频；`--datasets` 可列多个一起标定半径。

## 2.5 数据集扩充（迭代）

1. 新视频走 §2.1 ①②（可先用模型预测当预标注，再人工修，省时）。
2. 把新数据集名加入 §2.1 ③ 的 `--datasets`，重新生成划分。
3. 回到 §2.2 重新训练（或微调），得到新 `best.pth`。
4. 需要立体/位姿时再走 §2.4。

---

# 第三部分 · 方法简介

## 3.1 标注器（SAM2-Plus 交互式视频标注）
基于 SAM2 的多类视频标注工具（`demo/app_gui.py`）：点/笔刷给每个类别打正负点，SAM 实时分割；**前向视频传播**自动标注后续帧，可暂停修正后再续传；逐帧自动保存为单通道索引 PNG（与 UniMatch-V2 约定一致）。还有立体/双目校正版 `app_gui_stereo.py` / `app_gui_dual.py`。

## 3.2 分割模型（DINOv2 + DPT）
- 骨干 **DINOv2-base**（ViT，注意力走 PyTorch SDPA 融合核），解码头 **DPT**，输出 **4 类**：背景 / 针(needle) / 线(thread) / 持针钳(clamps)。
- 两种训练：**全监督** `train_supervised_basic.py`（零改进基线）；**半监督** UniMatch-V2 `unimatch_v2.py`（弱强一致性，利用无标签帧）。
- 推理：fp16，双眼一次前向(batch=2)，logits 在模型分辨率取 argmax 后最近邻上采样标签（省算力）。

## 3.3 关键点与位姿模块
- **2D 中心线**：取针 mask 的最大连通域，骨架化/椭圆弧拟合得到有序中心线（tip→tail）；遮挡断开时用椭圆弧跨缺口补全。
- **tip/tail 定向**：靠近缝线(thread)的一端为针尾(tail)，另一端为针尖(tip)；无线时用"针尾更粗"判断。
- **立体 3D**：左右中心线按弧长比例对应 → 三角化出 3D 点 → 拟合 3D 圆（平面 + 圆），等弧长采样 N 个关键点，反投影回两眼判可见性。
- **6-DoF 位姿**：以圆心为原点、平面法向为 z 轴、指向针尖方向为 x 轴构造 `(R, t, rvec)`。
- **版本差异**：v1 每帧自由估半径；**v2 固定已知半径做配准**（更稳，遮挡/左右不对称下更准）；v3 = v2 + 推理加速。算法细节见 [`NEEDLE_POSE_REGISTRATION.md`](NEEDLE_POSE_REGISTRATION.md)。
