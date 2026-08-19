# DIT Codec Distillation

一个契约驱动的 PyTorch 编解码器蒸馏工程，面向以下场景：

主要训练 FlashVSR `LQ_proj_in` 和条件 `TCDecoder` 时，请直接阅读 [FlashVSR 蒸馏教程](FLASHVSR_DISTILL_TUTORIAL.md)。教程从进入 Linux 项目目录开始，逐条给出命令、预期输出、YAML 参数、checkpoint、日志和恢复训练方法。

- 学生编码器输入 `[B,6,H/2,W/2]` 的 `Y00,Y01,Y10,Y11,U,V` 打包数据，输出 `[B,16,H/8,W/8]` latent；
- 学生解码器输入 latent，输出 U/V 仅在每个 `2x2` 左上角有效的 sparse YUV；
- 使用 Wan VAE 或 FlashVSR 组件作为教师；
- 学生模型是可训练但源码不可见的 PyTorch `nn.Module`；
- 不训练 DiT，且允许编码器、解码器分开蒸馏。

第一版提供完整 mock 模型。没有真实权重时也能运行数据读取、forward、loss、backward、验证、checkpoint 和 resume。

## 安装

需要 Python 3.10+ 和与你的 CUDA 环境匹配的 PyTorch。Ubuntu 推荐先创建独立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train,test]"
```

如果只使用 JSONL 日志，可以不安装 `train` extras。真实 Wan/FlashVSR 源码还依赖 `einops` 和 `tqdm`，已列入基础依赖。

## 最快运行

```bash
distill-codec make-mock-data --output work/mock_data --count 8 --size 64

distill-codec probe \
  --config configs/smoke/wan_encoder.yaml \
  --set "data.lq_root=work/mock_data/lq" \
  --set "data.gt_root=work/mock_data/gt"

distill-codec train \
  --config configs/smoke/wan_encoder.yaml \
  --set "data.lq_root=work/mock_data/lq" \
  --set "data.gt_root=work/mock_data/gt" \
  --set "run.output_dir=runs/wan_encoder"
```

Ubuntu 完整 CPU smoke：

```bash
PYTHON_BIN="$(command -v python)" ./scripts/run_smoke.sh work/smoke
```

Windows PowerShell 可使用现有入口：

```powershell
.\scripts\run_smoke.ps1 -PythonExe ".\.venv\Scripts\python.exe" -OutputRoot "work/smoke"
```

Smoke 会运行七个 Recipe：

- `wan_encoder_distill`
- `wan_decoder_distill`
- `wan_autoencoder_distill`
- `flashvsr_vae_encoder_distill`
- `flashvsr_lq_proj_distill`
- `flashvsr_decoder_unconditional_student`
- `flashvsr_decoder_conditional_student`

Mock smoke 验证工程链路，不代表真实蒸馏质量或默认 loss 权重已经最优。

## 训练启动手册

下面的流程是从零开始启动真实蒸馏时建议执行的顺序。训练器只读取单帧 LQ/GT 配对图像；如果教师需要视频帧，Adapter 会把同一张图像按配置重复到教师需要的帧数。DiT 不会被构造或训练。

### 1. 安装环境

在仓库根目录创建一个使用正确 PyTorch/CUDA 版本的环境，然后安装项目：

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip curl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[train,test]"
```

`cu124` 只是示例，必须替换成与你的驱动和 CUDA 版本匹配的 PyTorch 安装命令。CPU 验证可以使用 CPU 版本的 PyTorch；正式训练建议在 `nvidia-smi` 能看到目标 GPU 后再启动。

### 2. 先跑 mock 链路

先用 mock 模型确认 Python、数据读取、loss、反向传播、验证图和 resume 都正常：

```bash
python -m distill_codec.cli make-mock-data \
  --output work/mock_data \
  --count 8 \
  --size 64

python -m distill_codec.cli probe \
  --config configs/smoke/wan_encoder.yaml \
  --set "data.lq_root=work/mock_data/lq" \
  --set "data.gt_root=work/mock_data/gt"

PYTHON_BIN="$(command -v python)" ./scripts/run_smoke.sh work/smoke
```

每个 smoke recipe 都会先训练到 step 1，再从 step 1 checkpoint 恢复到 step 2。输出位于 `work/smoke/runs/<recipe>/`，包括 `metrics.jsonl`、`checkpoints/`、`validation/` 和可选的 `tensorboard/`。

### 3. 准备真实数据

LQ 和 GT 是 RGB 图片，不需要你提前生成 YUV。工程内部会执行：

```text
RGB [B,3,H,W]
  -> packed YUV422 [B,6,H/2,W/2]
     (Y00,Y01,Y10,Y11,U,V)
  -> student encoder latent [B,16,H/8,W/8]
```

目录必须一一对应，相对路径和文件名完全相同：

```text
$HOME/dit_codec/LQ/scene_01/000001.png
$HOME/dit_codec/GT/scene_01/000001.png
```

先用现有 Mock 配置做数据预检和一次前向。Mock 配置的 `probe` 不需要真实教师权重，可以提前发现配对、解码和尺寸问题：

```bash
python -m distill_codec.cli probe \
  --config configs/smoke/wan_encoder.yaml \
  --set "data.lq_root=$HOME/dit_codec/LQ" \
  --set "data.gt_root=$HOME/dit_codec/GT" \
  --set "data.lq_size=[256,256]" \
  --set "data.gt_size=[256,256]" \
  --set "trainer.device=cpu"
```

真实配置的 `probe` 会先构造教师和学生组件，再读取数据并执行一次前向。因此它需要有效的教师源码、checkpoint 和配置指定的 CUDA 设备；它不是一个绕过模型加载的纯数据扫描命令。

### 4. 配置真实教师和黑盒学生

先创建用于存放本机配置的目录：

```bash
mkdir -p configs/local
```

下面的 YAML 分别写入对应的 `configs/local/*.yaml`。`configs/local/` 不会被现有 Git 规则自动忽略；如果配置中含私有路径或包名，请不要提交这些本地文件。

#### `configs/local/wan_encoder.yaml`

Wan VAE 编码器蒸馏的完整主配置如下：

```yaml
includes:
  - ../teachers/wan_snapshot.yaml
  - ../students/private_blackbox.yaml

latent_spec:
  family: wan_vae_v2
  channels: 16
  layout: BCHW
  spatial_downsample: 8
  temporal_downsample: 1
  normalization: wan_vae

color:
  matrix: bt709
  range: full
  packed_order: Y00Y01Y10Y11UV
  chroma_location: top_left
  chroma_upsample: nearest

recipe:
  name: wan_encoder_distill
  source: gt
  weights: {latent: 1.0, cos: 0.1, stat: 0.1, compat: 0.1}

data:
  lq_root: ~/dit_codec/LQ
  gt_root: ~/dit_codec/GT
  lq_size: [256, 256]
  gt_size: [256, 256]

trainer:
  device: cuda
  batch_size: 2
  learning_rate: 0.0001
  max_steps: 100000
  validate_every: 1000
  checkpoint_every: 1000
  tensorboard: true
  amp: true

run:
  output_dir: ~/dit_codec/runs/wan_encoder
  seed: 7
```

`configs/teachers/wan_snapshot.yaml` 和 `configs/students/private_blackbox.yaml` 中的本地路径、factory 和 checkpoint 必须改成你的实际路径。真实黑盒 Encoder 的 factory 需要返回 `nn.Module`，并满足 `[B,6,H/2,W/2] -> [B,16,H/8,W/8]`；真实黑盒 Decoder 需要满足 latent -> sparse YUV `[B,3,H,W]`。

#### `configs/local/flashvsr_lq_proj.yaml`

这份配置蒸馏 FlashVSR `LQ_proj_in`。`student_condition_encoder` 必须直接声明，因为 [configs/students/private_blackbox.yaml](configs/students/private_blackbox.yaml) 只提供主 Encoder、Decoder 和条件 Decoder：

```yaml
includes:
  - ../teachers/flashvsr_snapshot.yaml

latent_spec:
  family: wan_vae_v2
  channels: 16
  layout: BCHW
  spatial_downsample: 8
  temporal_downsample: 1
  normalization: wan_vae

color:
  matrix: bt709
  range: full
  packed_order: Y00Y01Y10Y11UV
  chroma_location: top_left
  chroma_upsample: nearest

recipe:
  name: flashvsr_lq_proj_distill
  weights: {condition: 1.0}

components:
  student_condition_encoder:
    backend: external
    factory: my_encrypted_package.models:create_condition_encoder
    kwargs: {}
    adapter:
      kind: condition_encoder
      temporal_frames: 5
      condition_spec:
        family: flashvsr_lq_proj_v1_1
        layout: BNC
        feature_dim: 1536
        source: lq
        consumer: dit
        spatial_downsample: 16
        temporal_downsample: 5

data:
  lq_root: ~/dit_codec/LQ
  gt_root: ~/dit_codec/GT
  lq_size: [256, 256]
  gt_size: [256, 256]

trainer:
  device: cuda
  batch_size: 2
  learning_rate: 0.0001
  max_steps: 100000
  validate_every: 1000
  checkpoint_every: 1000
  tensorboard: true
  amp: true

run:
  output_dir: ~/dit_codec/runs/flashvsr_lq_proj
  seed: 7
```

#### `configs/local/flashvsr_tcdecoder.yaml`

这份配置蒸馏可直接替换 FlashVSR `TCDecoder` 的条件学生。Wan Encoder 只作为冻结的 latent provider，不训练 DiT：

```yaml
includes:
  - ../teachers/wan_snapshot.yaml
  - ../teachers/flashvsr_snapshot.yaml
  - ../students/private_blackbox.yaml

latent_spec:
  family: wan_vae_v2
  channels: 16
  layout: BCHW
  spatial_downsample: 8
  temporal_downsample: 1
  normalization: wan_vae

color:
  matrix: bt709
  range: full
  packed_order: Y00Y01Y10Y11UV
  chroma_location: top_left
  chroma_upsample: nearest

recipe:
  name: flashvsr_decoder_conditional_student
  weights: {teacher: 1.0, gt: 0.5, edge: 0.1}

components:
  conditional_student_decoder:
    factory: my_encrypted_package.models:create_conditional_decoder
    checkpoint: null

latent_provider:
  type: teacher_encoder
  source: gt

data:
  lq_root: ~/dit_codec/LQ
  gt_root: ~/dit_codec/GT
  lq_size: [256, 256]
  gt_size: [256, 256]

trainer:
  device: cuda
  batch_size: 1
  learning_rate: 0.0001
  max_steps: 100000
  validate_every: 1000
  checkpoint_every: 1000
  tensorboard: true
  amp: true

run:
  output_dir: ~/dit_codec/runs/flashvsr_tcdecoder
  seed: 7
```

这里把 `conditional_student_decoder.checkpoint` 覆盖为 `null`，表示从 factory 返回的初始状态开始训练。如果你有学生预训练权重，把它改成实际 `.pth` 路径。两个 snapshot 教师配置中的源码和权重路径也必须与本机一致。

### 5. 按目标选择 recipe

| 目标 | recipe | 需要的教师组件 | 学生组件 |
| --- | --- | --- | --- |
| Wan 主 latent 编码器 | `wan_encoder_distill` | Wan `teacher_encoder`、`teacher_decoder` | `student_encoder` |
| Wan latent 解码器 | `wan_decoder_distill` | Wan `teacher_encoder`、`teacher_decoder` | `student_decoder` |
| Wan 编解码器一起训练 | `wan_autoencoder_distill` | Wan `teacher_encoder`、`teacher_decoder` | `student_encoder`、`student_decoder` |
| Wan VAE 主 latent 编码器（独立 DiT 案例） | `flashvsr_vae_encoder_distill` | Wan `teacher_encoder`、`teacher_decoder` | `student_encoder` |
| FlashVSR `LQ_proj_in` 可替换条件编码器 | `flashvsr_lq_proj_distill` | FlashVSR `teacher_condition_encoder` | `student_condition_encoder` |
| FlashVSR 无条件输出蒸馏 | `flashvsr_decoder_unconditional_student` | `teacher_encoder`、`tc_decoder` | `student_decoder` |
| FlashVSR `TCDecoder` 可替换条件解码器 | `flashvsr_decoder_conditional_student` | `teacher_encoder`、`tc_decoder` | `conditional_student_decoder` |

`LQ_proj_in` 是 DiT 条件编码器，不是 Wan VAE 主 latent 编码器。训练它时，学生组件必须使用 `adapter.kind: condition_encoder`，并声明与教师一致的 `ConditionSpec`（FlashVSR v1.1 默认 `feature_dim: 1536`、`layout: BNC`）。

如果目标是在 FlashVSR 架构中直接替换教师模块，默认蒸馏链路是：

```text
LQ_proj_in -> student_condition_encoder
TCDecoder  -> conditional_student_decoder
```

这两个 Recipe 可以分开训练，不需要训练或运行 DiT。`flashvsr_decoder_unconditional_student` 只蒸馏教师输出，学生没有 LQ 条件输入，因此不是可直接替换 TCDecoder 的等价接口。Wan VAE Recipe 仍用于主 latent 编解码器实验，但不属于 FlashVSR 的 `LQ_proj_in` 条件分支。

### 6. 启动训练和恢复

以真实 Wan Encoder 为例：

```bash
python -m distill_codec.cli train \
  --config configs/local/wan_encoder.yaml \
  --set "data.lq_root=$HOME/dit_codec/LQ" \
  --set "data.gt_root=$HOME/dit_codec/GT" \
  --set "run.output_dir=$HOME/dit_codec/runs/wan_encoder"
```

中断后从 checkpoint 继续：

```bash
python -m distill_codec.cli train \
  --config configs/local/wan_encoder.yaml \
  --resume "$HOME/dit_codec/runs/wan_encoder/checkpoints/step_00001000.pt" \
  --set "trainer.max_steps=100000"
```

真实 FlashVSR `LQ_proj_in` 先检查配置和一次前向，再启动训练：

```bash
python -m distill_codec.cli probe \
  --config configs/local/flashvsr_lq_proj.yaml

python -m distill_codec.cli train \
  --config configs/local/flashvsr_lq_proj.yaml
```

真实 FlashVSR 条件 `TCDecoder` 使用另一份主配置：

```bash
python -m distill_codec.cli probe \
  --config configs/local/flashvsr_tcdecoder.yaml

python -m distill_codec.cli train \
  --config configs/local/flashvsr_tcdecoder.yaml
```

恢复时不要随意修改 batch size、梯度累积、seed、数据尺寸、latent/condition shape、optimizer 或 scheduler 周期。确实要修改非 shape 契约时，显式设置 `trainer.allow_contract_override: true`，不要用它绕过通道数、layout 或空间/时间下采样检查。

### 7. 权重和输出位置

仓库不包含任何真实模型权重，也不会自动把权重上传到 Git。配置模板使用以下本地路径；本次已从 FlashVSR v1.1 发布目录下载并核验前三个教师文件：

```text
$HOME/dit_codec/weights/Wan2.1_VAE.pth       # Wan VAE Encoder + Decoder，共用一份
$HOME/dit_codec/weights/LQ_proj_in.ckpt      # FlashVSR LQ_proj_in
$HOME/dit_codec/weights/TCDecoder.ckpt       # FlashVSR TCDecoder
$HOME/dit_codec/weights/student_encoder.pth  # 你的黑盒学生 Encoder（可选）
$HOME/dit_codec/weights/student_decoder.pth  # 你的黑盒学生 Decoder（可选）
```

官方来源：[`JunhaoZhuang/FlashVSR-v1.1`](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1)。本次下载使用可访问的 Hugging Face 镜像，文件 SHA-256 与官方 LFS 元数据一致：

Ubuntu 下载命令（支持重试和断点续传）：

```bash
WEIGHT_DIR="${WEIGHT_DIR:-$HOME/dit_codec/weights}"
DOWNLOAD_BASE="https://hf-mirror.com/JunhaoZhuang/FlashVSR-v1.1/resolve/main"
# 可以直连 Hugging Face 时改用官方地址：
# DOWNLOAD_BASE="https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/resolve/main"

mkdir -p "$WEIGHT_DIR"
for name in Wan2.1_VAE.pth LQ_proj_in.ckpt TCDecoder.ckpt; do
  curl --fail --location --retry 5 --retry-delay 3 \
    --continue-at - \
    --output "$WEIGHT_DIR/$name" \
    "$DOWNLOAD_BASE/$name"
done
```

下载完成后执行 SHA-256 校验；任一文件不匹配都会停止并报错：

```bash
printf '%s\n' \
  '38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981  Wan2.1_VAE.pth' \
  'd6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2  LQ_proj_in.ckpt' \
  'e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161  TCDecoder.ckpt' \
  | tee "$WEIGHT_DIR/SHA256SUMS" >/dev/null

(cd "$WEIGHT_DIR" && sha256sum -c SHA256SUMS)
```

| 文件 | 字节数 | MiB | SHA-256 |
| --- | ---: | ---: | --- |
| `Wan2.1_VAE.pth` | `507,609,880` | `484.095` | `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` |
| `LQ_proj_in.ckpt` | `575,694,948` | `549.025` | `d6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2` |
| `TCDecoder.ckpt` | `189,018,333` | `180.262` | `e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161` |

三份教师权重合计 `1,272,323,161 bytes`，约 `1.272 GB / 1.185 GiB`。当前 Linux 配置模板默认从 `~/dit_codec/weights` 加载它们；如果你的挂载点不同，请同步修改 YAML。把文件放好后也可以用下面的命令重新统计：

```bash
du -ch "$WEIGHT_DIR"/Wan2.1_VAE.pth \
  "$WEIGHT_DIR"/LQ_proj_in.ckpt \
  "$WEIGHT_DIR"/TCDecoder.ckpt
stat -c '%n %s bytes' "$WEIGHT_DIR"/Wan2.1_VAE.pth \
  "$WEIGHT_DIR"/LQ_proj_in.ckpt \
  "$WEIGHT_DIR"/TCDecoder.ckpt
```

本次 mock smoke 产生的学生 checkpoint 只用于验证工程链路，位于：

```text
work/live_run_20260819/smoke/runs/
```

共 14 个 `.pt`，总计 `4,496,814 bytes`，约 `4.29 MiB`（`0.00419 GiB`），不能替代真实教师权重或最终蒸馏权重。训练输出目录中的 checkpoint 只保存学生和优化器状态，不会复制教师权重。

### 8. 训练完成检查

每个运行目录至少应出现以下文件：

```text
metrics.jsonl
checkpoints/step_XXXXXXXX.pt
validation/step_XXXXXXXX.png
tensorboard/events.out.tfevents.*    # trainer.tensorboard=true 时
```

重点检查：`metrics.jsonl` 中的 loss 是有限值，`validation/` 图片尺寸正确，resume 后 `global_step` 连续递增，并且 decoder 指标同时记录 `rgb_mae_vs_teacher` 与 `rgb_mae_vs_gt`。

## 数据目录

LQ 和 GT 必须有完全相同的相对目录和文件名：

```text
$HOME/dit_codec/LQ/scene_01/000001.png
$HOME/dit_codec/GT/scene_01/000001.png
```

配置：

```yaml
data:
  lq_root: ~/dit_codec/LQ
  gt_root: ~/dit_codec/GT
  lq_size: [256, 256]
  gt_size: [256, 256]
```

启动时会检查缺失文件、不可解码图像和配置尺寸。框架不在线合成退化 LQ。
`distill-codec probe` 的 JSON 输出包含 `preflight`、loss 和输出 shape。使用真实配置时，CLI 会先加载所需组件和权重，然后创建数据集并报告配对数量、相对路径和实际 LQ/GT 尺寸集合；只想提前检查数据时，请使用上面的 Mock 配置命令。
FlashVSR Recipe 如果读取到原生低分辨率 LQ，会按官方推理语义用 bicubic 对齐到 GT 目标尺寸，再送入 `LQ_proj_in` 或 TCDecoder；这不是随机退化生成。

## 分层配置

主配置可以组合多个 YAML，后写的 include 和主文件覆盖前面的值：

```yaml
includes:
  - ../teachers/wan_snapshot.yaml
  - ../students/private_blackbox.yaml

recipe:
  name: wan_encoder_distill
trainer:
  max_steps: 10000
```

Include 路径相对当前 YAML 文件解析，并检测循环引用。

## 你的黑盒接口

### 编码器

```python
def create_encoder(**kwargs) -> torch.nn.Module:
    # forward: [B,6,H/2,W/2] -> [B,16,H/8,W/8]
    return encrypted_encoder
```

```yaml
student_encoder:
  backend: external
  factory: my_package.models:create_encoder
  checkpoint: ~/dit_codec/weights/encoder.pth
  kwargs: {}
  adapter:
    kind: encoder
    input_mode: packed_6ch
```

Adapter 自动执行 RGB 到 `Y00,Y01,Y10,Y11,U,V`。U/V 默认由每个 `2x2` 块平均采样。

### 解码器

```python
def create_decoder(**kwargs) -> torch.nn.Module:
    # forward: [B,16,H/8,W/8] -> [B,3,H,W] sparse YUV
    return encrypted_decoder
```

```yaml
student_decoder:
  backend: external
  factory: my_package.models:create_decoder
  checkpoint: ~/dit_codec/weights/decoder.pth
  kwargs: {}
  adapter:
    kind: decoder
    output_mode: sparse_yuv
```

Adapter 只读取 U/V 的 `[0::2,0::2]` 位置，完成色度上采样，然后在 RGB 空间计算 loss。

如果你的加密包有特殊权重格式，让 factory 自行加载权重，并从组件配置删除顶层 `checkpoint`。

## Latent 契约

不要只凭 `[B,16,32,32]` 判断兼容。配置必须描述：

```yaml
latent_spec:
  family: wan_vae_v2
  channels: 16
  layout: BCHW
  spatial_downsample: 8
  temporal_downsample: 1
  normalization: wan_vae
```

如果冻结 DiT 不是使用 Wan latent 训练，必须同时替换：

1. `LatentSpec`；
2. 教师 Encoder；
3. 教师 Decoder；
4. 解码器训练使用的 latent provider。

否则 shape 即使相同，也不能保证 DiT、Encoder 和 Decoder 可互换。

## Decoder latent 来源

默认在线使用冻结教师 Encoder：

```yaml
latent_provider:
  type: teacher_encoder
  source: gt
```

低显存时可以使用缓存：

```yaml
latent_provider:
  type: cached
  root: ~/dit_codec/latents
```

缓存目录：

```text
$HOME/dit_codec/latents/manifest.pt
$HOME/dit_codec/latents/scene_01/000001.pt
```

`manifest.pt` 必须包含 `{"latent_spec": {...}}`。单个文件可以直接保存 `[C,H,W]` tensor，也可以保存 `{"latent": tensor}`。

`dataset` provider 接口也已提供，用于自定义 Dataset 在 `DistillBatch.latent` 中携带 latent。`frozen_dit` 名称已保留，但第一版会明确报错，不在线运行 DiT。

## Wan 教师

`third_party/wan/` 保存固定 DiffSynth-Studio commit 的 `WanVideoVAE` 源码和 Apache-2.0 许可证，权重不在 Git 中。

主路径参考 [configs/teachers/wan_external.yaml](configs/teachers/wan_external.yaml)，它指向你的本地上游仓库；备份路径参考 [configs/teachers/wan_snapshot.yaml](configs/teachers/wan_snapshot.yaml)。将源码和 `Wan2.1_VAE.pth` 路径改为本机路径。Wrapper 负责：

- `[0,1] RGB -> [-1,1]`；
- 单帧 `[B,C,H,W]` 与官方 `[B,C,T,H,W]` 适配；
- Wan normalized posterior mean latent；
- Decoder 输出 `[-1,1] -> [0,1]`。

Encoder/Decoder 模板使用同一个 `shared_key`，因此同一训练进程只构造一份 Wan VAE 权重，降低显存占用。

运行真实权重前先执行 `distill-codec probe`。

## FlashVSR 教师

`third_party/flashvsr/` 保存固定 commit 的：

- `Causal_LQ4x_Proj/Buffer_LQ4x_Proj`；
- `TCDecoder`；
- Apache-2.0 许可证。

主路径参考 [configs/teachers/flashvsr_external.yaml](configs/teachers/flashvsr_external.yaml)，备份路径参考 [configs/teachers/flashvsr_snapshot.yaml](configs/teachers/flashvsr_snapshot.yaml)。FlashVSR v1.1 官方配置为：

```text
LQ_proj_in: in_dim=3, out_dim=1536, layer_num=1
TCDecoder: 16 main latent + 768 condition channels
```

TCDecoder 的 768 条件通道由 LQ 的 `4x8x8` space-time-to-channel 重排产生。工程 wrapper 接收 16 通道 latent 和 LQ RGB，内部按官方布局把单帧重复为 4 帧并交给 TCDecoder。
FlashVSR 官方预处理使用 `[-1,1]` RGB；snapshot wrapper 会从 Dataset 的 `[0,1]` 自动转换，TCDecoder 输出再保持为训练器使用的 `[0,1]`。

无条件学生只接收 latent，可以蒸馏教师 RGB 输出，但不是可直接替换 TCDecoder 的等价接口。条件学生同时接收 latent 和 LQ，才对应官方 TCDecoder 的替换边界。

`LQ_proj_in` 是 DiT condition encoder，不是主 VAE Encoder；它有独立的 `flashvsr_lq_proj_distill` Recipe。
因此 FlashVSR 的默认可替换组合是 `LQ_proj_in -> student_condition_encoder` 和 `TCDecoder  -> conditional_student_decoder`；二者仍然分开训练。
官方 causal projection 会用首个 4 帧块预热缓存，所以单帧数据默认重复为 5 帧，产生一个有效 condition 时间块。
多帧教师返回视频 tensor 时，Encoder/Decoder Adapter 默认取中心帧；可以显式配置 `frame_selection: first|center|last`。`ConditionSpec` 同时记录 `spatial_downsample` 和 `temporal_downsample`，用于校验 BNC token、batch 和 feature 维度。

## 输出与恢复

运行目录包含：

```text
metrics.jsonl
checkpoints/step_00000001.pt
validation/step_00000001.png
tensorboard/                  # tensorboard: true 时
```

恢复：

```bash
distill-codec train \
  --config configs/local/wan_encoder.yaml \
  --resume "$HOME/dit_codec/runs/wan_encoder/checkpoints/step_00001000.pt"
```

Checkpoint 保存学生、optimizer、optimizer 参数名顺序、scheduler、AMP scaler、RNG、完整配置和 latent/color/condition 契约，不复制教师权重。当前版本可以稳定恢复同时训练编码器和解码器的多学生 recipe；旧版 checkpoint 若包含多个可训练组件但没有参数顺序信息，会明确拒绝恢复，避免静默错配 Adam 状态。

恢复还会检查 batch size、梯度累积、seed、数据目录/尺寸、recipe source/weights 和 compatibility interval。只有显式设置 `trainer.allow_contract_override: true` 才允许修改 latent family、normalization、value range 或颜色等非 shape 语义；通道数、layout、空间/时间下采样和 condition shape 始终不能覆盖。

训练器可以选择 AdamW 或 Adam，并限制 step checkpoint 数量：

```yaml
trainer:
  optimizer: adamw             # adamw 或 adam，默认 adamw
  learning_rate: 0.0001
  weight_decay: 0.01
  scheduler: cosine            # none 或 cosine，默认 none
  scheduler_max_steps: 10000   # cosine 的固定周期；分段 resume 时不要修改
  keep_last_checkpoints: 5     # 必须为正数；不配置则保留全部
```

`max_steps` 可以控制本次运行先停在哪一步；使用 cosine 时，`scheduler_max_steps` 表示完整训练周期，必须在首次运行前设定，并在所有 resume 阶段保持一致。恢复时会严格检查 optimizer、学习率、weight decay、scheduler 和 scheduler 周期。

Encoder compatibility loss 可以关闭，或降低计算频率：

```yaml
recipe:
  compatibility_every: 4
  weights:
    compat: 0.1                 # 设为 0 时训练和验证都完全跳过教师 Decoder
```

当 compatibility 开启时，验证阶段不受 interval 限制并计算完整输出。Decoder 验证指标明确区分 `rgb_mae_vs_teacher` 和 `rgb_mae_vs_gt`；验证图固定为 `LQ | GT | teacher | student | absolute-error heatmap`。

模型组件可配置 `sha256`；设置后会在构造模型前校验 checkpoint。LPIPS 默认关闭，设置 `recipe.weights.lpips` 为非零前先安装：

```bash
python -m pip install -e ".[perceptual]"
```

## 当前限制

- 单设备训练；
- Dataset 只读取单帧配对图像；
- 视频教师通过重复同一帧获得时间维；
- 不训练或在线运行 DiT；
- 不自动下载权重；
- LPIPS 是可选依赖，默认图像 loss 是 L1 + edge；
- PSNR/SSIM 实现用于训练期轻量验证，不替代正式 benchmark 工具。

## 运行安全与可复现性

- wheel 内置与 `third_party/wan`、`third_party/flashvsr` 清单 SHA256 一致的源码快照；不指定 `source_file` 时，snapshot factory 使用包内副本。模型权重仍需自行提供。
- 外部模型权重和 cached latent 使用 PyTorch `weights_only=True` 加载，只接受 tensor、state dict 和基础容器。需要执行任意 Python 对象反序列化的私有格式应在你自己的可信 factory 中显式处理。
- TCDecoder 默认允许 `strict: false`，但至少要求 checkpoint 覆盖 50% 的目标参数量；可用 `minimum_coverage` 调高阈值，避免错误权重被静默当成教师。
- 标准 `PairedImageDataset` 不产生 latent，因此标准 CLI/Trainer 会提前拒绝 `latent_provider.type: dataset`。使用 `cached`，或编写能够填充 `DistillBatch.latent` 的自定义 Dataset/Trainer。
- checkpoint 记录已消费的训练 batch 数。相同配置、seed、数据集和 batch 参数下，resume 会重放 sampler 位置，使续训的数据顺序与不中断训练一致。

扩展新的编解码器见 [docs/ADDING_A_BACKEND.md](docs/ADDING_A_BACKEND.md)。
