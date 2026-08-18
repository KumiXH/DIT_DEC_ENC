# DIT Codec Distillation

一个契约驱动的 PyTorch 编解码器蒸馏工程，面向以下场景：

- 学生编码器输入 `[B,6,H/2,W/2]` 的 `Y00,Y01,Y10,Y11,U,V` 打包数据，输出 `[B,16,H/8,W/8]` latent；
- 学生解码器输入 latent，输出 U/V 仅在每个 `2x2` 左上角有效的 sparse YUV；
- 使用 Wan VAE 或 FlashVSR 组件作为教师；
- 学生模型是可训练但源码不可见的 PyTorch `nn.Module`；
- 不训练 DiT，且允许编码器、解码器分开蒸馏。

第一版提供完整 mock 模型。没有真实权重时也能运行数据读取、forward、loss、backward、验证、checkpoint 和 resume。

## 安装

需要 Python 3.10+ 和与你的 CUDA 环境匹配的 PyTorch：

```powershell
python -m pip install -e ".[train,test]"
```

如果只使用 JSONL 日志，可以不安装 `train` extras。真实 Wan/FlashVSR 源码还依赖 `einops` 和 `tqdm`，已列入基础依赖。

## 最快运行

```powershell
distill-codec make-mock-data --output work/mock_data --count 8 --size 64

distill-codec probe `
  --config configs/smoke/wan_encoder.yaml `
  --set "data.lq_root=work/mock_data/lq" `
  --set "data.gt_root=work/mock_data/gt"

distill-codec train `
  --config configs/smoke/wan_encoder.yaml `
  --set "data.lq_root=work/mock_data/lq" `
  --set "data.gt_root=work/mock_data/gt" `
  --set "run.output_dir=runs/wan_encoder"
```

完整 CPU smoke：

```powershell
./scripts/run_smoke.ps1 -PythonExe "C:/path/to/python.exe"
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

## 数据目录

LQ 和 GT 必须有完全相同的相对目录和文件名：

```text
D:/dataset/LQ/scene_01/000001.png
D:/dataset/GT/scene_01/000001.png
```

配置：

```yaml
data:
  lq_root: D:/dataset/LQ
  gt_root: D:/dataset/GT
  lq_size: [256, 256]
  gt_size: [256, 256]
```

启动时会检查缺失文件、不可解码图像和配置尺寸。框架不在线合成退化 LQ。
FlashVSR Recipe 如果读取到原生低分辨率 LQ，会按官方推理语义用 bicubic 对齐到 GT 目标尺寸，再送入 `LQ_proj_in` 或 TCDecoder；这不是随机退化生成。

## 分层配置

主配置可以组合多个 YAML，后写的 include 和主文件覆盖前面的值：

```yaml
includes:
  - ../data/paired_256.yaml
  - ../teachers/wan_external.yaml
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
  checkpoint: D:/weights/encoder.pth
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
  checkpoint: D:/weights/decoder.pth
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
  root: D:/latents
```

缓存目录：

```text
D:/latents/manifest.pt
D:/latents/scene_01/000001.pt
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

无条件学生只接收 latent，可以蒸馏教师 RGB 输出，但不能声明为与官方 TCDecoder 接口等价。条件学生则接收 latent 和 LQ。

`LQ_proj_in` 是 DiT condition encoder，不是主 VAE Encoder；它有独立的 `flashvsr_lq_proj_distill` Recipe。
官方 causal projection 会用首个 4 帧块预热缓存，所以单帧数据默认重复为 5 帧，产生一个有效 condition 时间块。

## 输出与恢复

运行目录包含：

```text
metrics.jsonl
checkpoints/step_00000001.pt
validation/step_00000001.png
tensorboard/                  # tensorboard: true 时
```

恢复：

```powershell
distill-codec train --config config.yaml --resume runs/example/checkpoints/step_00001000.pt
```

Checkpoint 保存学生、optimizer、optimizer 参数名顺序、scheduler、AMP scaler、RNG、完整配置和 latent/color/condition 契约，不复制教师权重。当前版本可以稳定恢复同时训练编码器和解码器的多学生 recipe；旧版 checkpoint 若包含多个可训练组件但没有参数顺序信息，会明确拒绝恢复，避免静默错配 Adam 状态。

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

模型组件可配置 `sha256`；设置后会在构造模型前校验 checkpoint。LPIPS 默认关闭，设置 `recipe.weights.lpips` 为非零前先安装：

```powershell
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
