# 可扩展编解码器蒸馏工程设计

日期：2026-08-18

## 1. 目标

构建一个可扩展的 PyTorch 编解码器蒸馏工程。第一版必须在没有真实 Wan/FlashVSR 权重、没有私有黑盒网络源码的情况下，通过 mock 模型完整运行训练、验证、checkpoint 保存和恢复；接入真实模型时只替换配置和 Adapter，不修改训练引擎。

工程首批提供两套参考方案：

1. Wan VAE 编码器/解码器蒸馏。
2. FlashVSR 条件分支和 TCDecoder 蒸馏，包括无条件学生和条件学生两种模式。

工程不把 latent 通道数、空间压缩倍率、时间布局、归一化、颜色矩阵或条件输入写死。每套 DiT 使用哪种 latent，由显式契约和 Recipe 决定。

## 2. 核心原则

### 2.1 区分主 latent 与条件特征

Wan/FlashVSR 的主数据通路和条件通路是两个不同接口：

```text
主 latent 通路：
RGB -> VAE Encoder -> latent -> DiT -> latent -> VAE/TC Decoder -> RGB

FlashVSR 条件通路：
LQ RGB/Video -> LQ_proj_in -> DiT condition features
```

Wan VAE Encoder 产生 DiT 使用的主 latent。`LQ_proj_in` 产生条件特征，不是 VAE 编码器，二者不能互相替代。

### 2.2 契约驱动

训练器不根据 shape 猜测兼容性。即使两个张量都是 `[B,16,32,32]`，只有以下字段一致时才能视为同一种 latent：

- 通道数；
- 空间和时间压缩倍率；
- tensor layout；
- 数值归一化；
- latent family；
- dtype 和有效数值范围。

### 2.3 Adapter 与 Recipe 分离

- Adapter 负责把具体模型转换为统一接口，包括输入预处理、forward 调用、输出解析和版本兼容。
- Recipe 负责定义教师、学生、数据源、latent 来源、条件输入、loss 和验证指标如何组合。
- Engine 只执行 Recipe，不感知 Wan、FlashVSR 或私有模型内部结构。

## 3. 用户黑盒学生契约

黑盒网络是可训练的 PyTorch `nn.Module`，支持正常反向传播，但工程不能依赖其内部源码。

### 3.1 学生编码器

```text
RGB [B,3,256,256]
 -> 可微 RGB 到 6 通道打包
 -> [B,6,128,128]
 -> 黑盒 Student Encoder
 -> latent [B,16,32,32]
```

每个 `2x2` RGB 块被转换并打包为：

```text
Y00, Y01, Y10, Y11, U, V
```

工程内部把该格式命名为 `packed_6ch_420`，避免仅凭 YUV420/YUV422 名称推断布局。具体行为由 `ColorSpec` 明确定义。

### 3.2 学生解码器

```text
latent [B,16,32,32]
 -> 黑盒 Student Decoder
 -> sparse YUV [B,3,256,256]
 -> 可微色度恢复和 YUV 到 RGB
 -> RGB [B,3,256,256]
```

输出 channel 0 是全分辨率 Y。channel 1 和 2 仅在每个 `2x2` 块左上角保存有效 U/V。Adapter 提取有效色度样本，然后按配置进行 nearest 或 bilinear 上采样，再转换成 RGB。

### 3.3 私有模型接入

私有网络通过 Python factory 加载：

```yaml
student:
  backend: external
  factory: my_encrypted_package.models:create_encoder
  checkpoint: D:/weights/my_encoder.pth
  kwargs: {}
```

Adapter 只要求模型满足输入输出契约。模型权重保存策略支持：

- 模型自身 `state_dict()`；
- Adapter 提供自定义 `state_dict/load_state_dict`；
- 必要时由用户 factory 返回带自定义保存钩子的对象。

## 4. 数据契约

第一版只支持严格配对的单帧 LQ/GT 图像目录：

```text
LQ_ROOT/scene1/0001.png
GT_ROOT/scene1/0001.png
```

以相对路径作为样本 ID。启动训练前完成以下检查：

- LQ 和 GT 相对路径集合完全匹配；
- 文件可以解码；
- 通道数符合要求；
- 图像尺寸满足当前 Recipe 和模型约束；
- 不允许重复样本 ID；
- 发现缺失或多余文件时立即报错，不在训练中静默跳过。

Dataset 输出：

```python
DistillBatch(
    lq_rgb=Tensor[B, 3, H, W],
    gt_rgb=Tensor[B, 3, H, W],
    relative_path=list[str],
)
```

Dataset 层统一输出 `[0,1]` RGB float。教师和学生 Adapter 各自负责后续范围转换。

视频教师需要多帧时，`TemporalAdapter` 将同一单帧重复为 `[B,3,T,H,W]`。教师输出多帧时，默认选取中心帧与单帧学生对齐；帧数和取帧策略可配置。第一版不读取真实视频序列。

## 5. 统一契约

### 5.1 LatentSpec

```yaml
latent_spec:
  family: wan_vae_v2
  channels: 16
  layout: BCTHW
  spatial_downsample: 8
  temporal_downsample: 4
  normalization: wan_vae
  value_range: unbounded
```

单帧学生可使用 `BCHW`；时间适配由 Adapter 显式 squeeze/unsqueeze，不能在 Engine 内隐式处理。

### 5.2 ConditionSpec

条件契约至少记录：

- condition family；
- tensor 或结构化输出的 layout；
- feature 维度；
- 对应的空间/时间采样关系；
- 来源是 LQ、GT 还是外部缓存；
- 目标消费者是 DiT 还是 Decoder。

`DiTConditionAdapter` 和 `DecoderConditionAdapter` 必须分开。`LQ_proj_in` 输出不能直接当作 TCDecoder 的 RGB/LQ 条件。

### 5.3 ColorSpec

```yaml
color:
  matrix: bt709
  range: full
  packed_order: Y00Y01Y10Y11UV
  chroma_location: top_left
  chroma_upsample: nearest
```

支持 `bt601/bt709`、`full/limited` 和 `nearest/bilinear`。训练配置必须明确指定，checkpoint 同时保存该契约。

## 6. 模型后端

所有教师和学生 Adapter 支持三种后端：

```yaml
backend: mock      # 无真实权重的可运行基线
backend: external  # 主要真实运行路径
backend: snapshot  # third_party 源码快照
```

### 6.1 external

这是实际训练的主路径。配置指定外部仓库或可导入 Python 包、factory、checkpoint 和构造参数。Adapter 处理不同上游版本的权重 key、构造函数和 forward 差异。

### 6.2 snapshot

`third_party/` 保存项目实际需要的 Wan 和 FlashVSR 官方源码快照、许可证和来源信息。快照用于版本对照、断网复现和备份，不作为修改上游实现的工作目录。

每个快照必须包含：

- 上游项目名称和 URL；
- 固定 commit/tag；
- 获取日期；
- 许可证原文；
- 复制的文件清单；
- 与外部后端相比的已知限制。

兼容性修改写在本工程 Adapter 内，不直接修改快照。若必须打补丁，则把独立 patch 文件和原因记录在 `third_party/patches/`，保持原文件可审计。

### 6.3 权重管理

`.pth`、`.pt`、`.ckpt` 和 `.safetensors` 等大模型权重不提交 Git。配置引用本地绝对或相对路径。工程提供文件名、来源说明和可选 SHA256 校验；不在训练入口中自动下载权重。

## 7. Mock 模型

Mock 必须是可训练的小型卷积网络，而不是随机 shape 占位：

- `MockStudentEncoder`：`[B,6,H/2,W/2] -> [B,16,H/8,W/8]`；
- `MockStudentDecoder`：`[B,16,H/8,W/8] -> sparse YUV [B,3,H,W]`；
- `MockConditionalStudentDecoder`：接收 latent 和 LQ RGB；
- `MockConditionEncoder`：LQ RGB video 到 condition features；
- `MockWanEncoder/MockWanDecoder`：冻结的主 latent 教师；
- `MockLQProjIn/MockTCDecoder`：冻结的 FlashVSR 条件教师。

Mock 教师和学生必须具有确定性初始化选项。Smoke test 要验证 loss 有限、学生参数得到非零梯度、optimizer 能更新参数，而不仅是验证 shape。

提供 mock 配对数据生成脚本，用少量同名 LQ/GT 图片完成 CPU 端到端运行。

## 8. Recipe 设计

### 8.1 wan_encoder_distill

```text
GT RGB -> Frozen Wan Encoder -> z_teacher
GT RGB -> packed_6ch_420 -> Student Encoder -> z_student
```

默认输入 GT。允许显式配置 `source: lq`，但同一次训练只能选择一种 source。

损失：

```text
L_encoder =
  lambda_latent * SmoothL1(z_student, z_teacher)
+ lambda_cos    * CosineLoss(z_student, z_teacher)
+ lambda_stat   * ChannelStatLoss(z_student, z_teacher)
+ lambda_compat * RGBLoss(WanDecoder(z_student), source_rgb)
```

兼容性 loss 使用冻结教师 Decoder，但不能阻断从 Decoder 输入到学生 latent 的梯度。冻结方式是将教师参数设置为 `requires_grad=False`；兼容性 forward 不能被整体包在 `torch.no_grad()` 中。

兼容性 loss 可关闭或按固定 step 间隔计算，以控制显存和训练时间。

### 8.2 wan_decoder_distill

```text
GT -> Frozen Wan Encoder -> z
z  -> Frozen Wan Decoder -> teacher_rgb
z  -> Student Decoder -> sparse_yuv -> student_rgb
```

默认 latent 来自冻结 Wan Encoder，不运行 DiT。

```text
L_decoder =
  lambda_teacher * L1(student_rgb, teacher_rgb)
+ lambda_gt      * L1(student_rgb, gt_rgb)
+ lambda_edge    * EdgeLoss(student_rgb, gt_rgb)
+ lambda_lpips   * LPIPS(student_rgb, gt_rgb)
```

LPIPS 是可选依赖，默认 smoke 配置关闭。

### 8.3 wan_autoencoder_distill

用于联合验证学生 Encoder 和 Decoder：

```text
GT -> packed input -> Student Encoder -> Student Decoder -> RGB
```

同时计算 encoder latent loss、teacher decoder compatibility loss 和 student reconstruction loss。第一版提供 Recipe，但不把联合训练作为默认方式；编码器和解码器分开训练仍是推荐的低显存流程。

### 8.4 flashvsr_vae_encoder_distill

复用 Wan 主 latent 编码器蒸馏逻辑。教师是 FlashVSR 所使用版本对应的 Wan VAE Encoder。它不使用 `LQ_proj_in` 作为主 latent 教师。

### 8.5 flashvsr_lq_proj_distill

```text
LQ repeated video -> Frozen LQ_proj_in -> teacher_condition
LQ repeated video -> StudentConditionEncoder -> student_condition
```

该 Recipe 使用独立 `StudentConditionEncoder` 接口。用户当前 `[B,16,32,32]` 黑盒编码器不会被错误接入此任务。

condition loss 根据 `ConditionSpec` 选择逐元素、cosine 和统计匹配；如果上游返回多层结构，Adapter 先规范化为命名 tensor 字典。

### 8.6 flashvsr_decoder_unconditional_student

```text
z + LQ condition -> Frozen TCDecoder -> teacher_rgb
z                -> Student Decoder  -> student_rgb
```

这是兼容用户当前单输入黑盒解码器的 FlashVSR 案例。条件只用于生成教师目标。验证报告必须明确标记该学生未接收 LQ 条件，避免把结果解释成完全等价的 TCDecoder 替代品。

### 8.7 flashvsr_decoder_conditional_student

```text
z + LQ condition -> Frozen TCDecoder           -> teacher_rgb
z + LQ condition -> ConditionalStudentDecoder  -> student_rgb
```

该 Recipe 使用独立的条件学生接口。Decoder 的 LQ condition 由 `DecoderConditionAdapter` 构造，不使用 `LQ_proj_in` 的输出替代。

## 9. LatentProvider

所有 Decoder Recipe 通过统一 `LatentProvider` 获得输入：

```yaml
latent_provider:
  type: teacher_encoder
  source: gt
```

第一版实现：

- `teacher_encoder`：在线使用冻结 VAE Encoder，默认路径；
- `cached`：按样本 ID 从磁盘读取预计算 latent；
- `dataset`：Dataset 直接提供 latent 的扩展接口。

保留但第一版不实现在线执行逻辑：

- `frozen_dit`：以后显存允许时在线运行冻结 DiT。

配置选择未实现的 provider 时必须给出明确错误。Cached latent 文件必须携带或关联 `LatentSpec`，加载时严格校验。

## 10. 训练引擎

第一版支持：

- 单 GPU 和 CPU mock；
- PyTorch AMP；
- 梯度累积；
- 可配置 optimizer 和 scheduler；
- 梯度裁剪；
- 周期验证；
- TensorBoard 标量与图像日志；
- 固定随机种子；
- Windows 路径；
- checkpoint 保存和恢复。

教师始终处于 eval 模式且参数冻结。Engine 根据 Recipe 决定哪些教师 forward 可用 `inference_mode/no_grad`，哪些必须保留对输入的 autograd 路径。

Checkpoint 保存：

- 学生和必要 Adapter 的 state dict；
- optimizer、scheduler 和 AMP scaler；
- epoch、global step；
- 完整解析后配置；
- `LatentSpec`、`ConditionSpec` 和 `ColorSpec`；
- 最佳验证指标；
- Python、NumPy、PyTorch 和 CUDA 随机数状态。

Checkpoint 不重复保存教师权重。恢复时严格检查 Recipe 名称和契约；只有显式 `allow_contract_override` 才允许跳过非 shape 字段，shape 不兼容永远报错。

## 11. 指标和可视化

编码器验证：

- latent MAE 和 RMSE；
- 空间位置通道 cosine similarity；
- 每通道 mean/std 偏差；
- 冻结教师 Decoder 解码学生 latent 后，相对目标 RGB 的 PSNR/SSIM。

解码器验证：

- 相对 teacher RGB 的 PSNR/SSIM；
- 相对 GT RGB 的 PSNR/SSIM；
- RGB 误差；
- 按当前 `ColorSpec` 转换后的 Y/U/V 分量误差；
- 可选 LPIPS。

每次验证对固定样本保存：

```text
LQ | GT | teacher | student | absolute error heatmap
```

指标必须区分 `vs_teacher` 和 `vs_gt`，不能只报告一个含义不清的 PSNR。

## 12. 配置和目录

```text
DIT_DEC_ENC/
├─ configs/
│  ├─ smoke/
│  ├─ data/
│  ├─ teachers/
│  ├─ students/
│  └─ recipes/
├─ src/distill_codec/
│  ├─ contracts/
│  ├─ color/
│  ├─ data/
│  ├─ adapters/
│  │  ├─ teachers/
│  │  └─ students/
│  ├─ models/mock/
│  ├─ recipes/
│  ├─ losses/
│  ├─ metrics/
│  ├─ engine/
│  └─ cli/
├─ third_party/
│  ├─ wan/
│  ├─ flashvsr/
│  └─ patches/
├─ tests/
├─ scripts/
├─ docs/superpowers/specs/
└─ pyproject.toml
```

配置按数据、教师、学生和 Recipe 分层，CLI 接收一个合并后的主配置。第一版不构建任意动态计算图；合法组件组合由 Recipe 明确约束，以获得更可靠的错误信息。

## 13. 错误处理

以下问题在训练开始前失败：

- 配对目录不一致；
- teacher/student factory 无法导入；
- 权重缺失或 SHA256 不匹配；
- 模型 probe forward 输出不符合契约；
- latent family、normalization、layout 或 shape 不匹配；
- Recipe 需要条件但 batch/Adapter 未提供；
- 颜色格式配置不完整；
- cached latent 与样本或 `LatentSpec` 不一致；
- 使用尚未实现的 provider 或 backend 能力。

错误信息包含组件名、期望契约、实际 shape/spec 和相关配置路径。训练中遇到 NaN/Inf 时报告 loss 分项、样本相对路径和 global step，并停止当前运行。

## 14. 测试与完成标准

单元测试覆盖：

- RGB 到 packed 6 通道的数值、shape 和梯度；
- sparse U/V 有效位置提取、色度恢复和 RGB 转换；
- LQ/GT 配对、缺失文件和尺寸错误；
- `LatentSpec/ConditionSpec/ColorSpec` 校验；
- Python factory 动态加载；
- teacher 参数冻结；
- compatibility loss 能经过冻结 Decoder 回传到学生 Encoder；
- 条件和无条件 TCDecoder Recipe；
- checkpoint round trip。

集成测试覆盖每个主要 Recipe 至少一次 CPU 训练 step。完整 smoke test 必须：

1. 生成 mock LQ/GT 配对数据；
2. 运行至少一个 encoder 和一个 decoder Recipe；
3. 完成训练和验证；
4. 确认学生参数更新且 loss 有限；
5. 保存 checkpoint；
6. 从 checkpoint 恢复并继续至少一个 step；
7. 生成 TensorBoard 日志和固定验证图像。

第一版完成的判定标准是：全新环境安装项目后，不需要真实权重即可通过测试和 mock smoke；配置本地真实权重与外部源码后，不修改训练器即可完成模型 probe 和对应 Recipe 的训练启动。

## 15. 第一版不做

- 不训练或蒸馏 DiT 主体；
- 不在线运行冻结 DiT；
- 不支持分布式训练；
- 不自动下载大模型权重；
- 不在线生成退化 LQ；
- 不支持真实视频序列 Dataset；
- 不假设或修改用户加密学生模型内部结构；
- 不承诺默认 loss 权重为最优值。

默认 loss 权重只是可运行起点。真实训练前应通过 loss/gradient 量级记录、小规模过拟合和消融实验调整。

