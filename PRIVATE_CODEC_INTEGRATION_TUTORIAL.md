# 私有黑盒编解码器接入教程：从 v0 复制新版本

这份教程就是第二份 README。工程已经提供一个可运行、带完整注释的 `v0` 卷积示例。你先运行它确认环境和调用链，再把整个 v0 目录复制成 v1、v2 或实验版本，依葫芦画瓢替换成自己的两个网络文件。

## 当前完成和验证状态

框架已经验证的部分：

- Encoder 的项目接口固定为 `encoder(rgb)`，项目只给黑盒网络 `[B,3,H,W]` RGB。
- 条件 Decoder 的项目接口固定为 `conditional_student_decoder(dit_latent, lq_rgb)`。
- 私有 Decoder 入口收到 `network`、原始 `lq_rgb`、`dit_latent` 和 `teacher_reference`。
- `teacher_reference` 只包含理论教师输入输出尺寸，不会 resize、padding、裁切或修改真实 Tensor。
- Encoder 可以返回符合配置的 BCHW 或 BCTHW latent。
- v0 示例网络已经验证：Encoder、条件 Decoder、无条件 Decoder、反向传播、配置导入和错误提示都有自动化测试。
- 真实私有网络仍需验证：复制 v0 并替换源码后，仍要在你的 CUDA 训练环境运行 probe 和短训练。
- 当前测试环境有一个与本功能无关的已知排除项：`test_multi_component_resume_is_stable_across_python_hash_seeds` 会在 Conda 子进程重复加载 OpenMP 时崩溃。

## 第 0 步：认识现在的目录

公共代码和版本代码已经分开：

```text
src/private_codec/
├── __init__.py
├── bridge.py
├── factories.py
└── versions/
    ├── __init__.py
    └── v0/
        ├── __init__.py
        ├── base_network.py
        ├── wrapped_network.py
        └── entrypoints.py
```

三个可以复制修改的 v0 文件是：

```text
src/private_codec/versions/v0/base_network.py
src/private_codec/versions/v0/wrapped_network.py
src/private_codec/versions/v0/entrypoints.py
```

它们的职责分别是：

- `base_network.py`：第一个网络文件，保存基础网络本身。
- `wrapped_network.py`：第二个网络文件，继承基础网络、设置不同初始化参数或包装私有 `forward`。
- `entrypoints.py`：中间地带，负责项目输入到私有网络输入的全部转换，以及私有输出到项目输出的映射。

公共基础设施不属于任何网络版本。增加 v1、v2 时，不要修改 `src/private_codec/bridge.py`，也不要修改 `src/private_codec/factories.py`。

## 第 1 步：准备 Linux Python 环境

以下命令以 Ubuntu/Bash 为准。先进入仓库根目录，创建独立虚拟环境并安装项目：

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train,test]"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

python -c "import torch; print(torch.__version__); print('cuda=', torch.cuda.is_available())"
```

上面的 `pip install -e` 会把当前源码仓库安装为 editable package；`PYTHONPATH` 同时保证你修改版本目录后立刻导入最新源码。正式训练前还需要安装与你的 NVIDIA 驱动和 CUDA 版本匹配的 PyTorch，并确认 `nvidia-smi` 可以看到目标 GPU。

## 第 2 步：先看懂 v0 Encoder

打开：

```text
src/private_codec/versions/v0/base_network.py
```

v0 Encoder 使用三层步长为 2 的卷积：

```text
RGB [B,3,H,W]
  -> stride-2 convolution
  -> stride-2 convolution
  -> stride-2 convolution
  -> latent [B,16,H/8,W/8]
```

默认 `latent_channels=16`，输入高宽必须能被 8 整除。这个限制只是 v0 示例自己的限制，不是公共 Bridge 强加给真实黑盒网络的限制。复制成新版本后，你可以在私有代码里自行 resize、padding 或使用完全不同的结构，只要最终返回值符合项目 `latent_spec`。

## 第 3 步：先看懂 v0 Decoder

同一个 `base_network.py` 中，条件 Decoder 做三件事：

```text
dit_latent -> 卷积投影 -> 上采样到 latent*8 目标尺寸 ┐
                                                    ├-> 特征融合 -> RGB
lq_rgb     -> 卷积投影 -> 上采样到同一目标尺寸 ------┘
```

v0 返回 `[B,3,Hlatent*8,Wlatent*8]` RGB。比如 `32x32` LQ 与 `8x8` latent 会得到 `64x64` RGB；LQ 特征在私有网络内部上采样。这个规则只是为了让 v0 能演示真实超分辨率训练，仍然说明尺寸处理属于私有网络，而不是 Bridge。你的版本可以采用别的规则。

`wrapped_network.py` 中的私有条件 Decoder 故意定义成：

```python
network(private_lq_rgb, private_latent)
```

它和项目内部的顺序不同，目的是直接演示 `entrypoints.py` 可以自由适配你的私有 `forward`。

## 第 4 步：理解唯一的中间层

打开：

```text
src/private_codec/versions/v0/entrypoints.py
```

每个版本都保留以下四个对外函数：

```python
def build_encoder(**kwargs):
    ...


def run_encoder(*, network, rgb, teacher_reference, **kwargs):
    ...


def build_decoder(**kwargs):
    ...


def run_decoder(
    *, network, lq_rgb, dit_latent, teacher_reference, **kwargs
):
    ...
```

Encoder 的完整调用链：

```text
项目 RGB [B,3,H,W]
  -> EncoderAdapter(input_mode=rgb)
  -> PrivateEncoderBridge
  -> run_encoder(network=..., rgb=..., teacher_reference=...)
  -> 你的 latent
```

条件 Decoder 的完整调用链：

```text
项目调用 conditional_student_decoder(dit_latent, lq_rgb)
  -> PrivateConditionalDecoderBridge
  -> run_decoder(
         network=network,
         lq_rgb=lq_rgb,
         dit_latent=dit_latent,
         teacher_reference=teacher_reference,
     )
  -> 你的 RGB [B,3,H,W]
```

项目位置参数顺序虽然是 `(dit_latent, lq_rgb)`，Bridge 到私有入口后全部使用有名字的关键字参数。v0 的 `run_decoder` 再调用：

```python
network(private_lq_rgb, private_latent)
```

因此，无论你的嵌套网络采用什么参数顺序，都在 `run_decoder` 中明确映射即可。

## 第 5 步：运行 v0 语法和导入检查

```bash
python -m py_compile src/private_codec/versions/v0/base_network.py
python -m py_compile src/private_codec/versions/v0/wrapped_network.py
python -m py_compile src/private_codec/versions/v0/entrypoints.py
python -c "import torch; import private_codec.versions.v0.entrypoints; print('v0 imports ok')"
```

`py_compile` 退出码为 0 只代表语法正确。下面的 Bridge probe 才会实际构造网络并执行 forward。

## 第 6 步：单独运行 v0 Encoder Bridge

```bash
python -c "import torch; from private_codec.factories import create_encoder; m=create_encoder(builder='private_codec.versions.v0.entrypoints:build_encoder', runner='private_codec.versions.v0.entrypoints:run_encoder', builder_kwargs={}, runner_kwargs={}, teacher_reference={'role':'encoder'}); x=torch.randn(1,3,256,256); y=m(x); print('encoder output=', tuple(y.shape), y.dtype)"
```

默认输出应为：

```text
encoder output= (1, 16, 32, 32)
```

这里用到的公共 factory 是 `private_codec.factories:create_encoder`。

## 第 7 步：单独运行 v0 条件 Decoder Bridge

```bash
python -c "import torch; from private_codec.factories import create_conditional_decoder; m=create_conditional_decoder(builder='private_codec.versions.v0.entrypoints:build_decoder', runner='private_codec.versions.v0.entrypoints:run_decoder', builder_kwargs={}, runner_kwargs={}, teacher_reference={'role':'conditional_decoder'}); z=torch.randn(1,16,32,32); lq=torch.randn(1,3,256,256); y=m(z,lq); print('decoder output=', tuple(y.shape), y.dtype)"
```

默认输出应为：

```text
decoder output= (1, 3, 256, 256)
```

这里用到的公共 factory 是 `private_codec.factories:create_conditional_decoder`，Adapter 配置必须保持 `output_mode: rgb`。

## 第 8 步：理解默认 YAML

打开：

```text
configs/students/private_codec.yaml
```

Encoder 默认使用 v0：

```yaml
student_encoder:
  backend: external
  factory: private_codec.factories:create_encoder
  checkpoint: null
  teacher_reference: auto
  kwargs:
    builder: private_codec.versions.v0.entrypoints:build_encoder
    runner: private_codec.versions.v0.entrypoints:run_encoder
    builder_kwargs: {}
    runner_kwargs: {}
  adapter:
    kind: encoder
    input_mode: rgb
    latent_temporal_frames: teacher
```

条件 Decoder 默认使用 v0：

```yaml
conditional_student_decoder:
  backend: external
  factory: private_codec.factories:create_conditional_decoder
  checkpoint: null
  teacher_reference: auto
  kwargs:
    builder: private_codec.versions.v0.entrypoints:build_decoder
    runner: private_codec.versions.v0.entrypoints:run_decoder
    builder_kwargs: {}
    runner_kwargs: {}
  adapter:
    kind: decoder
    output_mode: rgb
    accepts_condition: true
```

参数用途：

- `builder_kwargs`：初始化网络时传给 `build_encoder` 或 `build_decoder`，适合通道数、配置路径、版本名和私有权重路径。
- `runner_kwargs`：每次 forward 时传给 `run_encoder` 或 `run_decoder`，适合运行模式或可微处理选项。
- `checkpoint`：只有整个 Bridge 可以直接用标准 `state_dict` 加载时才填写；特殊权重格式可以在 `build_*` 内自行加载。
- `latent_temporal_frames: teacher`：只约束 BCTHW latent 的理论时间长度，不会把传给私有 Encoder 的 RGB 变成视频 Tensor。

## 第 9 步：查看教师理论尺寸

保留：

```yaml
teacher_reference: auto
```

Encoder 的 `teacher_reference` 类似：

```python
{
    "role": "encoder",
    "inputs": {
        "rgb": {
            "layout": "BCHW",
            "shape": [None, 3, 256, 256],
            "source": "gt",
        },
    },
    "outputs": {
        "latent": {
            "layout": "BCHW",
            "shape": [None, 16, 32, 32],
        },
    },
}
```

如果配置使用 BCTHW，它也可能给出 `[B,C,T,H,W]` 理论 latent 尺寸。条件 Decoder 会收到理论 LQ RGB、DiT latent 和输出 RGB 尺寸。

这些信息只方便打印、断言和调试。不要因为理论尺寸不同就在公共 Bridge 中修改真实 Tensor。FlashVSR 教师可能使用对齐到 GT 的 LQ，但私有 `run_decoder` 收到的仍是原始 `batch.lq_rgb`。

## 第 10 步：复制 v0 创建你的 v1

在仓库根目录执行：

```bash
cp -a src/private_codec/versions/v0 src/private_codec/versions/v1
```

复制后得到：

```text
src/private_codec/versions/v1/
├── __init__.py
├── base_network.py
├── wrapped_network.py
└── entrypoints.py
```

然后按顺序操作：

1. 把你的第一个网络源码放入 `versions/v1/base_network.py`。
2. 把继承网络或外套 `forward` 的第二个源码放入 `versions/v1/wrapped_network.py`。
3. 在 `versions/v1/entrypoints.py` 修改 import、初始化参数和 forward 映射。
4. 搜索 v0 文件中的 `COPY POINT` 注释；这些就是复制后最需要检查的位置。
5. 不要删除四个 entrypoint 函数，也不要改变它们面向 Bridge 的参数名称。

你的包内导入应使用相对路径，例如：

```python
from .base_network import BaseNetwork
```

## 第 11 步：把 YAML 从 v0 切到 v1

建议先复制一份 student YAML。Encoder 和条件 Decoder 需要修改四条 builder/runner 路径：

```yaml
kwargs:
  builder: private_codec.versions.v1.entrypoints:build_encoder
  runner: private_codec.versions.v1.entrypoints:run_encoder
  builder_kwargs:
    config_path: ~/dit_codec/models/private_codec/v1.yaml
    weights_path: ~/dit_codec/models/private_codec/v1.pth
  runner_kwargs: {}
```

条件 Decoder 使用：

```yaml
kwargs:
  builder: private_codec.versions.v1.entrypoints:build_decoder
  runner: private_codec.versions.v1.entrypoints:run_decoder
  builder_kwargs:
    config_path: ~/dit_codec/models/private_codec/v1.yaml
    weights_path: ~/dit_codec/models/private_codec/v1.pth
  runner_kwargs: {}
```

如果还使用旧的 Wan 无条件 Decoder 或 Autoencoder Recipe，还要修改它的模块路径和类名：

```yaml
student_decoder:
  kwargs:
    module_path: private_codec.versions.v1.wrapped_network
    class_name: V1UnconditionalDecoder
    init_kwargs: {}
```

如果你不重命名复制后的类，也可以继续填写实际存在的类名；关键是 `module_path` 和 `class_name` 必须共同指向新版本。

以后增加 v2、v3 时，继续复制完整版本目录并把路径改为 `private_codec.versions.v2.entrypoints`、`private_codec.versions.v3.entrypoints`。不要给每个版本在公共 factory 里新增函数。

## 第 12 步：验证你复制出的版本

先做语法和导入检查：

```bash
python -m py_compile src/private_codec/versions/v1/base_network.py
python -m py_compile src/private_codec/versions/v1/wrapped_network.py
python -m py_compile src/private_codec/versions/v1/entrypoints.py
python -c "import torch; import private_codec.versions.v1.entrypoints; print('v1 imports ok')"
```

再把第 6、7 步命令中的 v0 路径换成 v1，验证 Builder、Runner、尺寸和梯度。

必须遵守的返回契约：

- `build_encoder` 和 `build_decoder` 返回 `torch.nn.Module`。
- `run_encoder` 返回 Tensor；按照项目 `latent_spec` 返回 BCHW 或 BCTHW latent。
- `run_decoder` 返回 `[B,3,H,W]` RGB Tensor。
- 训练 forward 不能整体放进 `torch.no_grad()`。
- loss 前不能对学生输出调用 `.detach()`。
- 所有颜色转换、归一化、resize、padding、维度转换和嵌套网络选择由你的版本代码负责。

## 第 13 步：运行项目 probe

Encoder 配置：

```text
configs/local/private_codec_encoder.yaml
```

条件 Decoder 配置：

```text
configs/local/private_codec_conditional_decoder.yaml
```

准备好真实教师权重、数据路径和 CUDA 环境后运行：

```bash
python -m distill_codec.cli probe --config configs/local/private_codec_encoder.yaml
python -m distill_codec.cli probe --config configs/local/private_codec_conditional_decoder.yaml
```

`python -m distill_codec.cli probe` 会真实构造教师和私有学生并执行 forward。只有这一步通过，才能说明真实版本的路径、依赖、输入输出和项目契约已经接通。

## 第 14 步：短训练检查

条件 Decoder 示例：

```bash
python -m distill_codec.cli train --config configs/local/private_codec_conditional_decoder.yaml
```

开始长训练前至少确认：

- loss 是有限值；
- 学生参数存在非零梯度；
- optimizer 确实更新学生参数；
- validation 图片尺寸和颜色正确；
- checkpoint 能保存并恢复。

CPU 的 v0 测试不能替代真实私有网络的 CUDA probe 和短训练。

## 最终逐项检查

- [ ] 已先运行 v0 Encoder 和 Decoder Bridge probe
- [ ] 已把完整 `versions/v0` 复制成一个新版本目录
- [ ] `base_network.py` 已替换为第一个网络文件
- [ ] `wrapped_network.py` 已替换为继承或封装网络
- [ ] `entrypoints.py` 已保留并实现四个 builder/runner 函数
- [ ] Encoder 只从项目接收 RGB，并返回符合 `latent_spec` 的 latent
- [ ] 条件 Decoder 接收原始 LQ RGB 和 DiT latent，并返回 RGB
- [ ] 私有尺寸、颜色、归一化和嵌套逻辑全部留在版本目录内
- [ ] YAML 已从 v0 切换到新版本 import path
- [ ] `builder_kwargs` 和 `runner_kwargs` 已填写真实参数
- [ ] `teacher_reference` 只用于查看教师理论尺寸
- [ ] 新版本 Bridge probe 已通过
- [ ] 真实 CUDA probe 和短训练已确认梯度、输出和 checkpoint

## 相关文档

- [主 README](README.md)
- [FlashVSR 蒸馏教程](FLASHVSR_DISTILL_TUTORIAL.md)
- [新增 Backend 说明](docs/ADDING_A_BACKEND.md)
