# 私有黑盒编解码器接入教程

这份教程就是你的第二份 README。按顺序填写即可，不需要修改训练器、Recipe、公共 Bridge 或中央 factory。

## 当前完成和验证状态

框架已经验证的部分：

- Encoder 公共调用固定为 `encoder(rgb)`，项目只传 `[B,3,H,W]` RGB。
- 条件 Decoder 公共调用固定为 `conditional_student_decoder(dit_latent, lq_rgb)`。
- 私有 Decoder 入口收到关键字参数 `network`、原始 `lq_rgb`、`dit_latent` 和 `teacher_reference`。
- `teacher_reference` 只提供理论教师输入输出尺寸，不 resize、padding、裁切或修改真实 Tensor。
- 私有 RGB Encoder 可以返回 `BCHW` 或教师时间轴对应的 `BCTHW` latent。
- 新网络版本只需要增加版本目录并切换 YAML 中的 builder/runner，不修改 `factories.py`。
- 当前 CPU Conda 环境中，私有 Bridge、配置、Adapter 和教程直接相关测试已经通过。
- 全量测试只有 `tests/test_trainer.py::test_multi_component_resume_is_stable_across_python_hash_seeds` 会触发已在未修改 `main` 上复现的 OpenMP 子进程环境问题；排除该项后的全量测试已经通过。

真实私有网络尚未验证，因为下面三个文件仍是 0 字节空文件，等待你粘贴源码：

```text
src/private_codec/base_network.py
src/private_codec/wrapped_network.py
src/private_codec/entrypoints.py
```

因此，“框架桥接已通过测试”和“你的真实模型已经成功训练”是两件事。后者必须在你填完文件、准备好权重、数据和 CUDA 环境后运行 probe/训练才能确认。

## 第 0 步：确认使用哪个 Python

当前机器的 Conda base 环境存在，但 `python` 和 `conda` 没有加入当前 PowerShell 的 `PATH`。先在仓库根目录执行：

```powershell
$CodecConda = 'C:\Users\xh932\anaconda3\Scripts\conda.exe'
$env:PYTHONPATH = (Resolve-Path 'src').Path
& $CodecConda run --no-capture-output -n base python -c "import numpy, torch, pytest, sys; print(sys.executable); print(torch.__version__); print('cuda=', torch.cuda.is_available())"
```

当前核验到的是 Python 3.13.9、PyTorch 2.11.0 CPU 版，`torch.cuda.is_available()` 为 `False`。这个环境足以编辑代码和运行框架单元测试，但不能验证真实 CUDA 训练或 Flash Attention。

`PYTHONPATH` 让尚未执行 `pip install -e .` 的源码仓库也能导入 `private_codec` 和 `distill_codec`。后面的导入、Bridge、probe 和训练命令请在这个 PowerShell 会话中继续执行。

以后本教程中的 `python ...` 命令，在当前 PowerShell 中都可以写成：

```powershell
& $CodecConda run --no-capture-output -n base python ...
```

如果你进入了另一个已经激活的 CUDA Conda 环境，也可以直接使用该环境的 `python`。

## 第 1 步：填写基础网络

把第一个 `.py` 的完整网络源码粘贴到：

```text
src/private_codec/base_network.py
```

这个文件只保存网络本身。暂时不要为了适配工程而改它的输入、输出或 `forward`。

填完先检查 Python 语法：

```powershell
& $CodecConda run --no-capture-output -n base python -m py_compile src/private_codec/base_network.py
```

命令退出码为 0 且没有报错，只说明语法正确，不说明权重或 forward 已经正确。

## 第 2 步：填写继承封装网络

把继承基础网络、修改初始化参数或外套一层 `forward` 的第二个 `.py` 粘贴到：

```text
src/private_codec/wrapped_network.py
```

包内导入使用相对路径。例如：

```python
from .base_network import BaseNetwork


class YourWrappedNetwork(BaseNetwork):
    ...
```

然后检查语法和导入：

```powershell
& $CodecConda run --no-capture-output -n base python -m py_compile src/private_codec/wrapped_network.py
& $CodecConda run --no-capture-output -n base python -c "import numpy, torch; import private_codec.base_network, private_codec.wrapped_network; print('network imports ok')"
```

如果这里提示缺少第三方包，应该把依赖安装到真实训练环境，而不是修改公共 Bridge。

## 第 3 步：填写唯一的中间层

打开：

```text
src/private_codec/entrypoints.py
```

这里是你唯一需要实现“工程 Tensor 如何进入黑盒网络”的地方。填写以下四个函数，并把类名、初始化参数和 forward 调用替换成你的真实实现：

```python
from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor, nn

from .wrapped_network import YourWrappedNetwork


def build_encoder(**kwargs: Any) -> nn.Module:
    # 这里只负责初始化 Encoder；需要时也可以在这里读取特殊格式权重。
    return YourWrappedNetwork(codec_mode="encoder", **kwargs)


def run_encoder(
    *,
    network: nn.Module,
    rgb: Tensor,
    teacher_reference: Mapping[str, Any],
    **kwargs: Any,
) -> Tensor:
    # 项目只提供原始 RGB。颜色转换、归一化、resize、padding、维度转换、
    # 嵌套子网络选择都由你在这里完成。
    private_input = rgb
    dit_latent = network(private_input)
    return dit_latent


def build_decoder(**kwargs: Any) -> nn.Module:
    # 这里只负责初始化 Decoder。
    return YourWrappedNetwork(codec_mode="decoder", **kwargs)


def run_decoder(
    *,
    network: nn.Module,
    lq_rgb: Tensor,
    dit_latent: Tensor,
    teacher_reference: Mapping[str, Any],
    **kwargs: Any,
) -> Tensor:
    # lq_rgb 是原始 LQ RGB，dit_latent 是项目当前的 DiT latent。
    # 二者如何变换、融合及送入多少层嵌套网络，完全由你决定。
    output_rgb = network(dit_latent, lq_rgb)
    return output_rgb
```

必须遵守的返回契约：

- `build_encoder` 和 `build_decoder` 必须返回 `torch.nn.Module`。
- `run_encoder` 必须返回 Tensor。`latent_spec.layout: BCHW` 时返回 `[B,C,Hl,Wl]`；`BCTHW` 时返回 `[B,C,Tl,Hl,Wl]`。
- `run_decoder` 必须返回 `[B,3,H,W]` RGB Tensor。
- 训练 forward 不能整体放进 `torch.no_grad()`，返回值不能在 loss 前 `.detach()`，否则学生网络没有梯度。
- `teacher_reference` 可以打印或断言辅助调试，但不要用它强行修改项目传入的真实 Tensor。

填完执行：

```powershell
& $CodecConda run --no-capture-output -n base python -m py_compile src/private_codec/entrypoints.py
& $CodecConda run --no-capture-output -n base python -c "import numpy, torch; import private_codec.entrypoints; print('entrypoints import ok')"
```

## 第 4 步：理解项目实际传入什么

Encoder 的完整调用链是：

```text
项目 RGB [B,3,H,W]
  -> EncoderAdapter(input_mode=rgb)
  -> PrivateEncoderBridge
  -> run_encoder(network=..., rgb=..., teacher_reference=...)
  -> 你的 latent
```

项目不会提前替你转 YUV、归一化或 resize。

条件 Decoder 的完整调用链是：

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

工程内部的位置参数顺序是 `(dit_latent, lq_rgb)`，Bridge 到你的私有入口后全部变成有名字的关键字参数，因此不会因为你把 `lq_rgb` 写在前面而传反。

## 第 5 步：查看教师理论尺寸

在 `configs/students/private_codec.yaml` 中保留：

```yaml
teacher_reference: auto
```

Encoder 的参考信息类似：

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
            "layout": "BCTHW",
            "shape": [None, 16, 2, 32, 32],
        },
    },
}
```

条件 Decoder 的参考信息类似：

```python
{
    "role": "conditional_decoder",
    "inputs": {
        "lq_rgb": {
            "layout": "BCHW",
            "shape": [None, 3, 256, 256],
            "source": "teacher_condition",
        },
        "dit_latent": {
            "layout": "BCHW",
            "shape": [None, 16, 32, 32],
        },
    },
    "outputs": {
        "rgb": {
            "layout": "BCHW",
            "shape": [None, 3, 256, 256],
        },
    },
}
```

这些尺寸从最终合并后的 `data`、`recipe`、教师 adapter、`latent_provider` 和 `latent_spec` 推导。它们是教师理论路径的调试参考，不是对你私有网络内部处理方式的命令。

特别注意：FlashVSR 教师可能收到对齐到 GT 尺寸的 LQ，而你的 `run_decoder` 收到的仍是 `batch.lq_rgb` 中的原始 LQ RGB。

## 第 6 步：填写初始化和 forward 参数

打开：

```text
configs/students/private_codec.yaml
```

Encoder 已经连接到公共 factory：

```yaml
student_encoder:
  backend: external
  factory: private_codec.factories:create_encoder
  checkpoint: null
  teacher_reference: auto
  kwargs:
    builder: private_codec.entrypoints:build_encoder
    runner: private_codec.entrypoints:run_encoder
    builder_kwargs: {}
    runner_kwargs: {}
  adapter:
    kind: encoder
    input_mode: rgb
    latent_temporal_frames: teacher
```

条件 Decoder 已经连接到公共 factory：

```yaml
conditional_student_decoder:
  backend: external
  factory: private_codec.factories:create_conditional_decoder
  checkpoint: null
  teacher_reference: auto
  kwargs:
    builder: private_codec.entrypoints:build_decoder
    runner: private_codec.entrypoints:run_decoder
    builder_kwargs: {}
    runner_kwargs: {}
  adapter:
    kind: decoder
    output_mode: rgb
    accepts_condition: true
```

你通常只修改以下位置：

- `builder_kwargs`：初始化网络时传给 `build_encoder`/`build_decoder`，例如模型版本、通道数、配置路径和私有权重路径。
- `runner_kwargs`：每次 forward 时传给 `run_encoder`/`run_decoder`，例如运行模式或可微的处理选项。
- `checkpoint`：仅在你的 Bridge 整体可以直接用标准 `state_dict` 加载时填写；特殊权重格式建议在 `build_*` 中读取，并保持这里为 `null`。
- `latent_temporal_frames: teacher`：只定义 `BCTHW` latent 应匹配教师的理论时间轴，不改变传给 `run_encoder` 的 4D RGB。也可以填写正整数；它与 `teacher_reference` 调试开关相互独立。

示例：

```yaml
builder_kwargs:
  config_path: D:/models/private_codec/v1.yaml
  weights_path: D:/models/private_codec/v1.pth
  variant: v1
runner_kwargs:
  normalize_input: true
```

不要给每个模型版本在 `factories.py` 新增一个函数。

## 第 7 步：先单独验证 Bridge

在运行完整教师模型前，可以先用随机 Tensor 验证你的 builder/runner 是否能调用。下面的尺寸只是调试输入，请按你的实际模型改：

```powershell
& $CodecConda run --no-capture-output -n base python -c "import numpy, torch; from private_codec.factories import create_encoder; m=create_encoder(builder='private_codec.entrypoints:build_encoder', runner='private_codec.entrypoints:run_encoder', builder_kwargs={}, runner_kwargs={}, teacher_reference={'role':'encoder','inputs':{'rgb':{'layout':'BCHW','shape':[None,3,256,256]}},'outputs':{'latent':{'layout':'BCHW','shape':[None,16,32,32]}}}); y=m(torch.randn(1,3,256,256)); print('encoder output=', tuple(y.shape), y.dtype)"
```

```powershell
& $CodecConda run --no-capture-output -n base python -c "import numpy, torch; from private_codec.factories import create_conditional_decoder; m=create_conditional_decoder(builder='private_codec.entrypoints:build_decoder', runner='private_codec.entrypoints:run_decoder', builder_kwargs={}, runner_kwargs={}, teacher_reference={'role':'conditional_decoder','inputs':{'lq_rgb':{'layout':'BCHW','shape':[None,3,256,256]},'dit_latent':{'layout':'BCHW','shape':[None,16,32,32]}},'outputs':{'rgb':{'layout':'BCHW','shape':[None,3,256,256]}}}); y=m(torch.randn(1,16,32,32), torch.randn(1,3,256,256)); print('decoder output=', tuple(y.shape), y.dtype)"
```

这一步成功说明你的私有 builder 和 runner 可导入、可构造、能返回 Tensor。它还没有验证教师模型、数据集、loss 或训练反向传播。

## 第 8 步：选择项目配置并运行 probe

Encoder 使用：

```text
configs/local/private_codec_encoder.yaml
```

你的 `LQ RGB + DiT latent -> RGB` 条件 Decoder 使用：

```text
configs/local/private_codec_conditional_decoder.yaml
```

不要误用旧的 `configs/local/private_codec_decoder.yaml`。旧配置对应不接收 LQ 的 Wan 无条件 `student_decoder`。

准备好真实教师权重、数据路径和 CUDA 环境后运行：

```powershell
python -m distill_codec.cli probe --config configs/local/private_codec_encoder.yaml
python -m distill_codec.cli probe --config configs/local/private_codec_conditional_decoder.yaml
```

`python -m distill_codec.cli probe` 会真实构造教师、私有学生并执行 forward。只有这一步通过，才能说明真实模型的路径、依赖、输入输出和项目契约已经接通。

## 第 9 步：开始训练

条件 Decoder 示例：

```powershell
python -m distill_codec.cli train --config configs/local/private_codec_conditional_decoder.yaml
```

开始长训练前至少确认：

- loss 是有限值；
- 学生参数存在非零梯度；
- optimizer 确实更新学生参数；
- validation 图片尺寸和颜色正确；
- checkpoint 能保存并恢复。

当前 CPU 环境不能替代这一步的真实 CUDA 验证。

## 第 10 步：增加 V2、V3 或实验版本

新增 V2 时创建：

```text
src/private_codec/versions/v2/
├── __init__.py
├── base_network.py
├── wrapped_network.py
└── entrypoints.py
```

V2 的 `entrypoints.py` 仍实现同样四个函数。然后复制一份 student YAML，只修改路径和参数：

```yaml
kwargs:
  builder: private_codec.versions.v2.entrypoints:build_decoder
  runner: private_codec.versions.v2.entrypoints:run_decoder
  builder_kwargs:
    variant: v2
  runner_kwargs: {}
```

V3 使用 `private_codec.versions.v3.entrypoints`，以此类推。中央 `src/private_codec/factories.py` 和 `src/private_codec/bridge.py` 不需要知道任何版本号。

## 你不需要修改的文件

只要外部接口没有改变，就不要按版本修改：

```text
src/private_codec/bridge.py
src/private_codec/factories.py
src/distill_codec/adapters.py
src/distill_codec/recipes.py
src/distill_codec/trainer.py
```

## 最终逐项检查

- [ ] `base_network.py` 已粘贴基础网络并通过 `py_compile`
- [ ] `wrapped_network.py` 已粘贴继承网络并能导入
- [ ] `entrypoints.py` 已实现四个 builder/runner 函数
- [ ] `run_encoder` 只接收项目 RGB，并返回符合 `latent_spec` 的 BCHW/BCTHW latent
- [ ] `run_decoder` 接收原始 LQ RGB 和 DiT latent，并返回三通道 RGB
- [ ] 所有私有颜色、尺寸、归一化和嵌套网络逻辑都在 `entrypoints.py` 或私有网络内部
- [ ] `builder_kwargs` 和 `runner_kwargs` 已填写真实参数
- [ ] `teacher_reference` 只用于查看教师理论尺寸
- [ ] 随机 Tensor Bridge 检查已通过
- [ ] 真实 Encoder/Decoder probe 已在目标 CUDA 环境通过
- [ ] 短训练已确认梯度、optimizer、验证图和 checkpoint

## 相关文档

- [主 README](README.md)
- [FlashVSR 蒸馏教程](FLASHVSR_DISTILL_TUTORIAL.md)
- [新增 Backend 说明](docs/ADDING_A_BACKEND.md)
