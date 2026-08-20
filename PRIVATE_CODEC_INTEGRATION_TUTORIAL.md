# 私有黑盒编解码器 Bridge 接入教程

这个教程对应你的最终接口：工程只把自己已有的 Tensor 交给黑盒；颜色转换、归一化、resize、padding、嵌套网络选择和真实 `forward` 方案全部由你在中间层实现。

## 你只需要填写三个空文件

```text
src/private_codec/base_network.py
src/private_codec/wrapped_network.py
src/private_codec/entrypoints.py
```

- `base_network.py`：粘贴第一个网络源码。
- `wrapped_network.py`：粘贴继承基础网络、封装 `forward` 的第二个源码。
- `entrypoints.py`：填写 Encoder/Decoder 的初始化方式，以及工程 Tensor 到私有网络输入的转换方式。

工程侧公共文件已经实现，不要按网络版本修改：

```text
src/private_codec/bridge.py
src/private_codec/factories.py
src/distill_codec/adapters.py
src/distill_codec/recipes.py
```

## 接口边界

### Encoder

工程调用：

```python
dit_latent = encoder(rgb)
```

这里的 `rgb` 是工程持有的 `[B,3,H,W]` RGB Tensor。Bridge 转交给你的入口：

```python
dit_latent = run_encoder(
    network=network,
    rgb=rgb,
    teacher_reference=teacher_reference,
)
```

工程不会提前转换 RGB。你在 `run_encoder` 中自行进行所有处理并返回 latent。

### 条件 Decoder

工程内部调用顺序保持为：

```python
rgb = conditional_student_decoder(dit_latent, lq_rgb)
```

Bridge 使用关键字参数转交给你的入口：

```python
rgb = run_decoder(
    network=network,
    lq_rgb=lq_rgb,
    dit_latent=dit_latent,
    teacher_reference=teacher_reference,
)
```

所以工程参数顺序和私有函数书写顺序不会混淆。`lq_rgb` 是原始 LQ RGB，不会为了适配教师而先 resize。你自行决定如何将 LQ RGB 和 DiT latent 送入一个或多个嵌套网络，最后返回 `[B,3,H,W]` RGB。

## 第一步：粘贴两个网络文件

把基础网络完整粘贴到：

```text
src/private_codec/base_network.py
```

把继承网络完整粘贴到：

```text
src/private_codec/wrapped_network.py
```

包内导入应使用相对路径：

```python
from .base_network import BaseNetwork
```

不要为了工程接口修改网络内部实现。工程适配逻辑放在下一步的 `entrypoints.py`。

## 第二步：填写四个入口函数

打开空文件：

```text
src/private_codec/entrypoints.py
```

填写以下四个函数。下面只展示接口骨架，类名和参数替换成你的实际实现：

```python
from typing import Any, Mapping

from torch import Tensor, nn

from .wrapped_network import YourWrappedNetwork


def build_encoder(**kwargs: Any) -> nn.Module:
    return YourWrappedNetwork(codec_mode="encoder", **kwargs)


def run_encoder(
    *,
    network: nn.Module,
    rgb: Tensor,
    teacher_reference: Mapping[str, Any],
    **kwargs: Any,
) -> Tensor:
    # 你可以在这里转换颜色、归一化、resize、padding 或选择嵌套网络。
    return network(rgb)


def build_decoder(**kwargs: Any) -> nn.Module:
    return YourWrappedNetwork(codec_mode="decoder", **kwargs)


def run_decoder(
    *,
    network: nn.Module,
    lq_rgb: Tensor,
    dit_latent: Tensor,
    teacher_reference: Mapping[str, Any],
    **kwargs: Any,
) -> Tensor:
    # 这里完全由你定义私有网络真实输入及调用方式。
    return network(dit_latent, lq_rgb)
```

`build_*` 只在组件构造时执行一次，必须返回 `torch.nn.Module`。`run_*` 在每次训练 forward 时执行，必须返回 Tensor，不能整体放在 `torch.no_grad()` 中，也不能在 loss 前 `.detach()`。

## 第三步：填写初始化参数

配置文件已经指向 Bridge：

```text
configs/students/private_codec.yaml
```

Encoder 配置：

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
```

条件 Decoder 配置：

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

- `builder_kwargs` 原样传给 `build_encoder` 或 `build_decoder`。
- `runner_kwargs` 原样传给每次 `run_encoder` 或 `run_decoder`。
- `checkpoint` 可加载标准 Bridge `state_dict`；特殊私有权重格式应在 `build_*` 内处理。
- `input_mode: rgb` 表示项目不对 Encoder 输入做 packed YUV 转换。
- `output_mode: rgb` 表示你的 Decoder 已返回 RGB。

## `teacher_reference` 是什么

`teacher_reference: auto` 让工程把理论上的教师输入输出尺寸作为调试信息传入黑盒。它只是一份普通字典，不参与真实 forward。

Encoder 示例：

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

条件 Decoder 示例：

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

这些 shape 从 `data`、`recipe`、`latent_provider` 和 `latent_spec` 自动推导。未知 batch 写作 `None`。

关键点：它描述教师理论路径，并不要求真实 Tensor 与它一致。FlashVSR 教师可能收到对齐到 GT 尺寸的 LQ，但你的 `run_decoder` 仍收到原始 LQ。Bridge 不会因为参考尺寸执行 resize、padding、裁切或校验。

## 第四步：选择训练配置

Encoder：

```text
configs/local/private_codec_encoder.yaml
```

FlashVSR 条件 Decoder：

```text
configs/local/private_codec_conditional_decoder.yaml
```

条件 Decoder 配置使用：

```yaml
recipe:
  name: flashvsr_decoder_conditional_student
```

它组合冻结的 Wan teacher encoder、冻结的 FlashVSR TCDecoder 和你的 `conditional_student_decoder`。

仓库仍保留 `configs/local/private_codec_decoder.yaml` 与 `configs/local/private_codec_autoencoder.yaml`，用于旧的 Wan 无条件 `student_decoder` 接口。你的 `LQ RGB + DiT latent` Decoder 应使用新的条件配置。

## 第五步：验证和训练

先确认三个私有模块可导入：

```bash
python -c "import private_codec.base_network, private_codec.wrapped_network, private_codec.entrypoints"
```

Encoder probe：

```bash
python -m distill_codec.cli probe \
  --config configs/local/private_codec_encoder.yaml
```

条件 Decoder probe：

```bash
python -m distill_codec.cli probe \
  --config configs/local/private_codec_conditional_decoder.yaml
```

开始训练：

```bash
python -m distill_codec.cli train \
  --config configs/local/private_codec_conditional_decoder.yaml
```

probe 会真实调用你的 builder 和 runner。入口函数未填写、返回类型错误、latent 契约错误或 Decoder 未返回三通道 RGB 时，会直接报告对应错误。

## 增加 V2、V3 或实验版本

不要在 `src/private_codec/factories.py` 中为每个版本增加函数。每个版本自己提供相同的四个入口：

```text
src/private_codec/versions/v2/
├── __init__.py
├── base_network.py
├── wrapped_network.py
└── entrypoints.py
```

V2 的导入路径例如：

```text
private_codec.versions.v2.entrypoints
```

复制一份 student YAML，只修改 builder、runner 和参数：

```yaml
kwargs:
  builder: private_codec.versions.v2.entrypoints:build_decoder
  runner: private_codec.versions.v2.entrypoints:run_decoder
  builder_kwargs:
    variant: v2
  runner_kwargs: {}
```

V3 同理。中央 Bridge 和 factory 始终不感知版本号，因此不同版本可以并存，也可以通过切换 YAML 做实验。

## 最终检查

- [ ] 两个网络源码已放入 `base_network.py` 和 `wrapped_network.py`
- [ ] `wrapped_network.py` 使用正确的包内相对导入
- [ ] `entrypoints.py` 已实现四个函数
- [ ] `build_*` 返回 `nn.Module`
- [ ] `run_encoder` 返回项目 latent
- [ ] `run_decoder` 接收原始 LQ RGB 和 DiT latent，并返回 RGB
- [ ] 网络内部需要的转换全部在私有入口内完成
- [ ] `teacher_reference` 只用于日志或调试，没有用于强制改变真实 Tensor
- [ ] 已运行目标配置的 probe

## 相关文件

- [主 README](README.md)
- [FlashVSR 蒸馏教程](FLASHVSR_DISTILL_TUTORIAL.md)
- [新增 Backend 说明](docs/ADDING_A_BACKEND.md)
