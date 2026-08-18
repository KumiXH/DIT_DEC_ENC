# Adding a Codec Backend

新增模型时优先复用现有 Adapter，不修改 Trainer。

## 1. 写 factory

Factory 必须返回 `torch.nn.Module`：

```python
def create_model(checkpoint=None, **kwargs):
    model = MyModel(**kwargs)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model
```

配置使用 `module.path:create_model`。如果让通用 loader 加载标准 state dict，则把 `checkpoint` 放在组件顶层；如果 factory 自己加载特殊格式，则把路径放进 `kwargs`。

## 2. 选择 Adapter

- Encoder：`kind: encoder`，输入模式是 `rgb`、`rgb_video` 或 `packed_6ch`。
- Decoder：`kind: decoder`，输出模式是 `rgb` 或 `sparse_yuv`。
- DiT condition encoder：`kind: condition_encoder`。

只有模型 forward 不能被这些模式描述时，才新增专用 wrapper。Wrapper 应只处理模型接口和 layout，不实现训练 loop 或 loss。

## 3. 定义契约

为主 latent 定义 `LatentSpec`。条件模块定义 `ConditionSpec`。非默认颜色矩阵、range、UV 采样和上采样方式定义在 `ColorSpec`。

先运行：

```powershell
distill-codec probe --config your_config.yaml
```

Probe 必须通过 shape 和契约检查后再启动长训练。

## 4. 新增 Recipe 的条件

只有数据流或 loss 语义不同才新增 Recipe。仅模型版本、权重或 forward 包装不同，使用 Adapter 和配置解决。

新增 Recipe 时至少测试：

- 一个 CPU forward/backward step；
- 教师参数没有梯度；
- 学生存在非零梯度；
- 条件 key 和 shape 不匹配时明确报错；
- checkpoint 可以恢复。

## 5. 第三方源码

官方源码备份放在 `third_party/<project>/`：

- 保持字节不变；
- 保存许可证；
- `manifest.yaml` 记录不可变 commit、源路径和 SHA256；
- 不提交模型权重；
- 兼容性修改放在 `src/distill_codec/integrations/`。

发布 wheel 时，把需要作为默认 factory 资源的审计副本同步到 `src/distill_codec/vendor/<project>/`，并用测试校验它与 `third_party/<project>/manifest.yaml` 的 SHA256 一致。不要在两个位置分别修改源码。

## 6. 权重与 latent 文件

通用 loader 使用 `torch.load(..., weights_only=True)`。标准 checkpoint 应只包含 tensor、state dict 和基础容器。非 tensor 私有格式应由受信任的 external factory 自行加载，并在文档中明确其安全边界。

非 strict 权重加载必须设置最低参数覆盖率，不能只忽略 missing/unexpected keys。标准 Trainer 只使用配对图像 Dataset；若选择 `latent_provider.type: dataset`，自定义 Dataset 必须为每个样本提供 latent，并配套自定义 Trainer 或 probe。
