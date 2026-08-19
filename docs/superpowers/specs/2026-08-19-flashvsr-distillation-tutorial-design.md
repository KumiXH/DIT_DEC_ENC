# FlashVSR 蒸馏教程设计

_面向 Linux 用户的 `LQ_proj_in` 与 `TCDecoder` 分步训练教程规格，2026-08-19_

---

## 📋 目标

新增仓库根目录文档 `FLASHVSR_DISTILL_TUTORIAL.md`。读者已经完成 Python、PyTorch 和项目依赖安装，教程从进入仓库开始，带领读者分别完成：

1. 验证 CLI 和 mock 链路
2. 准备同构的 PNG LQ/GT 数据目录
3. 放置并检查 Wan VAE、`LQ_proj_in` 和 `TCDecoder` 教师权重
4. 接入私有 `student_condition_encoder` 和 `conditional_student_decoder` factory
5. 创建两份本地 YAML
6. 分别执行 `probe`、训练、查看日志、查看验证图和恢复训练
7. 找到最终学生 checkpoint，并理解 checkpoint 中保存的内容

教程不介绍 Linux、CUDA、Python 虚拟环境或 PyTorch 的安装。

## 🔄 教学流程

```mermaid
flowchart LR
    accTitle: FlashVSR Distillation Tutorial Flow
    accDescr: The reader verifies the mock pipeline, prepares data and weights, then trains LQ_proj_in and TCDecoder as two independent distillation tasks.

    enter_repo(["进入仓库"]) --> mock_check["运行 mock 检查"]
    mock_check --> prepare_inputs["准备数据与权重"]
    prepare_inputs --> connect_student["接入黑盒学生"]
    connect_student --> train_lq["蒸馏 LQ_proj_in"]
    train_lq --> train_decoder["蒸馏 TCDecoder"]
    train_decoder --> inspect_outputs(["检查权重与结果"])

    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef success fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class mock_check,prepare_inputs,connect_student,train_lq,train_decoder process
    class inspect_outputs success
```

`LQ_proj_in` 与 `TCDecoder` 是两个独立训练任务。教程不会暗示前一个任务必须训练完成后才能启动后一个任务；按顺序编排只是为了教学清晰。

## 📚 文档结构

教程采用“执行、预期结果、产物、成功判断、失败检查”的固定格式。主要章节为：

- 开始前的文件清单
- 一分钟理解两条蒸馏链路
- 第一次运行 mock
- 准备 PNG 配对数据
- 准备和校验三个教师权重
- 实现或替换黑盒学生 factory
- 创建 `configs/local/flashvsr_lq_proj.yaml`
- 蒸馏 `LQ_proj_in`
- 创建 `configs/local/flashvsr_tcdecoder.yaml`
- 蒸馏条件 `TCDecoder`
- 查看 JSONL、TensorBoard、验证图和 checkpoint
- 中断恢复和常用覆盖参数
- YAML 参数完整解释
- 常见错误排查
- 命令速查表

每条命令的预期输出只展示稳定字段和结构。loss、耗时、显存和 checkpoint 大小会标注为示意值，不承诺固定数值。

## ⚙️ 配置边界

主线只使用仓库内置 snapshot adapter：

- `configs/teachers/flashvsr_snapshot.yaml`
- `configs/teachers/wan_snapshot.yaml`
- `configs/students/private_blackbox.yaml`

`LQ_proj_in` 主配置必须包含 `teacher_condition_encoder` 和直接声明的 `student_condition_encoder`。

条件 `TCDecoder` 主配置必须包含：

- Wan `teacher_encoder`，作为冻结的主 latent provider
- FlashVSR `tc_decoder`，作为教师解码器
- `conditional_student_decoder`，作为待训练学生

教程明确指出此工程不运行或训练 DiT。`TCDecoder` 训练使用 Wan Encoder 生成的冻结 latent，LQ RGB 作为条件输入。

## 🧾 YAML 参数解释

参数字典覆盖以下顶层字段，并说明数据类型、作用、常用值、能否修改和修改风险：

| 顶层字段 | 解释范围 |
| --- | --- |
| `includes` | 合并顺序、相对路径、覆盖规则 |
| `latent_spec` | family、channels、layout、空间和时间下采样、normalization |
| `color` | YUV 矩阵、范围、打包顺序、色度位置和上采样 |
| `recipe` | recipe 名称、source、各 loss 权重 |
| `components` | backend、factory、checkpoint、kwargs、freeze、adapter |
| `condition_spec` | BNC 布局、feature dim、consumer 和时空下采样 |
| `latent_provider` | teacher encoder、source、缓存方案边界 |
| `data` | LQ/GT 根目录、尺寸约束和配对规则 |
| `trainer` | device、batch、优化器、AMP、步数、验证、保存、日志和恢复契约 |
| `run` | 输出目录和随机种子 |

参数解释以当前代码实现为准，不把尚未实现或仅预留的能力写成可用功能。

## ✅ 验证策略

实现教程时增加文档一致性测试，至少验证：

- 教程引用的仓库内文件全部存在
- 两段完整 YAML 可以被 PyYAML 解析
- 两段 YAML 经过 `load_config` 和 `preflight_config`
- 所有训练命令指向教程中已创建的本地 YAML
- `LQ_proj_in` 配置包含 `student_condition_encoder`
- `TCDecoder` 配置包含正确的三个 include 和 `teacher_encoder` latent provider
- 教程没有把真实 `probe` 描述为无需加载权重
- Linux 命令不包含 PowerShell 或 Windows 路径

最后运行完整 pytest、Ruff、mypy、Bash 语法检查和 `git diff --check`。

## 🚫 非目标

- 不下载或上传真实模型权重
- 不修改训练网络、loss 或 snapshot adapter
- 不新增可提交的真实训练 YAML；真实路径和私有 factory 保留在用户本地配置
- 不训练 DiT
- 不承诺训练质量、收敛步数或显存占用
- 不覆盖完整 FlashVSR 上游源码接入方式
