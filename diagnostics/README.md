# 验证记录

本目录保存测量证据与复测脚本。配置以 [项目 README](../README.md) 和
[图片 API](../docs/image-api.md) 为准；耗时只代表对应设备、权重和输入。
不同配置的记录用于对照，文件名中的 `final` 不表示正在运行的服务状态。

## GLM 推理

设备为 M3 Ultra / 512 GiB；mlx-vlm 0.7.1、MLX 0.32.2。
下表使用微调版 `GLM-5.3-Flash-FineTunning-MXFP8` 及匹配 MTP，测量日期为
2026-09-22；权重与基础接口记录为 2026-09-20。

| 检查 | 结论 | 证据 |
| --- | --- | --- |
| 权重与接口 | 主模型约 307.7 GiB、MTP 约 7.2 GiB；MXFP8、group_size=32、bits=8；四个未融合张量抽查 RMS 误差 2.4562%–2.5370% | [接口](finetuning-smoke.json)、[思考流](finetuning-reasoning.json) |
| 上下文配置 | 有效上限 393216；配置及短推理检查，不是完整 384K 生成 | [记录](context-384k.json) |
| 预填充词表投影 | 约 8K / 16K 冷输入的平均预填充分别由 20.844 / 42.510 s 降至 20.356 / 41.512 s，吞吐均提高约 2.4% | [汇总](prefill-summary.json)、[启用前](prefill-before.json)、[启用后](prefill-after.json) |
| MTP 块长度 2 | 八组合计解码 21.41 → 22.41 token/s，约 +4.7%；两组深度思考约慢 2% | [汇总](mtp-summary.json)、[上游默认](mtp-default.json)、[块长度 2](mtp-block2.json) |
| MTP 块上限 4 | 合计 19.16 token/s，比上游默认慢约 10.5%，未采用 | [记录](mtp-block4.json) |
| 2 GiB 分配池 | 22 次请求空闲记录均在限额内，20 次保留约 0.63–0.64 GiB；未测得明确提速 | [汇总](allocator-summary.json)、[空闲清空](allocator-before.json)、[有限保留](allocator-after.json) |

预填充对照含 2K / 8K / 16K 冷输入、16K 续接和代码生成，共 10 组完整输出一致。
MTP 对照含代码、中文、深度思考和约 8.7K 输入后的生成，各两种温度、384 token 预算；
八组输出一致。[默认复测](mtp-default-repeat.json) 与 [块长度 2 复测](mtp-block2-repeat.json)
确认了代码、中文和思考样例的相同趋势；[默认并发](mtp-default-batch.json) 与
[块长度 2 并发](mtp-block2-batch.json) 的四个输出也一致。

分配池对照含 18 个短请求和两组长输入冷/热请求，输出及 APC 命中 token 数一致。
三组平均总耗时分别为 3.511 → 3.511 s、26.406 → 26.452 s、4.643 → 4.619 s，差异均小于 1%。
[并发对照](allocator-batch.json) 另验证了四个输出一致。
这些固定样例不能代表所有任务；单次配置切换的耗时差异不构成统计显著性结论。
原始 `peak_memory` 是进程历史峰值；比较前须统一进程内已执行的负载，不能当作单次请求峰值。

上述对照均保持 temperature=0 / 1、top_p=0.95、seed=874；深度思考使用 `max`，
其余使用 `low`。预填充分块为 2048 token，APC 内存预算为 32 GiB，磁盘层关闭。
预填充实验使用上游 MTP 策略；分配池实验使用块长度 2。

回归覆盖 BF16 / MXFP8 缓存、采样 logits 和 token 一致性，左右填充、非有限值、
请求取消及真实分配池超额清理。各次完整测试日志见 [预填充](prefill-tests.txt)、
[分配池](allocator-tests.txt)；在线验证见 [六项接口检查](allocator-smoke.txt) 和
[四项生命周期检查](allocator-lifecycle.json)。这些日志对应各自测量时的代码版本。

## 复测

先设置 `API_KEY` 并确认服务空闲。性能对照在两次运行间重启 GLM 后端，切换配置并
清除内存 APC；两次使用相同 `--run-id`。脚本不清理共享 APC。
`--compare` 检查输入和完整输出；不要把已经热缓存的输入当作冷输入重测。

| 脚本 | 对照启动配置 | 参数 |
| --- | --- | --- |
| `benchmark_prefill.py` | `GLM_PREFILL_SKIP_UNUSED_LOGITS=0` / `1` | `--run-id NAME --output PATH [--compare PATH]` |
| `benchmark_decode.py` | `DRAFT_BLOCK_SIZE=auto` / `2` | `--run-id NAME --label NAME --output PATH [--compare PATH]` |
| `check_mtp_batch.py` | 与 MTP 对照配置相同 | `--run-id NAME --label NAME --output PATH [--compare PATH]` |
| `benchmark_allocator.py` | `GLM_ALLOCATOR_CACHE_GB=0` / `2` | `--run-id NAME --label NAME --output PATH [--compare PATH]` |

例如，分别在两种分配池配置启动后执行：

```sh
~/.venvs/mlx-vlm/bin/python diagnostics/benchmark_allocator.py \
  --run-id my-comparison --label clear --output /tmp/allocator-before.json
~/.venvs/mlx-vlm/bin/python diagnostics/benchmark_allocator.py \
  --run-id my-comparison --label bounded --output /tmp/allocator-after.json \
  --compare /tmp/allocator-before.json
```

功能检查同样使用显式输出位置，避免覆盖本目录的测量证据：

```sh
~/.venvs/mlx-vlm/bin/python diagnostics/check_lifecycle_service.py \
  --output /tmp/lifecycle.json
~/.venvs/mlx-vlm/bin/python diagnostics/check_long_service.py \
  --tokens 20000 --check-budget --output /tmp/long-context.json
```

长输入检查要求 APC 开启，`APC_EXACT_MAX_TOKENS=0` 或不小于目标长度。
`--check-budget` 读取服务的有效上下文上限，分别验证恰好达到上限和超出一个 token。
自定义模型应同时指定 `--model` 和匹配的 `--model-path`；其余参数见各脚本 `--help`。

## GLM 长上下文与生命周期

本节使用原版 `GLM-5.3-Flash-MXFP8`，不是微调版；测量日期为 2026-09-19。

| 检查 | 结论 / 证据 |
| --- | --- |
| 199947 token 冷输入 / 199977 token 续接 | 三处检索正确，续接命中 198656 token；[记录](service-long-context.json) |
| 约 256K Responses 续接 | 三处检索正确；[记录](service-responses-256k.json) |
| 262027 token 冷输入 / 262057 token 续接 | 1127.14 s / 15.02 s，续接命中 260096 token；[结果及预算边界](service-262144.json) |
| 262K Responses 流式 | [事件与终态](service-262144-stream.json) |
| 262047 token 冷输入中断连 | 7.48 s 后取消，无生成 token，空闲 active 无增长；[记录](service-262144-cold-disconnect.json) |
| 正常完成、断连、显式取消 | [生命周期](lifecycle-service.json) |
| 约 20K 冷输入 / 续接与输入释放 | [记录](service-final-cleanup.json) |

取消耗时受计算块影响，不是响应时间上界。空闲 `active` 包含权重与 APC；
这些记录使用空闲清池策略，不能用 `allocator_cache=0` 判断有限保留策略是否正常。
稀疏注意力的大地址边界见 [探针](address-overflow.txt)。缩小缓存状态的测试不能替代
完整模型生成；完整 1M 推理与四路满上下文未验证。

## Qwen-Image

mlx-vlm 0.7.2、MLX 0.32.2；以下样例使用原版 encoder，测量日期为 2026-09-22。
原始权重复制校验见 [SHA-256 清单](qwen-image-copy.json)。

| 检查 | 结论 / 证据 |
| --- | --- |
| 文生图 1024×1024 / 40 步 | 首次约 201.73 s，含加载且 GLM 同时处理请求；[记录](qwen-image-smoke.json)、[图片](qwen-image-1024.png) |
| 单图编辑 512×512 / 40 步 | 已加载模型约 38.73 s；[记录](qwen-edit-single.json)、[图片](qwen-edit-single.png) |
| 双图编辑 512×512 / 40 步 | 含加载约 41.44 s；[记录](qwen-edit-final.json)、[图片](qwen-edit-multi-512.png) |
| 双图编辑 1024×1024 / 40 步 | 约 158 s，样例偏锐化；[记录](qwen-edit-http.json)、[图片](qwen-edit-multi.png) |
| 参考编码分辨率 1024 | 约 207 s，未消除偏锐化；[记录](qwen-edit-multi-reference1024.json)、[图片](qwen-edit-multi-reference1024.png) |
| 10 张输入、CFG、断连及聊天并行响应 | [HTTP 检查](qwen-edit-http.json)；10 张及 CFG 仅用 2 步验证路径 |
| 编辑后卸载 | 测试空闲期为 3 s，MLX active 降为 0.00 GiB；[记录](qwen-edit-final.json) |

## Qwen-Image / Heretic encoder

mlx-vlm 0.7.2、MLX 0.32.2；BF16 Heretic encoder 配合原版 DiT、VAE 和 processor。
2026-09-22 使用临时组件目录和 `ImageEngine` 测量；输出为 512×512、40 步、seed=42，
耗时不含模型初始化。正式服务的路径及探活见 [启用记录](qwen-heretic-active.json)。

| 检查 | 结论 / 证据 |
| --- | --- |
| 文件与配置 | 14 个源文件 SHA-256 校验；[安装清单](qwen-heretic-install.json) |
| 参数加载 | 750 个参数名称和形状匹配，并逐值检查一个投影权重与源权重一致；[验证](qwen-heretic-validation.json) |
| 文生图 / 单图编辑 / 双图编辑 | 39.75 / 39.10 / 40.68 s；[文生图](qwen-heretic-generation.png)、[单图](qwen-heretic-single.png)、[双图](qwen-heretic-multi.png) |
| 资源释放 | `ImageEngine.close()` 后 MLX active 约 1 KiB；[验证](qwen-heretic-validation.json) |

图片数值测试使用缩小的模型，对比缓存计算与完整注意力计算。PNG 供人工检查，
少步数路径检查和单组样例不能代表全面画质或行为评估。
