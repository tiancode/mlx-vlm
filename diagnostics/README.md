# 验证记录

这里保留可复查的测试输入、输出和测量证据。配置与操作说明以
[项目 README](../README.md) 和 [图片 API](../docs/image-api.md) 为准。
耗时与内存数据只代表对应测量环境，不是接口承诺。

## GLM 微调权重与接口

环境：M3 Ultra / 512 GiB，mlx-vlm 0.7.1、MLX 0.32.2；权重为
`GLM-5.3-Flash-FineTunning-MXFP8` 及匹配的 MTP。测量日期：2026-09-20。

| 检查 | 结果 / 记录 |
| --- | --- |
| 转换产物 | 主模型约 307.7 GiB，MTP 约 7.2 GiB；MXFP8 group_size=32、bits=8 |
| 四个未融合张量抽查 | RMS 相对误差 2.4562%–2.5370%，低于脚本的 5% 阈值 |
| 文本、图片理解、Responses、Messages | [基础接口](finetuning-smoke.json) |
| 默认思考流事件 | [reasoning / message 声明和索引](finetuning-reasoning.json) |
| 384K 上下文配置与简短推理 | [配置检查](context-384k.json)，不含完整 384K 生成 |

## GLM 长上下文与生命周期

以下完整模型结果来自原版 `GLM-5.3-Flash-MXFP8`，不是微调权重的长上下文基准。
环境同上；测量日期：2026-09-19。

| 检查 | 结果 / 记录 |
| --- | --- |
| 199947 token 冷输入 / 199977 token 续接 | 三处检索正确，续接命中 198656 token；[结果](service-long-context.json) |
| 约 256K Responses 续接 | 三处检索正确；[结果](service-responses-256k.json) |
| 262027 token 冷输入 / 262057 token 续接 | 1127.14 s / 15.02 s，续接命中 260096 token；[结果及预算边界](service-262144.json) |
| 262K Responses 流式 | [事件与终态](service-262144-stream.json) |
| 262047 token 冷输入中断连 | 7.48 s 后取消，无生成 token，空闲 active 无增长；[结果](service-262144-cold-disconnect.json) |
| 正常完成、断连、显式取消 | [生命周期检查](lifecycle-service.json) |
| 请求输入映射清理后的冷输入 / 续接 | [约 20K 检索与缓存结果](service-final-cleanup.json) |

取消在计算块结束后生效，测量耗时不是取消上界。空闲 `active` 仍包含权重和 APC；
`allocator_cache=0` 表示空闲分配池已释放。

稀疏注意力内核的 head / batch 地址乘积会越过 32 位范围，
[地址探针输出](address-overflow.txt) 展示了边界。`long_context_patch.py` 在乘法前
将操作数提升到 `size_t`，同时检查原始 logits 的非有限值，并为磁盘缓存隔离命名空间。
测试中的缓存往返使用缩小的状态维度，不能替代完整模型生成验证。
完整 1M 推理与四路满上下文未验证。

复查命令（会对运行中的服务发出长推理请求）：

```sh
~/.venvs/mlx-vlm/bin/python diagnostics/probe_sparse_address.py
~/.venvs/mlx-vlm/bin/python diagnostics/check_long_service.py --tokens 200000
~/.venvs/mlx-vlm/bin/python diagnostics/check_lifecycle_service.py
```

## Qwen-Image

以下记录使用原版 encoder：mlx-vlm 0.7.2、MLX 0.32.2，原始精度 Qwen-Image-2.1。
测量日期：2026-09-22。权重复制校验见 [SHA-256 清单](qwen-image-copy.json)。

| 检查 | 结果 / 记录 |
| --- | --- |
| 文生图 1024×1024 / 40 步 | 首次约 201.73 s，含加载且 GLM 同时处理请求；[记录](qwen-image-smoke.json)、[图片](qwen-image-1024.png) |
| 单图编辑 512×512 / 40 步 | 已加载模型约 38.73 s；[记录](qwen-edit-single.json)、[图片](qwen-edit-single.png) |
| 双图编辑 512×512 / 40 步 | 含加载约 41.44 s；[记录](qwen-edit-final.json)、[图片](qwen-edit-multi-512.png) |
| 双图编辑 1024×1024 / 40 步 | 约 158 s，样例偏锐化；[结果](qwen-edit-http.json)、[图片](qwen-edit-multi.png) |
| 提高参考编码分辨率至 1024 | 约 207 s，未消除偏锐化；[结果](qwen-edit-multi-reference1024.json)、[图片](qwen-edit-multi-reference1024.png) |
| 10 张输入、CFG、文生图回归、断连、聊天并行响应 | [HTTP 检查](qwen-edit-http.json)；10 张及 CFG 使用 2 步，仅验证执行路径 |
| 编辑后实际卸载 | 测试使用 3 s 空闲期，MLX active 降为 0.00 GiB；默认空闲期为 300 s；[结果](qwen-edit-final.json) |

GPU 数值测试对比缓存计算与完整注意力计算，不加载整套权重。
PNG 样例供人工检查；少步数的尺寸和执行路径检查不代表最终画质。

## Qwen-Image / Heretic encoder

环境：mlx-vlm 0.7.2、MLX 0.32.2，BF16 Heretic encoder，原版 DiT、VAE 和 processor。
模型组件在临时目录中组合后直接调用服务使用的 `ImageEngine` 验证，测量日期：2026-09-22。
输出均为 512×512、40 步、seed=42；耗时从模型初始化完成后计算。

| 检查 | 结果 / 记录 |
| --- | --- |
| 文件完整性与配置兼容 | 14 个源文件逐一校验 SHA-256；[安装清单](qwen-heretic-install.json) |
| encoder 参数 | 750 个参数名称和形状全部对应，并逐值检查一个投影权重与 Heretic 源权重一致 |
| 文生图 | 39.75 s；[图片](qwen-heretic-generation.png) |
| 单图编辑 | 39.10 s，雪中狐狸添加围巾；[图片](qwen-heretic-single.png) |
| 双图编辑 | 40.68 s，组合狐狸与杯子；[图片](qwen-heretic-multi.png) |
| 资源释放 | `ImageEngine.close()` 后 MLX active 约 1 KiB；[验证记录](qwen-heretic-validation.json) |

这些样例用于确认加载、生成和编辑路径正常，不代表全面画质或行为评估。
正式服务的 encoder 路径和启动探活见 [启用记录](qwen-heretic-active.json)。
