# 本地 GLM 与 Qwen-Image 服务

使用 MLX 在 Apple Silicon 上运行 GLM-5.3-Flash-FineTunning 与 Qwen-Image-2.1。
统一入口为 `http://<本机地址>:1235/v1`，所有接口使用同一 API key。

| 服务 | 地址 | 模型 / 用途 |
| --- | --- | --- |
| `model_proxy.py` | `0.0.0.0:1235` | 对外入口、模型别名与 Responses 流式适配 |
| `server_launch.py` | `127.0.0.1:1236` | `glm-5.3-flash`，文本、图片理解与视频输入 |
| `image_server.py` | `127.0.0.1:1238` | `qwen-image-2.1`，文生图及单图、多图编辑 |

图片接口按需加载权重，空闲 300 秒卸载。GLM 权重在后端存活期间常驻。
图片接口使用独立环境，调用方法见 [图片 API](docs/image-api.md)。

## 启动与停止

从本目录执行：

```sh
bash setup_image.sh    # 安装或同步图片环境的固定版本依赖
./start.sh
./smoke.sh            # 向运行中的 GLM 发出六项接口检查
./stop.sh
```

GLM 环境为 `~/.venvs/mlx-vlm`（mlx-vlm 0.7.1、MLX 0.32.2），
图片环境为 `~/.venvs/qwen-image`，依赖固定在 [requirements-image.txt](requirements-image.txt)。
Python 环境应位于内置 APFS 卷，避免 exFAT 的 AppleDouble 伴生文件干扰包扫描。

默认权重路径：

| 用途 | 路径 |
| --- | --- |
| GLM 主模型 | `~/models/GLM-5.3-Flash-FineTunning-MXFP8` |
| GLM MTP 草稿模型 | `~/models/GLM-5.3-Flash-FineTunning-MXFP8-mtp` |
| Qwen 图片模型 | `~/models/Qwen-Image-2.1` |
| Qwen Heretic encoder | `~/models/Qwen-Image-2.1-Text-Encoder-Heretic` |

图片模型的 `text_encoder` 链接到 Heretic 权重；DiT、VAE、processor 仍使用原版组件。
配置兼容说明和回退路径见 [图片 API](docs/image-api.md#编码器)。

启动器可复用路径匹配且健康的后端。复用时，新传入的后端参数不会覆盖现有进程配置；
需要更换模型或后端参数时，先停止服务再启动。`DRAFT` 默认取 `${MODEL}-mtp`，
`DRAFT='' ./start.sh` 可关闭 MTP。Ctrl+C 只停止本次启动的子进程；`stop.sh` 还处理复用的后端。

`PREWARM=auto` 仅在主权重位于 `/Volumes/` 时顺序预读，以减少慢盘缺页导致的
Metal 命令超时风险。默认内置 SSD 路径跳过预读；`PREWARM=0` / `1` 可显式控制。
端口已占用时启动器会退出并保留现有 PID 记录；就绪检查超时会停止本次启动的子进程。

## 接口

下面命令中的 `API_KEY` 应设为启动服务时使用的值，默认值见 `start.sh`。
客户端使用公开模型名 `glm-5.3-flash` 或 `qwen-image-2.1`。

```sh
curl --fail-with-body http://127.0.0.1:1235/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"你好"}],"reasoning_effort":"low","max_tokens":1024}'
```

| 端点 | 说明 |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions，支持多模态输入 |
| `POST /v1/responses` | OpenAI Responses，支持流式及 `previous_response_id` |
| `POST /v1/responses/{id}/cancel` | 取消本进程跟踪的活跃 Responses 流式请求 |
| `POST /v1/messages` | Anthropic Messages |
| `POST /v1/images/generations` | 文生图，默认 1024×1024 |
| `POST /v1/images/edits` | 单图 / 多图编辑，1–10 张参考图，默认 512×512 |
| `GET /v1/models` | GLM 公开别名；图片服务可用时合并图片模型及加载状态 |
| `GET /health` | GLM 后端健康状态 |
| `GET /v1/images/health` | 图片服务状态；探活不加载权重 |
| `GET /metrics` | GLM 运行指标 |
| `GET /v1/cache/stats`、`POST /v1/cache/reset` | GLM 前缀缓存查询与清理 |
| `GET /v1/settings` | GLM 当前运行参数 |

生成端点也接受不带 `/v1` 的路径。图片服务不可用时，图片请求返回 503，
GLM 请求仍可使用，模型列表仅列出可查询的模型。

代理只对 GLM 请求的顶层 `model` 字段进行改写，固定到启动时的真实权重路径，
避免上游因模型名不匹配而切换权重。图片请求按路径转发，由图片服务校验模型名。
JSON 与 SSE 响应中的模型字段映射回公开别名，正文中的同名文本保持原值。

Responses 流适配会补齐 reasoning item 的声明和结束事件，并使 message 的索引与终态一致。
思考内容同时提供 `summary[].summary_text` 和 `content[].reasoning_text`。
Chat Completions 使用 `reasoning_content`，Messages 使用 `thinking` 块。

## GLM 参数与内存

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `MODEL` | 上表 GLM 主模型路径 | 固定加载的权重 |
| `DRAFT` | `${MODEL}-mtp` | MTP 权重；空字符串关闭 |
| `DRAFT_BLOCK_SIZE` | `2` | MTP 块长度（包含 anchor）；`auto` 使用上游自适应策略。未设置时也接受上游 `MLX_VLM_DRAFT_BLOCK_SIZE` |
| `PYTHON` | `~/.venvs/mlx-vlm/bin/python` | GLM 与代理解释器 |
| `HOST` / `PORT` | `0.0.0.0` / `1235` | 对外监听地址；HOST 也可设为 `127.0.0.1` |
| `BACKEND_PORT` | `1236` | GLM 本机端口 |
| `PUBLIC_NAME` | `glm-5.3-flash` | GLM 对外模型名 |
| `API_KEY` | 见 `start.sh` | 统一鉴权密钥 |
| `CONTEXT` | `393216` | 输入与生成预算之和的上限；0 使用模型声明上限 |
| `TOKEN_QUEUE_TIMEOUT` | `3000` | 等待生成队列下一个 token 的超时秒数 |
| `PREWARM` | `auto` | 外置权重预读策略 |
| `APC` | `1` | 启用前缀缓存 |
| `APC_MEMORY_MAX_GB` | `32` | APC 内存预算，GiB |
| `APC_NUM_BLOCKS` | `2048` | 上游块池配置；GLM 的 exact 路径不使用该块池 |
| `APC_DISK_ENABLED` | `0` | 是否将 exact 快照写入磁盘 |
| `APC_DISK_PATH` | `~/.cache/mlx-vlm/apc` | 磁盘缓存目录 |
| `APC_DISK_MAX_GB` | `200` | 磁盘缓存容量，GiB |
| `APC_DISK_MIN_TOKENS` | `8192` | 小于此长度的快照仅保存在内存 |
| `APC_STRIP_DERIVED` | `1` | 快照中剥离可重算的 projected 缓存 |
| `APC_RESERVE_FRACTION` | `0.25` | prefill 预留占 APC 内存预算的上限比例，范围 `(0, 1]` |
| `APC_EXACT_MAX_TOKENS` | `0` | exact 快照限长；0 不额外限制 |
| `GLM_PREFILL_SKIP_UNUSED_LOGITS` | `1` | 跳过中间预填充块中不参与采样的词表投影；0 恢复原路径 |
| `GLM_ALLOCATOR_CACHE_GB` | `2` | 可复用 GPU 分配池预算，GiB；可用小数，0 恢复空闲时清空 |

图片参数见 [图片 API 配置表](docs/image-api.md)。脚本固定传入 `--max-num-seqs 4`
和默认 `--max-tokens 65536`。模型声明上下文为 1,048,576 token，但完整 1M 推理及
四路满上下文的峰值内存未验证；384K 配置检查也不等同于完整 384K 生成基准。

聊天模板在未指定 `reasoning_effort` 时使用 `max`，可显式传 `low` 或 `high`。
`max_tokens` 包含思考消耗；预算耗尽可能只返回 reasoning、没有正文。
当前模型的生成配置为 `temperature=1.0`、`top_p=0.95`，请求可覆盖。

MTP 块长度包含 anchor，默认值 2 对应每轮一个草稿 token，主模型验证每个候选。
`auto` 使用上游自适应策略。性能随任务变化，对照数据见 [验证记录](diagnostics/README.md)。

GLM 的 APC 使用完整前缀快照（exact），按预算和 LRU 淘汰，没有固定 TTL。
磁盘层启用时会持久化符合阈值的完整快照，多轮长对话可能产生较大写入量。
剥离 projected 可缩小快照，但压缩潜变量的每 token 估算不包含临时张量，
不能据此保证整个推理进程或多请求峰值内存。

请求完成或取消后清理私有状态和输入嵌入；所有 GLM 任务空闲时同步计算流并恢复
wired limit。GPU 分配池在限额内保留，超额则清空；MLX 释放大缓冲区时可能暂时超额。
分配池预算只管已释放的缓冲区，独立于权重、活动张量和 APC，不是总内存上限。
设为 0 表示空闲时清空，推理期间仍可复用缓冲区。预填充、周期性解码及异常路径
也会清池。权重及预算内的 APC 常驻，空闲 `active` 内存不会归零。
客户端断连与显式取消在生成调度边界生效，不能抢占已提交的 Metal 命令。
开启新会话不会清空 APC；Responses 的文本历史与 GPU 缓存分别管理。

预填充优化仅作用于不需要输出 logits 的中间块，检查全部位置的隐藏状态是否有限。
最终采样位置、混合批次中到达输入末尾的行及解码路径保留原计算和 logits 检查。
该优化不改变权重、KV 精度、输入分块、采样参数或思考预算。

## 权重转换

已有完整产物时无需重新转换。转换脚本不覆盖现有输出目录。

```sh
./convert.sh
~/.venvs/mlx-vlm/bin/python verify_mxfp8.py
```

默认源为 `/Volumes/model/GLM-5.3-Flash-FineTunning`，输出为
`~/models/GLM-5.3-Flash-FineTunning-MXFP8` 及同名 `-mtp` 目录。
可用 `SRC`、`DST`、`PROCESSOR_SOURCE`、`PYTHON` 覆盖。

`prepare_checkpoint.py` 校验分片布局、长度和索引，缺索引时从分片头重建。
缺 processor 配置时，仅在视觉结构与媒体 token 配置一致后从参考模型补齐；
tokenizer 和聊天模板取自源模型。

FP8 E4M3 的 128×128 block scale 先还原到 BF16，再转为每 32 个值一组的 MXFP8。
转换使用 CPU，避免慢盘 mmap 缺页阻塞 Metal 命令。不要额外加 `-q`，
否则会改变原本保留为 BF16 的张量。`finalize_mxfp8.py` 验证输出并写入量化配置，
`verify_mxfp8.py` 抽查未融合张量的相对误差；抽查不能替代全模型质量评估。

## 代码与验证

| 文件 | 职责 |
| --- | --- |
| `start.sh` / `stop.sh` | 服务就绪检查、监管与停止 |
| `start_image.sh` / `stop_image.sh` | 单独管理图片进程 |
| `service_runner.py` | 转发关闭信号并轮转日志 |
| `model_proxy.py` | 路由、模型别名与 Responses 流适配 |
| `server_launch.py` | 安装 GLM 运行时补丁后启动上游 |
| `long_context_patch.py` | 稀疏注意力地址扩宽、非有限 logits 检查及 exact 限长 |
| `prefill_patch.py` | 跳过未使用的预填充词表投影，保留隐藏状态检查与采样路径 |
| `request_cleanup_patch.py` / `request_lifecycle.py` | 请求取消与资源释放 |
| `image_server.py` / `image_inputs.py` | 图片 API、校验、队列和按需加载 |
| `qwen_image_edit.py` | 共享权重的 MLX 单图与多图编辑 |
| `smoke.sh` / `smoke.py` | 对运行中的 GLM 做健康、发现及四类推理检查 |
| `tests/` | 不加载完整权重的回归测试 |

`smoke.sh [base_url] [api_key]` 检查 HTTP 状态、回答及 Responses 事件顺序和终态，
任一检查失败返回非零。自定义模型别名可使用 `PUBLIC_NAME` 或 `--model`。

补丁只作用于服务进程，不修改已安装的包或权重。它们依赖指定的 mlx-vlm 版本，
升级依赖时应复核内核源码、缓存结构与流事件，并重新运行对应测试。

```sh
# GLM 环境；图片数值测试在此环境中跳过
~/.venvs/mlx-vlm/bin/python -m unittest discover -s tests -p 'test_*.py' -v
# 图片编辑数值测试，需要 Metal
~/.venvs/qwen-image/bin/python tests/test_qwen_image_edit.py
```

大地址 GPU 测试默认跳过；`RUN_LARGE_METAL_TESTS=1` 可启用，额外分配最高约 8.6 GB。
完整模型测量的环境、范围及原始结果见 [验证记录](diagnostics/README.md)。

日志为 `logs/glm.log`、`logs/proxy.log`、`logs/image.log`，每个默认 64 MiB，
另保留三份轮转备份；可通过 `SERVICE_LOG_MAX_BYTES`、`SERVICE_LOG_BACKUPS` 调整。
转换日志单独追加到 `logs/convert.log`。

```sh
tail -f logs/glm.log
curl --fail-with-body http://127.0.0.1:1235/health -H "Authorization: Bearer $API_KEY"
curl --fail-with-body http://127.0.0.1:1235/v1/images/health -H "Authorization: Bearer $API_KEY"
```
