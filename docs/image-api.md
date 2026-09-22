# Qwen-Image-2.1 图片接口

客户端使用 `http://<本机地址>:1235/v1` 和统一 API key。
图片服务可用时，`GET /v1/models` 同时列出 `glm-5.3-flash` 和 `qwen-image-2.1`。
环境安装、权重路径及统一启动方式见 [项目 README](../README.md)。以下命令从项目根目录执行。

**按需加载**：启动、探活和查询模型列表都不加载图片权重；第一条生图或编辑请求才加载，
空闲 300 秒自动卸载并清空 MLX 缓存。首次请求包含加载耗时。
`GET /v1/images/health` 可查看 `loaded`、`active` 和 `queued`，同样需要鉴权。
调用示例（`API_KEY` 设置为启动时使用的密钥）：

```sh
curl --fail-with-body http://127.0.0.1:1235/v1/images/generations \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-image-2.1","prompt":"A red fox sitting in fresh snow","size":"1024x1024","steps":40,"seed":42,"response_format":"b64_json"}' \
  | ~/.venvs/qwen-image/bin/python -c \
    'import sys,json,base64; from pathlib import Path; r=json.load(sys.stdin); Path("image.png").write_bytes(base64.b64decode(r["data"][0]["b64_json"]))'
```

文生图和编辑都输出 RGB PNG/base64。文生图 `size` 默认 `1024x1024`，两边分别支持
256–2048 且为 16 的倍数；`steps` 默认 40、范围 1–80；`n` 固定为 1。
可传 `seed`、`guidance`（默认 1）和 `negative_prompt`。
仅接受本地固定模型；输出格式为 RGB PNG，不接受输出文件路径或透明背景参数。
在聊天中自动画图还需客户端调用此图片接口或配置对应工具。

## 单图与多图编辑

`POST /v1/images/edits` 支持 1–10 张参考图，共用同一组权重、队列和空闲卸载机制。
模型按上传顺序理解“图 1 / image 1”“图 2 / image 2”等引用。
单图可改背景、服装、物体或文字；多图可组合主体、场景和风格。
这是提示词驱动的整图编辑，目前不提供 `mask` 局部遮罩接口。

单图（multipart 的文件字段为 `image`）：

```sh
curl --fail-with-body http://127.0.0.1:1235/v1/images/edits \
  -H "Authorization: Bearer $API_KEY" \
  -F 'model=qwen-image-2.1' \
  -F 'image=@input.png' \
  -F 'prompt=保留狐狸的姿态和雪景，给它戴上一条红色针织围巾' \
  -F 'size=512x512' -F 'steps=40' -F 'seed=42' \
  -o edited.json
```

多图（重复 `image[]`，也支持重复 `image`）：

```sh
curl --fail-with-body http://127.0.0.1:1235/v1/images/edits \
  -H "Authorization: Bearer $API_KEY" \
  -F 'image[]=@fox.png' \
  -F 'image[]=@cup.png' \
  -F 'prompt=保留图1的狐狸和雪景，把图2的杯子放在狐狸前爪旁边，保留杯子的颜色和图案' \
  -F 'size=512x512' -F 'steps=40' -F 'seed=42' \
  -o edited.json
```

将返回内容保存为图片：

```sh
~/.venvs/qwen-image/bin/python -c \
  'import json,base64; from pathlib import Path; r=json.loads(Path("edited.json").read_text()); Path("edited.png").write_bytes(base64.b64decode(r["data"][0]["b64_json"]))'
```

也接受 `Content-Type: application/json`，单图使用 `"image":"data:image/png;base64,..."`，
多图使用 `"images":["data:image/png;base64,...","data:image/jpeg;base64,..."]`；
每项也可直接传纯 base64 字符串。不读取远程 URL 或服务器文件路径。
返回结构与文生图相同，图片在 `data[0].b64_json`。

| 编辑参数/限制 | 说明 |
| --- | --- |
| 输入 | 静态 PNG、JPEG、WebP，自动应用 EXIF 方向；透明输入可用，输出为 RGB |
| 数量与大小 | 1–10 张；单张 ≤10 MiB、≤1600 万像素；总计 ≤4000 万像素；请求体 ≤64 MiB |
| 宽高比 | 每张参考图介于 1:16 和 16:1 |
| `size` | 编辑输出尺寸，默认 `512x512`；两边 256–2048，须为 **32** 的倍数 |
| `reference_size` | 参考图编码分辨率，默认 `512`，可选 `768` / `1024`；按原宽高比缩放到约该值平方的像素面积，更高值保留更多细节，也增加耗时和内存 |
| `prompt` / `negative_prompt` | 每项最多 1024 个文本 token；`negative_prompt` 在 `guidance > 1` 时生效 |
| 其它 | 与文生图相同：`steps`、`seed`、`guidance`、`n=1`、`response_format=b64_json`、`output_format=png` |

编辑使用原生 MLX，与文生图共用权重，无需额外下载编辑模型或安装 PyTorch。
实现及参考来源见 [`qwen_image_edit.py`](../qwen_image_edit.py)。

编辑默认使用 512×512。实测的 1024 双图样例有偏锐化，提高参考编码分辨率未消除，
因此高分辨率结果需要按任务评估；编辑不保证未指定区域逐像素保持不变。
画质样例和耗时见 [验证记录](../diagnostics/README.md#qwen-image)。

图片任务串行运行，最多额外排队 2 条，队满返回 429。入队后的请求超时默认为
1800 秒，包含排队、加载和生成，超时返回 504；上传与校验不计入此期限。
客户端断连或超时会触发取消；取消在计算阶段之间检查，已提交的操作会先结束。
图片服务故障时，聊天及 GLM 模型发现仍然可用。

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `IMAGE_ENABLED` | `1` | 设为 `0`，统一启动时禁用图片入口 |
| `IMAGE_MODEL` | `~/models/Qwen-Image-2.1` | 本地权重目录 |
| `IMAGE_PYTHON` | `~/.venvs/qwen-image/bin/python` | 图片专用 Python |
| `IMAGE_PORT` | `1238` | 本机监听端口 |
| `IMAGE_IDLE_TIMEOUT` | `300` | 空闲卸载等待秒数 |
| `IMAGE_REQUEST_TIMEOUT` | `1800` | 从入队开始计算的超时秒数 |
| `IMAGE_QUEUE_SIZE` | `2` | 等待队列长度 |
| `IMAGE_MEMORY_GB` | `80` | MLX 图求值的内存参考限额，GiB；不是进程总内存硬上限 |

例如 `IMAGE_IDLE_TIMEOUT=60 ./start.sh` 可缩短保留时间；参数对新启动的图片进程生效，
复用已运行进程时保留该进程原配置。图片日志为 `logs/image.log`。
可单独运行 `bash start_image.sh`；它使用 `IMAGE_API_KEY`，默认沿用 `API_KEY`。
`bash stop_image.sh` 只停止本工程的图片进程，可单独重启图片接口来应用参数。

## 编码器

当前使用本地 `Qwen-Image-2.1-Text-Encoder-Heretic`（BF16，保留视觉塔）：

| 路径 | 用途 |
| --- | --- |
| `~/models/Qwen-Image-2.1-Text-Encoder-Heretic` | 使用中的 encoder 权重 |
| `~/models/Qwen-Image-2.1/text_encoder` | 指向上述目录的软链接 |
| `~/models/Qwen-Image-2.1-Text-Encoder-Original` | 原版 encoder，供回退使用 |

文生图与单图、多图编辑共用此 encoder。对外模型名仍为 `qwen-image-2.1`，
DiT、VAE、processor 和提示词模板沿用原版。

Heretic 的 Transformers 配置将 RoPE 参数保存在 `text_config.rope_parameters`。
本地 `config.json` 额外提供等价的 `rope_theta`、`rope_scaling`，供 mlx-vlm 0.7.2 读取；
视觉配置的 `model_type` 使用该版本支持的 `qwen3_vl` 别名。
原始配置保存在同目录的 `config.transformers-original.json`，safetensors 权重不作转换。
[安装校验记录](../diagnostics/qwen-heretic-install.json) 包含源文件 SHA-256 与兼容配置摘要。

切换 encoder 目录或软链接前应停止图片服务，完成后用 `bash start_image.sh` 启动。
仅修改磁盘链接不会替换已加载在进程中的权重。
