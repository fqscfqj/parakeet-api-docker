# 注意：已转移到新项目[NVIDIA ASR to OpenAI API](https://github.com/fqscfqj/nvidia-asr-openai-api)
# Parakeet API (Docker)

基于 NVIDIA NeMo 的中文/多语种语音识别服务，提供与 OpenAI Whisper 兼容的 `/v1/audio/transcriptions` 接口。已打包为支持 GPU 的 Docker 镜像，可一键运行。

- 预置模型：默认 `nvidia/parakeet-tdt-0.6b-v3`
- 支持25种语言，提供自动语言检测功能
- 支持长音频分片与重叠拼接，提供 SRT/VTT/verbose_json 等输出
- 自动检测 CUDA 兼容性：不兼容或无 GPU 时降级 CPU 模式（速度较慢）
- OpenAI Whisper API 兼容格式，包括错误响应


## 目录

- 快速开始（Windows PowerShell）
- 先决条件
- 使用预构建镜像运行
- 从源码构建并运行
- API 使用示例
- 语言检测与支持
- 配置与环境变量
- 端口、卷与文件结构
- 健康检查与监控
- 常见问题与排障
- 许可与致谢


## 快速开始（Windows PowerShell）

1. 准备目录并启动容器（使用预构建镜像）

```powershell
# 在仓库根目录执行
mkdir .\models -Force; mkdir .\temp_uploads -Force

# 启动（需要已安装 NVIDIA Container Toolkit）
docker compose up -d

# 查看日志（可选）
docker compose logs -f
```

1. 健康检查

- 简单健康：`http://localhost:5092/health/simple`
- 详细健康：`http://localhost:5092/health`

1. 试用 API（示例：JSON 文本输出）

```powershell
# 使用 curl.exe（建议在 PowerShell 下显式调用 curl.exe）
$audio = "C:\\path\\to\\audio.mp3"
curl.exe -X POST "http://localhost:5092/v1/audio/transcriptions" \
  -F "file=@$audio" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

> 如启用 API Key，需添加 `-H "Authorization: Bearer YOUR_API_KEY"`。


## 先决条件

- 操作系统：Linux/Windows（本文示例以 Windows PowerShell 为主）
- Docker：Docker Desktop 或 Docker Engine（Compose V2）
- GPU（可选但推荐）：
  - 已安装 NVIDIA 显卡与驱动（建议 535+），并安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  - 镜像基于 `nvidia/cuda:13.0.0-runtime-ubuntu22.04`，需满足对应驱动要求

无 GPU 时也可运行（自动 CPU 模式），但推理速度会显著降低。


## 使用预构建镜像运行

项目已提供 `docker-compose.yml`，默认拉取镜像 `ghcr.io/fqscfqj/parakeet-api-docker:full`。

```powershell
mkdir .\models -Force; mkdir .\temp_uploads -Force
docker compose up -d
# 更新镜像
# docker compose pull; docker compose up -d
```

Compose 主要配置：

- 端口映射：`5092:5092`
- 卷：
  - `./models:/app/models`（模型与缓存）
  - `./temp_uploads:/app/temp_uploads`（临时转码与切片文件）
- GPU：通过 `deploy.resources.reservations.devices` 申请全部可用 GPU


## 从源码构建并运行

如果需要定制 Dockerfile 或加速国内构建，可用 `docker-compose-build.yml`：

```powershell
mkdir .\models -Force; mkdir .\temp_uploads -Force
docker compose -f docker-compose-build.yml up -d --build
```

构建镜像包含：

- Python3.10 + Pip
- PyTorch/cu130 + torchaudio（来自官方 CUDA 13.0 轮子）
- NeMo ASR 及依赖、FFmpeg、健康检查脚本


## API 使用示例

- 端点：`POST /v1/audio/transcriptions`
- 字段（multipart/form-data）：
  - `file`：音/视频文件
  - `model`：兼容字段，默认 `whisper-1`
  - `response_format`：`json` | `text` | `srt` | `vtt` | `verbose_json`
  - `language`：可选，默认自动
  - `prompt`、`temperature`：可选

示例：返回 SRT 字幕

```powershell
$audio = "C:\\path\\to\\audio.wav"
curl.exe -X POST "http://localhost:5092/v1/audio/transcriptions" \
  -F "file=@$audio" \
  -F "model=whisper-1" \
  -F "response_format=srt"
```

启用 API Key：

```powershell
# 在 docker-compose.yml 中设置环境变量 API_KEY 后，调用时带上 Header
$audio = "C:\\path\\to\\audio.mp3"
$apiKey = "YOUR_API_KEY"
curl.exe -X POST "http://localhost:5092/v1/audio/transcriptions" \
  -H "Authorization: Bearer $apiKey" \
  -F "file=@$audio" -F "response_format=json"
```

## 语言检测与支持

### 支持的语言（25种）

本API支持以下25种语言的转写（基于 parakeet-tdt-0.6b-v3 模型）：

| 语言代码 | 语言名称 | 语言代码 | 语言名称 | 语言代码 | 语言名称 |
|---------|---------|---------|---------|---------|---------|
| bg | 保加利亚语 | hr | 克罗地亚语 | cs | 捷克语 |
| da | 丹麦语 | nl | 荷兰语 | en | 英语 |
| et | 爱沙尼亚语 | fi | 芬兰语 | fr | 法语 |
| de | 德语 | el | 希腊语 | hu | 匈牙利语 |
| it | 意大利语 | lv | 拉脱维亚语 | lt | 立陶宛语 |
| mt | 马耳他语 | pl | 波兰语 | pt | 葡萄牙语 |
| ro | 罗马尼亚语 | sk | 斯洛伐克语 | sl | 斯洛文尼亚语 |
| es | 西班牙语 | sv | 瑞典语 | ru | 俄语 |
| uk | 乌克兰语 | | | | |

### 自动语言检测

当请求中未指定 `language` 参数时，系统会自动检测音频语言：

1. **检测流程**：
   - 提取音频前段（默认45秒）进行快速转写
   - 使用 langdetect 库分析转写文本的语言
   - 如果检测到支持的语言，则使用该语言进行完整转写
   - 如果检测到不支持的语言，根据 `ENABLE_AUTO_LANGUAGE_REJECTION` 设置处理

2. **处理规则**：
   - **显式指定语言**：验证语言是否在支持列表中，不支持则返回 OpenAI 格式错误
   - **自动检测支持的语言**：使用检测到的语言进行转写
   - **自动检测不支持的语言**：
     - 如果 `ENABLE_AUTO_LANGUAGE_REJECTION=true`：返回 OpenAI 格式错误
     - 如果 `ENABLE_AUTO_LANGUAGE_REJECTION=false`：默认使用英语进行转写

3. **响应格式**：
   ```json
   {
     "text": "转写文本内容",
     "language": "auto-detected-lang-code"  // 仅在 verbose_json 格式中返回
   }
   ```

4. **配置选项**：
   - `ENABLE_AUTO_LANGUAGE_REJECTION`：是否拒绝不支持的语言（默认 `true`）
   - `LID_CLIP_SECONDS`：用于语言检测的音频片段长度（默认 `45` 秒）

### 使用示例

```bash
# 显式指定支持的语言
curl -X POST "http://localhost:5092/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "language=en" \
  -F "response_format=json"

# 自动检测语言
curl -X POST "http://localhost:5092/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json"  # 返回检测到的语言

# 显式指定不支持的语言（返回错误）
curl -X POST "http://localhost:5092/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "language=zh"  # 返回 OpenAI 格式错误响应
```


## 配置与环境变量

常用环境变量（可在 Compose 的 `environment:` 中设置）：

- 模型与加载
  - `MODEL_ID`：默认 `nvidia/parakeet-tdt-0.6b-v3`
  - `MODEL_LOCAL_PATH`：优先加载本地 `.nemo` 文件路径（挂载到 `./models` 后可指向 `/app/models/xxx.nemo`）
  - `ENABLE_LAZY_LOAD`：是否懒加载模型（默认 `true`）
  - `IDLE_TIMEOUT_MINUTES`：闲置自动卸载模型的分钟数，`0` 表示禁用（默认 `30`）
  - `API_KEY`：若设置，则启用 Bearer Token 认证
  - `HF_ENDPOINT`：Hugging Face 镜像端点，默认 `https://hf-mirror.com`

- 性能与显存
  - `PRESET`：`speed` | `balanced` | `quality` | `simple`（=balanced）。用于在启动时推导参数
  - `GPU_VRAM_GB`：显存容量（整数，GB）。若不设置，会尝试自动检测
  - `CHUNK_MINITE`：每段切片时长（分钟，默认 `10`，显存小可调小）
  - `MAX_CONCURRENT_INFERENCES`：最大并发推理数（默认 `1`）
  - `GPU_MEMORY_FRACTION`：单进程可使用的显存比例（默认 `0.90~0.95`）
  - `DECODING_STRATEGY`：`greedy` | `beam`，`RNNT_BEAM_SIZE`：Beam 宽度
  - `AGGRESSIVE_MEMORY_CLEANUP`：激进显存清理（默认 `true`）
  - `ENABLE_TENSOR_CORE`、`ENABLE_CUDNN_BENCHMARK`、`TENSOR_CORE_PRECISION`：Tensor Core/Benchmark 相关

- 闲置资源优化
  - `IDLE_MEMORY_CLEANUP_INTERVAL`：闲置时内存清理间隔（秒，默认 `120`）
  - `IDLE_DEEP_CLEANUP_THRESHOLD`：深度清理阈值（秒，默认 `600`）
  - `ENABLE_IDLE_CPU_OPTIMIZATION`：启用闲置时CPU优化（默认 `true`）
  - `IDLE_MONITORING_INTERVAL`：闲置监控间隔（秒，默认 `30`）
  - `ENABLE_AGGRESSIVE_IDLE_OPTIMIZATION`：启用超级激进内存优化（默认 `true`）
  - `IMMEDIATE_CLEANUP_AFTER_REQUEST`：请求完成后立即清理（默认 `true`）
  - `MEMORY_USAGE_ALERT_THRESHOLD_GB`：内存使用超过此值时强制清理（默认 `6.0`GB）
  - `AUTO_MODEL_UNLOAD_THRESHOLD_MINUTES`：模型自动卸载阈值（默认 `10`分钟）

> 💡 **资源优化建议**：新版本大幅增强了闲置优化策略，可将8GB闲置内存降低至2-3GB。启用超级激进优化后，系统会在模型闲置时执行多轮深度清理。可通过 `/health` 端点监控 `idle_status` 和资源使用情况。

- 切片与句子完整性
  - `ENABLE_OVERLAP_CHUNKING`：重叠切片（默认 `true`），`CHUNK_OVERLAP_SECONDS`：重叠秒数（默认 `30`）
  - `ENABLE_SILENCE_ALIGNED_CHUNKING`：静音对齐分割（默认 `true`）
  - `SILENCE_THRESHOLD_DB`（默认 `-38dB`）、`MIN_SILENCE_DURATION`（默认 `0.35`）、`SILENCE_MAX_SHIFT_SECONDS`（默认 `2.0`）

- 字幕后处理与换行
  - `MERGE_SHORT_SUBTITLES`（默认 `true`）、`MIN_SUBTITLE_DURATION_SECONDS`（默认 `1.5`）
  - `SHORT_SUBTITLE_MERGE_MAX_GAP_SECONDS`、`SHORT_SUBTITLE_MIN_CHARS`、`SUBTITLE_MIN_GAP_SECONDS`
  - `SPLIT_LONG_SUBTITLES`（默认 `true`）、`MAX_SUBTITLE_DURATION_SECONDS`（默认 `6.0`）
  - `MAX_SUBTITLE_CHARS_PER_SEGMENT`（默认 `84`）、`PREFERRED_LINE_LENGTH`（默认 `42`）、`MAX_SUBTITLE_LINES`（默认 `2`）
  - `ENABLE_WORD_TIMESTAMPS_FOR_SPLIT`（默认 `false`）

- 其他
  - `ENABLE_FFMPEG_DENOISE`（默认 `false`）、`DENOISE_FILTER`：FFmpeg 去噪/均衡/动态范围预处理
  - `NUMBA_CACHE_DIR`（默认 `/tmp/numba_cache`）：已在镜像中处理并赋予权限
  - `PUID` / `PGID`：容器启动时会将运行用户切换为指定 UID/GID，便于卷权限管理

> 小贴士：如果只是“能用就行”，先保留默认值；如遇显存不足，可降低 `CHUNK_MINITE`、设为 `PRESET=quality` 或将 `DECODING_STRATEGY=greedy`。


## 端口、卷与文件结构

- 端口：容器内监听 `5092`，可在 Compose 中改为其他宿主端口
- 卷：
  - `./models:/app/models`：保存/缓存模型（优先加载 `.nemo`）
  - `./temp_uploads:/app/temp_uploads`：临时转码与切片数据
- 关键文件：
  - `app.py`：Flask + Waitress 服务，提供 API 与切片/后处理逻辑
  - `Dockerfile`：CUDA 13.0 运行时 + 依赖安装 + 健康检查 + 启动脚本
  - `docker-compose.yml`：使用预构建镜像
  - `docker-compose-build.yml`：本地构建
  - `healthcheck.sh`：容器健康检查脚本


## 健康检查与监控

- `/health/simple`：返回 200 表示存活
- `/health`：返回 JSON，包含 GPU/CPU、内存与模型加载状态等
- 容器内置 `HEALTHCHECK`，Compose/编排平台可据此做重启策略


## 常见问题与排障（FAQ）

- 问：日志提示 “CUDA 不可用/兼容性错误”，服务退回 CPU？
  - 答：检查主机 NVIDIA 驱动是否满足 CUDA 13.x 运行时需求；确认已安装 NVIDIA Container Toolkit；Compose 中 device 预留是否生效。无法满足时可继续用 CPU，但速度会慢。

- 问：首次启动加载模型很慢或失败？
  - 答：默认从 Hugging Face 拉取，可设置 `MODEL_LOCAL_PATH` 指向本地 `.nemo`；或配置 `HF_ENDPOINT` 使用镜像。确保 `./models` 卷可写。

- 问：显存不足/频繁 OOM？
  - 答：将 `CHUNK_MINITE` 调小（如 6~8）；将 `DECODING_STRATEGY=greedy`；`PRESET=quality` 会自动调低并发与显存占比；必要时关闭 `ENABLE_OVERLAP_CHUNKING`。

- 问：想要进一步优化闲置时的资源占用？
  - 答：新版提供了超级激进的内存优化策略：
    - `ENABLE_AGGRESSIVE_IDLE_OPTIMIZATION=true`：启用超级激进内存清理
    - `IMMEDIATE_CLEANUP_AFTER_REQUEST=true`：请求完成后立即清理
    - `MEMORY_USAGE_ALERT_THRESHOLD_GB=6.0`：内存超过6GB时自动强制清理
    - `AUTO_MODEL_UNLOAD_THRESHOLD_MINUTES=10`：模型闲置10分钟后自动卸载
    - `IDLE_MEMORY_CLEANUP_INTERVAL=120`：每2分钟执行内存清理
    - `IDLE_DEEP_CLEANUP_THRESHOLD=600`：闲置10分钟后执行深度清理
    - `IDLE_MONITORING_INTERVAL=30`：每30秒检查一次闲置状态

- 问：如何解决8GB闲置内存占用问题？
  - 答：使用以下环境变量配置可将闲置内存大幅降低：
    ```bash
    ENABLE_AGGRESSIVE_IDLE_OPTIMIZATION=true
    MEMORY_USAGE_ALERT_THRESHOLD_GB=4.0
    AUTO_MODEL_UNLOAD_THRESHOLD_MINUTES=5
    IDLE_MEMORY_CLEANUP_INTERVAL=60
    IMMEDIATE_CLEANUP_AFTER_REQUEST=true
    ```

- 问：返回的字幕太碎或闪烁？
  - 答：可调 `MIN_SUBTITLE_DURATION_SECONDS`、`SHORT_SUBTITLE_MERGE_MAX_GAP_SECONDS`、`SHORT_SUBTITLE_MIN_CHARS`；或关闭 `MERGE_SHORT_SUBTITLES=false`。

- 问：端口冲突？
  - 答：修改 Compose 的 `ports` 映射，例如 `"18080:5092"`。

- 问：权限问题（Windows 卷）？
  - 答：可通过设置 `PUID` / `PGID`（Linux 更常用）或确保 Docker Desktop 共享磁盘权限正常。遇到权限受限时，删除卷目录后重建也可缓解。


## 许可与致谢

- 本项目：见 `LICENSE`
- 模型与依赖：NVIDIA NeMo（ASR）、PyTorch、FFmpeg、Hugging Face 等开源生态


---
完成度与验证

- 构建：Docker/Compose 清单已就绪
- 运行：提供 GPU/CPU 双路径与健康检查
- 用法：给出 PowerShell 友好命令与 curl 示例
- 覆盖需求：已新增用户友好 README（本文件）
