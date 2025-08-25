import os,sys,json,math

# 设置环境变量来解决numba缓存问题
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_JIT'] = '0'

# 设置matplotlib配置目录，避免权限问题
# 优先使用启动脚本设置的目录，如果不存在则使用备用目录
if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'
    os.makedirs('/tmp/matplotlib_config', exist_ok=True)
    os.chmod('/tmp/matplotlib_config', 0o777)
else:
    # 确保已设置的目录存在且有正确权限
    mpl_dir = os.environ['MPLCONFIGDIR']
    try:
        os.makedirs(mpl_dir, exist_ok=True)
        os.chmod(mpl_dir, 0o755)
    except (PermissionError, OSError):
        # 如果无法创建或设置权限，回退到tmp目录
        os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'
        os.makedirs('/tmp/matplotlib_config', exist_ok=True)
        os.chmod('/tmp/matplotlib_config', 0o777)

host = '0.0.0.0'
port = 5092
threads = 4
# 默认按照N分钟将音视频裁切为多段，减少显存占用。现在可以通过环境变量 CHUNK_MINITE 来调整。
# 8G显存建议设置为 10-15 分钟以获得最佳性能。
CHUNK_MINITE = int(os.environ.get('CHUNK_MINITE', '10'))
# 服务闲置N分钟后自动卸载模型以释放显存，设置为0则禁用
IDLE_TIMEOUT_MINUTES = int(os.environ.get('IDLE_TIMEOUT_MINUTES', '30'))
# 懒加载开关，默认为 true。设置为 'false' 可在启动时预加载模型。
ENABLE_LAZY_LOAD = os.environ.get('ENABLE_LAZY_LOAD', 'true').lower() not in ['false', '0', 'f']
# Whisper 兼容的 API Key。如果留空，则不进行身份验证。
API_KEY = os.environ.get('API_KEY', None)
import shutil
from typing import Any, Dict
import uuid
import subprocess
import datetime
import threading
import time
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, Response
from waitress import serve
from pathlib import Path
# ROOT_DIR is not needed in Docker environment
# 仅当未显式配置时才设置 HF 镜像（可通过环境变量覆盖）
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# HF_HOME is set in the Dockerfile
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'
# PATH for ffmpeg is handled by the Docker image's system PATH

# 减少 PyTorch CUDA 分配碎片，降低 OOM 几率（可通过外部环境变量覆盖）
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

import nemo.collections.asr as nemo_asr  # type: ignore
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import gc
import psutil
try:
    # huggingface_hub may not be present in the editor environment; import defensively
    from huggingface_hub import HfApi, hf_hub_download  # type: ignore
except Exception:
    # Provide fallbacks so static checkers and runtime in minimal environments won't crash.
    HfApi = None  # type: ignore
    def hf_hub_download(*args, **kwargs):
        raise RuntimeError("huggingface_hub is not installed")

# --- 全局设置与模型状态 ---
asr_model = None
last_request_time = None
model_lock = threading.Lock()
cuda_available = False  # 全局CUDA兼容性标志

# 受支持的语言（ISO 639-1，两字母小写），基于 parakeet-tdt-0.6b-v3 公告
SUPPORTED_LANG_CODES = {
    'bg','hr','cs','da','nl','en','et','fi','fr','de','el','hu','it','lv','lt','mt','pl','pt','ro','sk','sl','es','sv','ru','uk'
}

# 语言自动拒绝（当未显式传入 language 时，先对短片段做语言初判；若不受支持则直接返回 Whisper 风格错误）
ENABLE_AUTO_LANGUAGE_REJECTION = os.environ.get('ENABLE_AUTO_LANGUAGE_REJECTION', 'true').lower() in ['true', '1', 't']
LID_CLIP_SECONDS = int(os.environ.get('LID_CLIP_SECONDS', '45'))

# 推理并发控制（避免多请求同时占用显存导致 OOM）
MAX_CONCURRENT_INFERENCES = int(os.environ.get('MAX_CONCURRENT_INFERENCES', '1'))
inference_semaphore = threading.Semaphore(MAX_CONCURRENT_INFERENCES)

# 显存优化配置
AGGRESSIVE_MEMORY_CLEANUP = os.environ.get('AGGRESSIVE_MEMORY_CLEANUP', 'true').lower() in ['true', '1', 't']
ENABLE_GRADIENT_CHECKPOINTING = os.environ.get('ENABLE_GRADIENT_CHECKPOINTING', 'true').lower() in ['true', '1', 't']
MAX_CHUNK_MEMORY_MB = int(os.environ.get('MAX_CHUNK_MEMORY_MB', '1500'))
FORCE_CLEANUP_THRESHOLD = float(os.environ.get('FORCE_CLEANUP_THRESHOLD', '0.8'))

# 闲置时资源优化配置
IDLE_MEMORY_CLEANUP_INTERVAL = int(os.environ.get('IDLE_MEMORY_CLEANUP_INTERVAL', '300'))  # 闲置时内存清理间隔(秒)，默认5分钟
IDLE_DEEP_CLEANUP_THRESHOLD = int(os.environ.get('IDLE_DEEP_CLEANUP_THRESHOLD', '1800'))  # 深度清理阈值(秒)，默认30分钟
ENABLE_IDLE_CPU_OPTIMIZATION = os.environ.get('ENABLE_IDLE_CPU_OPTIMIZATION', 'true').lower() in ['true', '1', 't']
IDLE_MONITORING_INTERVAL = int(os.environ.get('IDLE_MONITORING_INTERVAL', '60'))  # 闲置监控间隔(秒)，默认1分钟

# Tensor Core 优化配置
ENABLE_TENSOR_CORE = os.environ.get('ENABLE_TENSOR_CORE', 'true').lower() in ['true', '1', 't']
ENABLE_CUDNN_BENCHMARK = os.environ.get('ENABLE_CUDNN_BENCHMARK', 'true').lower() in ['true', '1', 't']
TENSOR_CORE_PRECISION = os.environ.get('TENSOR_CORE_PRECISION', 'highest')  # highest, high, medium
GPU_MEMORY_FRACTION = float(os.environ.get('GPU_MEMORY_FRACTION', '0.95'))  # 进程允许使用的显存比例

# 句子完整性优化配置
ENABLE_OVERLAP_CHUNKING = os.environ.get('ENABLE_OVERLAP_CHUNKING', 'true').lower() in ['true', '1', 't']
CHUNK_OVERLAP_SECONDS = float(os.environ.get('CHUNK_OVERLAP_SECONDS', '30'))  # 重叠时长
SENTENCE_BOUNDARY_THRESHOLD = float(os.environ.get('SENTENCE_BOUNDARY_THRESHOLD', '0.5'))  # 句子边界检测阈值


# 静音对齐切片与前处理配置
ENABLE_SILENCE_ALIGNED_CHUNKING = os.environ.get('ENABLE_SILENCE_ALIGNED_CHUNKING', 'true').lower() in ['true', '1', 't']
SILENCE_THRESHOLD_DB = os.environ.get('SILENCE_THRESHOLD_DB', '-38dB')  # ffmpeg silencedetect 噪声阈值
MIN_SILENCE_DURATION = float(os.environ.get('MIN_SILENCE_DURATION', '0.35'))  # 认为是静音的最小时长(秒)
SILENCE_MAX_SHIFT_SECONDS = float(os.environ.get('SILENCE_MAX_SHIFT_SECONDS', '2.0'))  # 目标分割点附近允许向静音对齐的最大偏移(秒)

ENABLE_FFMPEG_DENOISE = os.environ.get('ENABLE_FFMPEG_DENOISE', 'false').lower() in ['true', '1', 't']
# 合理的默认去噪/均衡/动态范围设置，尽可能温和，避免过拟合
DENOISE_FILTER = os.environ.get(
    'DENOISE_FILTER',
    'afftdn=nf=-25,highpass=f=50,lowpass=f=8000,dynaudnorm=m=7:s=5'
)

# 解码策略（若模型支持）
DECODING_STRATEGY = os.environ.get('DECODING_STRATEGY', 'greedy')  # 可选: greedy, beam
RNNT_BEAM_SIZE = int(os.environ.get('RNNT_BEAM_SIZE', '4'))

# Nemo 转写运行时配置（批量与DataLoader）
TRANSCRIBE_BATCH_SIZE = int(os.environ.get('TRANSCRIBE_BATCH_SIZE', '1'))
TRANSCRIBE_NUM_WORKERS = int(os.environ.get('TRANSCRIBE_NUM_WORKERS', '0'))

# 字幕后处理配置（防止字幕显示时间过短）
MERGE_SHORT_SUBTITLES = os.environ.get('MERGE_SHORT_SUBTITLES', 'true').lower() in ['true', '1', 't']
MIN_SUBTITLE_DURATION_SECONDS = float(os.environ.get('MIN_SUBTITLE_DURATION_SECONDS', '1.5'))
SHORT_SUBTITLE_MERGE_MAX_GAP_SECONDS = float(os.environ.get('SHORT_SUBTITLE_MERGE_MAX_GAP_SECONDS', '0.3'))
SHORT_SUBTITLE_MIN_CHARS = int(os.environ.get('SHORT_SUBTITLE_MIN_CHARS', '6'))
SUBTITLE_MIN_GAP_SECONDS = float(os.environ.get('SUBTITLE_MIN_GAP_SECONDS', '0.06'))

# 长字幕拆分与换行（可选）
# - 将过长/过久的字幕拆为多条；同时对每条字幕内文本进行换行，便于观看
SPLIT_LONG_SUBTITLES = os.environ.get('SPLIT_LONG_SUBTITLES', 'true').lower() in ['true', '1', 't']
MAX_SUBTITLE_DURATION_SECONDS = float(os.environ.get('MAX_SUBTITLE_DURATION_SECONDS', '6.0'))
MAX_SUBTITLE_CHARS_PER_SEGMENT = int(os.environ.get('MAX_SUBTITLE_CHARS_PER_SEGMENT', '84'))  # 约两行，每行~42
PREFERRED_LINE_LENGTH = int(os.environ.get('PREFERRED_LINE_LENGTH', '42'))
MAX_SUBTITLE_LINES = int(os.environ.get('MAX_SUBTITLE_LINES', '2'))
# 若为 true，尝试使用词级时间戳进行更精确的拆分（模型若未返回words则自动回退）
ENABLE_WORD_TIMESTAMPS_FOR_SPLIT = os.environ.get('ENABLE_WORD_TIMESTAMPS_FOR_SPLIT', 'false').lower() in ['true', '1', 't']
# 通过标点优先切分，逗号/句号/问号/感叹号/分号等
SUBTITLE_SPLIT_PUNCTUATION = os.environ.get('SUBTITLE_SPLIT_PUNCTUATION', '。！？!?.,;；，,')

# 简化配置：预设与GPU显存（GB）
PRESET = os.environ.get('PRESET', 'balanced').lower()  # speed | balanced | quality | simple(=balanced)
GPU_VRAM_GB_ENV = os.environ.get('GPU_VRAM_GB', '').strip()


# 确保临时上传目录存在
if not os.path.exists('/app/temp_uploads'):
    os.makedirs('/app/temp_uploads')

def setup_tensor_core_optimization():
    """配置Tensor Core优化设置"""
    global cuda_available
    if not cuda_available:
        print("CUDA不可用，跳过Tensor Core优化配置")
        return
    
    print("正在配置 Tensor Core 优化...")
    
    try:
        # 启用 cuDNN benchmark 模式
        if ENABLE_CUDNN_BENCHMARK:
            cudnn.benchmark = True
            cudnn.deterministic = False  # 为了性能，允许非确定性
            print("✅ cuDNN benchmark 已启用")
        else:
            cudnn.benchmark = False
            cudnn.deterministic = True
            print("❌ cuDNN benchmark 已禁用（确定性模式）")
        
        # 启用 cuDNN 允许 TensorCore
        if ENABLE_TENSOR_CORE:
            cudnn.allow_tf32 = True  # 允许TF32（A100等支持）
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ Tensor Core (TF32) 已启用")
        else:
            cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print("❌ Tensor Core 已禁用")
        
        # 设置 Tensor Core 精度策略
        if TENSOR_CORE_PRECISION == 'highest':
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            print("✅ 设置为最高精度模式")
        elif TENSOR_CORE_PRECISION == 'high':
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            print("✅ 设置为高精度模式")
        else:  # medium
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            print("✅ 设置为中等精度模式")
        
        # 设置内存分配策略以优化 Tensor Core 使用
        try:
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
            print(f"✅ GPU 内存分配比例: {GPU_MEMORY_FRACTION*100:.0f}%")
        except Exception as e:
            print(f"⚠️ 设置内存分配比例失败: {e}")
        print("✅ GPU 内存分配策略已优化")
    except Exception as e:
        print(f"⚠️ Tensor Core优化配置失败: {e}")

def get_tensor_core_info():
    """获取 Tensor Core 支持信息"""
    global cuda_available
    if not cuda_available:
        return "N/A - CUDA不可用"
    
    try:
        device = torch.cuda.get_device_properties(0)
        major, minor = device.major, device.minor
        
        # 检测 Tensor Core 支持
        if major >= 7:  # V100, T4, RTX 20/30/40系列等
            if major == 7:
                return f"✅ Tensor Core 1.0 (计算能力 {major}.{minor})"
            elif major == 8:
                if minor >= 0:
                    return f"✅ Tensor Core 2.0 + TF32 (计算能力 {major}.{minor})"
                else:
                    return f"✅ Tensor Core 2.0 (计算能力 {major}.{minor})"
            elif major >= 9:
                return f"✅ Tensor Core 3.0+ (计算能力 {major}.{minor})"
        elif major >= 6:  # P100等
            return f"⚠️ 有限Tensor Core支持 (计算能力 {major}.{minor})"
        else:
            return f"❌ 不支持Tensor Core (计算能力 {major}.{minor})"
        
        return f"未知 (计算能力 {major}.{minor})"
    except Exception as e:
        return f"❌ 获取GPU信息失败: {e}"

def optimize_tensor_operations():
    """优化张量操作以更好地利用 Tensor Core"""
    global cuda_available
    if not cuda_available:
        print("CUDA不可用，跳过Tensor Core预热")
        return
    
    try:
        # 设置优化的 CUDA 流
        torch.cuda.set_sync_debug_mode(0)  # 禁用同步调试以提升性能
        
        # 预热GPU，确保Tensor Core正确激活
        # 创建一些对齐到8/16倍数的矩阵进行预热
        device = torch.cuda.current_device()
        dummy_a = torch.randn(128, 128, device=device, dtype=torch.float16)
        dummy_b = torch.randn(128, 128, device=device, dtype=torch.float16)
        
        # 执行矩阵乘法预热Tensor Core
        with torch.cuda.amp.autocast():
            _ = torch.matmul(dummy_a, dummy_b)
        
        torch.cuda.synchronize()
        del dummy_a, dummy_b
        torch.cuda.empty_cache()
        print("✅ Tensor Core 预热完成")
    except Exception as e:
        print(f"⚠️ Tensor Core 预热失败: {e}")

def detect_sentence_boundaries(text: str) -> list:
    """检测句子边界，返回句子结束位置列表"""
    import re
    
    # 中英文句号、问号、感叹号等
    sentence_endings = re.finditer(r'[.!?。！？]+[\s]*', text)
    boundaries = [match.end() for match in sentence_endings]
    return boundaries

def find_best_split_point(segments: list, target_time: float, tolerance: float = 2.0) -> int:
    """在目标时间附近找到最佳的句子分割点"""
    if not segments:
        return 0
    
    best_index = 0
    min_distance = float('inf')
    
    # 寻找最接近目标时间的句子结束点
    for i, segment in enumerate(segments):
        segment_end = segment.get('end', 0)
        distance = abs(segment_end - target_time)
        
        # 检查是否是句子结束（包含标点符号）
        text = segment.get('segment', '').strip()
        if text and (text.endswith('.') or text.endswith('。') or 
                     text.endswith('!') or text.endswith('！') or
                     text.endswith('?') or text.endswith('？')):
            # 句子结束点，权重更高
            distance *= 0.5
        
        if distance < min_distance and distance <= tolerance:
            min_distance = distance
            best_index = i + 1  # 返回下一个段落的索引
    
    return min(best_index, len(segments))

def merge_overlapping_segments(all_segments: list, chunk_boundaries: list, overlap_seconds: float) -> list:
    """合并重叠区域的segments，去除重复内容"""
    if not ENABLE_OVERLAP_CHUNKING or len(chunk_boundaries) <= 1:
        return all_segments
    
    # 简化并更鲁棒：按时间排序，然后基于重叠窗口内去重同文段落
    if not all_segments:
        return []
    all_segments_sorted = sorted(all_segments, key=lambda s: (s.get('start', 0.0), s.get('end', 0.0)))
    merged = []
    for seg in all_segments_sorted:
        text = seg.get('segment', '').strip()
        if not text:
            continue
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        # 若时间上高度重叠，且文本高相似（或完全相同），则保留更长/置信度更高的一条
        overlap = min(prev['end'], seg['end']) - max(prev['start'], seg['start'])
        window = overlap_seconds * 0.9 if overlap_seconds else 0.0
        def normalized(t: str) -> str:
            return ''.join(t.split()).lower()
        same_text = normalized(prev.get('segment', '')) == normalized(text)
        if overlap > 0 and overlap >= min(prev['end'] - prev['start'], seg['end'] - seg['start']) * 0.5:
            if same_text or overlap >= window:
                # 选择时间更长的段落
                if (prev['end'] - prev['start']) >= (seg['end'] - seg['start']):
                    # 可能扩展尾部
                    prev['end'] = max(prev['end'], seg['end'])
                else:
                    merged[-1] = seg
                continue
        # 否则直接追加
        merged.append(seg)
    print(f"合并完成，最终 {len(merged)} 个segments")
    return merged

def enforce_min_subtitle_duration(
    segments: list,
    min_duration: float,
    merge_max_gap: float,
    min_chars: int,
    min_gap: float,
) -> list:
    """对转写的 segments 进行后处理，避免字幕显示时间过短：
    1) 尝试将过短或文本过少的相邻段合并（两段间隙不超过 merge_max_gap）。
    2) 若仍短于 min_duration，尽量将当前段的结束时间延长到 min_duration，但不与下一段重叠（预留 min_gap）。

    segments: [{'start': float, 'end': float, 'segment': str}, ...]
    返回：处理后的 segments（按开始时间排序，且不重叠）
    """
    if not segments:
        return []

    # 按开始时间排序，深拷贝以免修改原对象
    segments_sorted = sorted(
        [
            {
                'start': float(s.get('start', 0.0)),
                'end': float(s.get('end', 0.0)),
                'segment': str(s.get('segment', '')),
            }
            for s in segments
        ],
        key=lambda s: (s['start'], s['end'])
    )

    result: list = []
    i = 0
    n = len(segments_sorted)

    while i < n:
        current = segments_sorted[i]
        current_text = str(current.get('segment', '')).strip()

        # 尝试前向合并，直到满足最短时长或无可合并对象
        while MERGE_SHORT_SUBTITLES:
            duration = max(0.0, float(current.get('end', 0.0)) - float(current.get('start', 0.0)))
            too_short = duration < min_duration or len(current_text) <= min_chars
            if not too_short or i + 1 >= n:
                break
            next_seg = segments_sorted[i + 1]
            gap = max(0.0, float(next_seg.get('start', 0.0)) - float(current.get('end', 0.0)))
            if gap > merge_max_gap:
                break
            # 合并到 current
            next_text = str(next_seg.get('segment', '')).strip()
            current['end'] = max(float(current.get('end', 0.0)), float(next_seg.get('end', 0.0)))
            current_text = (current_text + ' ' + next_text).strip()
            current['segment'] = current_text
            i += 1  # 吞并下一段
        # 合并完成后，如仍短则尝试延长，但不得与下一段重叠
        duration = max(0.0, float(current.get('end', 0.0)) - float(current.get('start', 0.0)))
        if duration < float(min_duration):
            desired_end = float(current.get('start', 0.0)) + float(min_duration)
            if i + 1 < n:
                next_start = float(segments_sorted[i + 1].get('start', 0.0))
                safe_end = max(float(current.get('end', 0.0)), min(desired_end, next_start - float(min_gap)))
                # 只有在不会导致非法区间时才更新
                if safe_end > float(current.get('start', 0.0)):
                    current['end'] = safe_end
            else:
                # 已是最后一段，直接延长
                current['end'] = desired_end

        result.append(current)
        i += 1

    # 最后再保证不重叠与单调递增
    cleaned: list = []
    for seg in result:
        if not cleaned:
            cleaned.append(seg)
            continue
        prev = cleaned[-1]
        if seg['start'] < prev['end']:
            seg['start'] = prev['end'] + min_gap
            if seg['start'] > seg['end']:
                seg['start'] = seg['end']
        cleaned.append(seg)

    return cleaned

def process_chunk_segments(segments: list, overlap_start: float, overlap_seconds: float) -> list:
    """处理单个chunk的segments，处理重叠区域"""
    if not segments:
        return []
    
    processed = []
    overlap_end = overlap_start + overlap_seconds
    
    for segment in segments:
        segment_start = segment['start']
        segment_end = segment['end']
        
        # 如果segment完全在重叠区域之前，直接添加
        if segment_end <= overlap_start:
            processed.append(segment)
        # 如果segment跨越重叠区域开始，需要检查是否截断
        elif segment_start < overlap_start < segment_end:
            # 检查是否在句子中间截断
            text = segment.get('segment', '').strip()
            if text and not any(punct in text for punct in ['.', '。', '!', '！', '?', '？']):
                # 在句子中间，保留完整segment
                processed.append(segment)
            else:
                # 可以安全截断的句子结束
                processed.append(segment)
        
    return processed

def create_overlap_chunks(total_duration: float, chunk_duration: float, overlap_seconds: float) -> list:
    """创建带重叠的chunk时间段"""
    chunks = []
    current_start = 0.0
    
    while current_start < total_duration:
        chunk_end = min(current_start + chunk_duration, total_duration)
        
        chunk_info = {
            'start': current_start,
            'end': chunk_end,
            'duration': chunk_end - current_start
        }
        chunks.append(chunk_info)
        
        # 下一个chunk的开始时间（考虑重叠）
        if chunk_end >= total_duration:
            break
            
        current_start = chunk_end - overlap_seconds
        
    print(f"创建了 {len(chunks)} 个重叠chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk['start']:.1f}s - {chunk['end']:.1f}s (时长: {chunk['duration']:.1f}s)")
    
    return chunks

def check_cuda_compatibility():
    """检查CUDA兼容性，如果不兼容则禁用CUDA"""
    global cuda_available
    
    try:
        if not torch.cuda.is_available():
            print("CUDA不可用，将使用CPU模式")
            cuda_available = False
            return False
        
        # 尝试获取设备数量来测试CUDA兼容性
        device_count = torch.cuda.device_count()
        if device_count == 0:
            print("未检测到CUDA设备，将使用CPU模式")
            cuda_available = False
            return False
            
        # 尝试获取设备属性来进一步测试兼容性
        device_props = torch.cuda.get_device_properties(0)
        print(f"✅ 检测到兼容的GPU: {device_props.name}")
        cuda_available = True
        return True
    except RuntimeError as e:
        if "forward compatibility was attempted on non supported HW" in str(e):
            print("⚠️ CUDA兼容性错误: GPU硬件不支持当前CUDA版本")
            print("这通常是因为主机的GPU驱动版本过旧，不支持容器中的CUDA 13.x 运行时")
            print("将自动切换到CPU模式运行")
        elif "CUDA" in str(e):
            print(f"⚠️ CUDA初始化失败: {e}")
            print("将自动切换到CPU模式运行")
        else:
            print(f"⚠️ 未知CUDA错误: {e}")
            print("将自动切换到CPU模式运行")
        
        cuda_available = False
        return False
    except Exception as e:
        print(f"⚠️ GPU兼容性检查失败: {e}")
        print("将自动切换到CPU模式运行")
        cuda_available = False
        return False

def get_gpu_memory_usage():
    """获取GPU显存使用情况"""
    global cuda_available
    if not cuda_available:
        return 0, 0, 0
    
    try:
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return allocated, reserved, total
    except Exception as e:
        print(f"⚠️ 获取GPU内存信息失败: {e}")
        return 0, 0, 0

def aggressive_memory_cleanup():
    """激进的显存清理函数"""
    global cuda_available
    if cuda_available:
        try:
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            # 同步所有CUDA操作
            torch.cuda.synchronize()
            # 重置峰值内存统计
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"⚠️ CUDA清理操作失败: {e}")
    
    # 强制Python垃圾回收
    for _ in range(3):
        gc.collect()
    
    if cuda_available:
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ CUDA缓存清理失败: {e}")

def idle_deep_memory_cleanup():
    """闲置时深度内存清理函数"""
    global cuda_available
    print("🧹 执行闲置时深度内存清理...")
    
    # 执行标准的激进清理
    aggressive_memory_cleanup()
    
    # 额外的深度清理措施
    if cuda_available:
        try:
            # 多次清空CUDA缓存以确保彻底
            for _ in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 重置所有内存统计
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except Exception as e:
            print(f"⚠️ 深度CUDA清理失败: {e}")
    
    # 更强力的垃圾回收
    for _ in range(5):
        gc.collect()
    
    # 尝试设置低优先级 (仅在支持的系统上)
    if ENABLE_IDLE_CPU_OPTIMIZATION:
        try:
            import os
            import psutil
            current_process = psutil.Process()
            # 设置为低优先级 (仅在闲置时)
            if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS'):
                current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            elif hasattr(current_process, 'nice'):
                current_process.nice(10)  # 设置为低优先级
        except Exception as e:
            # 静默失败，不影响主要功能
            pass
    
    allocated, reserved, total = get_gpu_memory_usage()
    print(f"✅ 深度清理完成，当前显存使用: {allocated:.2f}GB / {total:.2f}GB")

def should_force_cleanup():
    """检查是否需要强制清理显存"""
    global cuda_available
    if not cuda_available:
        return False
    
    allocated, reserved, total = get_gpu_memory_usage()
    usage_ratio = allocated / total if total > 0 else 0
    return usage_ratio > FORCE_CLEANUP_THRESHOLD

def optimize_model_for_inference(model):
    """优化模型以减少推理时的显存占用"""
    if model is None:
        return model
    
    # 设置为评估模式
    model.eval()
    
    # 启用梯度检查点（如果支持）
    if ENABLE_GRADIENT_CHECKPOINTING and hasattr(model, 'encoder'):
        try:
            if hasattr(model.encoder, 'use_gradient_checkpointing'):
                model.encoder.use_gradient_checkpointing = True
            elif hasattr(model.encoder, 'gradient_checkpointing'):
                model.encoder.gradient_checkpointing = True
        except Exception as e:
            print(f"无法启用梯度检查点: {e}")
    
    # 禁用自动求导（推理时不需要梯度）
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def create_streaming_config():
    """创建流式处理配置以减少显存占用"""
    return {
        'batch_size': 1,  # 单批处理减少显存占用
        'num_workers': 0,  # 避免多进程带来的额外内存开销
        'pin_memory': False,  # 不使用锁页内存以节省系统内存
        'drop_last': False,
        'persistent_workers': False  # 不保持worker进程
    }

def load_model_if_needed():
    """按需加载模型，如果模型未加载，则进行加载。"""
    global asr_model, cuda_available
    # 使用锁确保多线程环境下模型只被加载一次
    with model_lock:
        if asr_model is None:
            print("="*50)
            print("模型当前未加载，正在初始化...")
            # 新模型默认：v3；支持通过环境变量覆盖
            model_id = os.environ.get('MODEL_ID', 'nvidia/parakeet-tdt-0.6b-v3').strip()
            model_local_path_env = os.environ.get('MODEL_LOCAL_PATH', '').strip()
            print(f"首选模型: {model_id}")
            try:
                # 首先检查CUDA兼容性
                cuda_available = check_cuda_compatibility()
                
                # 确保numba缓存目录存在
                numba_cache_dir = os.environ.get('NUMBA_CACHE_DIR', '/tmp/numba_cache')
                if not os.path.exists(numba_cache_dir):
                    os.makedirs(numba_cache_dir, exist_ok=True)
                    os.chmod(numba_cache_dir, 0o777)
                
                # 本地优先策略：优先使用 MODEL_LOCAL_PATH ；否则尝试常见文件名；否则走 HF 自动下载
                candidate_local_paths = []
                if model_local_path_env:
                    candidate_local_paths.append(model_local_path_env)
                # 新版 v3 默认文件名（若用户手动下载 .nemo）
                candidate_local_paths.append("/app/models/parakeet-tdt-0.6b-v3.nemo")
                # 兼容旧版 v2 文件名（向后兼容）
                candidate_local_paths.append("/app/models/parakeet-tdt-0.6b-v2.nemo")

                model_path = next((p for p in candidate_local_paths if os.path.exists(p)), None)

                if cuda_available:
                    print(f"✅ 检测到兼容的CUDA环境，将使用 GPU 加速并开启半精度(FP16)优化。")
                    
                    # 设置 Tensor Core 优化
                    setup_tensor_core_optimization()
                    optimize_tensor_operations()
                    
                    # 显示 GPU 和 Tensor Core 信息
                    device_info = torch.cuda.get_device_properties(0)
                    print(f"GPU: {device_info.name}")
                    print(f"Tensor Core 支持: {get_tensor_core_info()}")
                    
                    # 先在CPU上加载模型，然后转移到GPU并启用FP16
                    if model_path:
                        # 本地 .nemo
                        # 检查文件权限
                        if not os.access(model_path, os.R_OK):
                            raise PermissionError(f"无法读取模型文件: {model_path}，请检查文件权限。")
                        print(f"从本地 .nemo 恢复: {model_path}")
                        loaded_model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path, map_location=torch.device('cpu'))
                    else:
                        # 从 HF 自动下载或尝试直接抓取 .nemo 文件到本地缓存目录
                        print(f"尝试从 Hugging Face 获取模型文件: {model_id}")
                        os.makedirs('/app/models', exist_ok=True)
                        downloaded_path = None
                        try:
                            if HfApi is None:
                                raise RuntimeError("huggingface_hub not available")
                            api = HfApi()
                            repo_files = api.list_repo_files(model_id)
                            nemo_files = [f for f in repo_files if f.endswith('.nemo')]
                            if nemo_files:
                                target_fname = nemo_files[0]
                                print(f"发现远端 .nemo 文件: {target_fname}，开始下载...")
                                downloaded_path = hf_hub_download(repo_id=model_id, filename=target_fname, cache_dir='/app/models')
                                print(f"已下载模型到: {downloaded_path}")
                            else:
                                print("远端仓库未发现 .nemo 文件，回退到 NeMo.from_pretrained() 方法加载")
                        except Exception as e:
                            print(f"尝试从 Hugging Face 获取 .nemo 失败: {e}")

                        if downloaded_path and os.path.exists(downloaded_path):
                            loaded_model = nemo_asr.models.ASRModel.restore_from(restore_path=downloaded_path, map_location=torch.device('cpu'))
                        else:
                            print(f"使用 NeMo 的 from_pretrained 加载模型: {model_id}")
                            loaded_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
                    loaded_model = loaded_model.cuda()
                    loaded_model = loaded_model.half()
                    
                    # 应用推理优化
                    loaded_model = optimize_model_for_inference(loaded_model)
                    
                    # 显示显存使用情况
                    allocated, reserved, total = get_gpu_memory_usage()
                    print(f"模型加载后显存使用: {allocated:.2f}GB / {total:.2f}GB ({allocated/total*100:.1f}%)")
                else:
                    print("🔄 使用 CPU 模式运行。")
                    print("注意: CPU模式下推理速度会较慢，建议使用兼容的GPU。")
                    if model_path:
                        # 本地 .nemo
                        if not os.access(model_path, os.R_OK):
                            raise PermissionError(f"无法读取模型文件: {model_path}，请检查文件权限。")
                        print(f"从本地 .nemo 恢复: {model_path}")
                        loaded_model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)
                    else:
                        # 从 HF 自动下载或尝试直接抓取 .nemo 文件到本地缓存目录（CPU 分支）
                        print(f"尝试从 Hugging Face 获取模型文件: {model_id}")
                        os.makedirs('/app/models', exist_ok=True)
                        downloaded_path = None
                        try:
                            if HfApi is None:
                                raise RuntimeError("huggingface_hub not available")
                            api = HfApi()
                            repo_files = api.list_repo_files(model_id)
                            nemo_files = [f for f in repo_files if f.endswith('.nemo')]
                            if nemo_files:
                                target_fname = nemo_files[0]
                                print(f"发现远端 .nemo 文件: {target_fname}，开始下载...")
                                downloaded_path = hf_hub_download(repo_id=model_id, filename=target_fname, cache_dir='/app/models')
                                print(f"已下载模型到: {downloaded_path}")
                            else:
                                print("远端仓库未发现 .nemo 文件，回退到 NeMo.from_pretrained() 方法加载")
                        except Exception as e:
                            print(f"尝试从 Hugging Face 获取 .nemo 失败: {e}")

                        if downloaded_path and os.path.exists(downloaded_path):
                            loaded_model = nemo_asr.models.ASRModel.restore_from(restore_path=downloaded_path)
                        else:
                            print(f"使用 NeMo 的 from_pretrained 加载模型: {model_id}")
                            loaded_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
                    loaded_model = optimize_model_for_inference(loaded_model)
                
                # 配置解码策略（若模型支持）
                try:
                    configure_decoding_strategy(loaded_model)
                except Exception as e:
                    print(f"⚠️ 配置解码策略失败，将使用默认解码: {e}")

                asr_model = loaded_model
                print("✅ NeMo ASR 模型加载成功！")
                print("="*50)
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                print("="*50)
                import traceback
                traceback.print_exc()
                # 向上抛出异常，以便接口可以捕获并返回错误信息
                raise e
    return asr_model

def predownload_model_artifacts():
    """在后台下载模型文件到本地缓存目录，但不加载到内存。
    这个函数用于在启用懒加载时提前把大文件拉取到 `/app/models`，以缩短后续首次加载延时。
    """
    try:
        model_id = os.environ.get('MODEL_ID', 'nvidia/parakeet-tdt-0.6b-v3').strip()
        model_local_path_env = os.environ.get('MODEL_LOCAL_PATH', '').strip()
        print(f"[predownload] 启动模型预下载检查: {model_id}")

        # 本地优先：如果已存在本地文件则无需下载
        candidate_local_paths = []
        if model_local_path_env:
            candidate_local_paths.append(model_local_path_env)
        candidate_local_paths.append('/app/models/parakeet-tdt-0.6b-v3.nemo')
        candidate_local_paths.append('/app/models/parakeet-tdt-0.6b-v2.nemo')
        for p in candidate_local_paths:
            if p and os.path.exists(p):
                print(f"[predownload] 发现本地模型文件，无需下载: {p}")
                return

        # 创建缓存目录
        os.makedirs('/app/models', exist_ok=True)

        # 尝试使用 huggingface_hub 下载远端 .nemo 文件（仅下载，不恢复/加载）
        if HfApi is None:
            print("[predownload] huggingface_hub 不可用，跳过预下载")
            return

        try:
            api = HfApi()
            repo_files = api.list_repo_files(model_id)
            nemo_files = [f for f in repo_files if f.endswith('.nemo')]
            if not nemo_files:
                print(f"[predownload] 远端仓库未发现 .nemo 文件: {model_id}，跳过预下载")
                return
            target_fname = nemo_files[0]
            print(f"[predownload] 发现远端 .nemo 文件: {target_fname}，开始下载到 /app/models ...")
            try:
                downloaded_path = hf_hub_download(repo_id=model_id, filename=target_fname, cache_dir='/app/models')
                if downloaded_path and os.path.exists(downloaded_path):
                    print(f"[predownload] 已下载模型文件: {downloaded_path}")
                else:
                    print(f"[predownload] 下载返回路径无效或不存在: {downloaded_path}")
            except Exception as e:
                print(f"[predownload] hf_hub_download 失败: {e}")
        except Exception as e:
            print(f"[predownload] 查询远端仓库文件列表失败: {e}")
    except Exception as e:
        print(f"[predownload] 预下载线程异常: {e}")

def unload_model():
    """从内存/显存中卸载模型。"""
    global asr_model, last_request_time, cuda_available
    with model_lock:
        if asr_model is not None:
            print(f"模型闲置超过 {IDLE_TIMEOUT_MINUTES} 分钟，正在从内存中卸载...")
            
            # 显示卸载前的显存使用
            if cuda_available:
                allocated_before, _, total = get_gpu_memory_usage()
                print(f"卸载前显存使用: {allocated_before:.2f}GB / {total:.2f}GB")
            
            asr_model = None
            
            # 卸载后立即执行深度清理
            idle_deep_memory_cleanup()
            
            # 显示卸载后的显存使用
            if cuda_available:
                allocated_after, _, total = get_gpu_memory_usage()
                print(f"卸载后显存使用: {allocated_after:.2f}GB / {total:.2f}GB")
                print(f"释放显存: {allocated_before - allocated_after:.2f}GB")
            
            last_request_time = None # 重置计时器，防止重复卸载
            print("✅ 模型已成功卸载并完成深度清理。")

def model_cleanup_checker():
    """后台线程，周期性检查模型是否闲置过久并执行卸载。"""
    last_cleanup_time = datetime.datetime.now()
    
    while True:
        # 根据系统状态自适应调整检查间隔
        current_time = datetime.datetime.now()
        
        # 基础监控间隔
        sleep_interval = IDLE_MONITORING_INTERVAL
        
        if asr_model is not None and last_request_time is not None:
            idle_duration = (current_time - last_request_time).total_seconds()
            
            # 检查是否需要卸载模型
            if idle_duration > IDLE_TIMEOUT_MINUTES * 60:
                unload_model()
                # 模型卸载后立即执行深度清理
                idle_deep_memory_cleanup()
                last_cleanup_time = current_time
            
            # 根据闲置时间调整检查频率
            elif idle_duration > IDLE_DEEP_CLEANUP_THRESHOLD:
                # 长时间闲置时，降低检查频率但执行深度清理
                sleep_interval = max(120, IDLE_MONITORING_INTERVAL * 2)  # 最少2分钟间隔
                if (current_time - last_cleanup_time).total_seconds() > IDLE_MEMORY_CLEANUP_INTERVAL:
                    idle_deep_memory_cleanup()
                    last_cleanup_time = current_time
            
            elif idle_duration > IDLE_MEMORY_CLEANUP_INTERVAL:
                # 中等闲置时间，执行轻量清理
                if (current_time - last_cleanup_time).total_seconds() > IDLE_MEMORY_CLEANUP_INTERVAL:
                    if AGGRESSIVE_MEMORY_CLEANUP and should_force_cleanup():
                        print("🧹 执行闲置期间内存清理...")
                        aggressive_memory_cleanup()
                        last_cleanup_time = current_time
        
        else:
            # 模型未加载或未有请求时，使用较长的检查间隔并定期清理
            sleep_interval = max(120, IDLE_MONITORING_INTERVAL * 3)
            if (current_time - last_cleanup_time).total_seconds() > IDLE_MEMORY_CLEANUP_INTERVAL * 2:
                aggressive_memory_cleanup()
                last_cleanup_time = current_time
        
        time.sleep(sleep_interval)


# --- Flask 应用初始化 ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/app/temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  

# --- 辅助函数 ---
def get_audio_duration(file_path: str) -> float:
    """使用 ffprobe 获取音频文件的时长（秒）"""
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"无法获取文件 '{file_path}' 的时长: {e}")
        return 0.0

def format_srt_time(seconds: float) -> str:
    """将秒数格式化为 SRT 时间戳格式 HH:MM:SS,ms"""
    delta = datetime.timedelta(seconds=seconds)
    # 格式化为 0:00:05.123000
    s = str(delta)
    # 分割秒和微秒
    if '.' in s:
        parts = s.split('.')
        integer_part = parts[0]
        fractional_part = parts[1][:3] # 取前三位毫秒
    else:
        integer_part = s
        fractional_part = "000"

    # 填充小时位
    if len(integer_part.split(':')) == 2:
        integer_part = "0:" + integer_part
    
    return f"{integer_part},{fractional_part}"


def segments_to_srt(segments: list) -> str:
    """将 NeMo 的分段时间戳转换为 SRT 格式字符串"""
    srt_content = []
    for i, segment in enumerate(segments):
        start_time = format_srt_time(segment['start'])
        end_time = format_srt_time(segment['end'])
        text = segment['segment'].strip()
        if text and PREFERRED_LINE_LENGTH > 0:
            text = wrap_text_for_display(
                text,
                preferred_line_length=PREFERRED_LINE_LENGTH,
                max_lines=MAX_SUBTITLE_LINES,
            )
        
        if text: # 仅添加有内容的字幕
            srt_content.append(str(i + 1))
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("") # 空行分隔
            
    return "\n".join(srt_content)


def parse_ffmpeg_silence_log(ffmpeg_stderr: str) -> list:
    """解析 ffmpeg silencedetect 输出，返回静音区间 [(start, end), ...]。"""
    import re
    silence_starts = []
    silence_intervals = []
    # silencedetect 输出示例:
    # [silencedetect @ 0x...] silence_start: 12.345
    # [silencedetect @ 0x...] silence_end: 13.789 | silence_duration: 1.444
    start_re = re.compile(r"silence_start:\s*([0-9.]+)")
    end_re = re.compile(r"silence_end:\s*([0-9.]+)")
    for line in ffmpeg_stderr.splitlines():
        m = start_re.search(line)
        if m:
            silence_starts.append(float(m.group(1)))
            continue
        m = end_re.search(line)
        if m and silence_starts:
            start = silence_starts.pop(0)
            end = float(m.group(1))
            silence_intervals.append((start, end))
    return silence_intervals


def find_nearest_silence(target_time: float, silence_intervals: list, max_shift: float) -> float:
    """在 target_time 附近查找最近的静音边界，返回建议的切片开始时间。若未找到合适静音点，则返回 target_time。"""
    if not silence_intervals:
        return target_time
    best_time = target_time
    best_dist = max_shift + 1.0
    for start, end in silence_intervals:
        for edge in (start, end):
            dist = abs(edge - target_time)
            if dist < best_dist and dist <= max_shift:
                best_dist = dist
                best_time = edge
    return best_time


def detect_silences_with_ffmpeg(source_wav: str) -> list:
    """使用 ffmpeg silencedetect 检测静音区间。"""
    command = [
        'ffmpeg', '-hide_banner', '-nostats', '-i', source_wav,
        '-af', f'silencedetect=noise={SILENCE_THRESHOLD_DB}:d={MIN_SILENCE_DURATION}',
        '-f', 'null', '-' 
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    # 无论返回码如何，stderr 都包含 silencedetect 输出
    return parse_ffmpeg_silence_log(result.stderr)

# --- Flask 路由 ---

@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查端点 - 用于Docker健康检查和服务监控
    """
    try:
        current_time = datetime.datetime.now()
        # 检查基本服务状态
        health_status: Dict[str, Any] = {
            "status": "healthy",
            "timestamp": current_time.isoformat(),
            "service": "parakeet-api",
            "version": "1.0.0"
        }
        
        # 检查CUDA状态
        global cuda_available
        if cuda_available:
            try:
                allocated, reserved, total = get_gpu_memory_usage()
                health_status["gpu"] = {
                    "available": True,
                    "memory_allocated_gb": round(allocated, 2),
                    "memory_reserved_gb": round(reserved, 2),
                    "memory_total_gb": round(total, 2),
                    "memory_usage_percent": round((allocated/total)*100, 1) if total > 0 else 0,
                    "memory_reserved_percent": round((reserved/total)*100, 1) if total > 0 else 0
                }
            except Exception as e:
                health_status["gpu"] = {
                    "available": True,
                    "error": str(e)
                }
        else:
            health_status["gpu"] = {
                "available": False,
                "mode": "cpu"
            }
        
        # 检查模型状态和闲置信息
        model_info = {
            "loaded": asr_model is not None,
            "lazy_load": ENABLE_LAZY_LOAD
        }
        
        if last_request_time is not None:
            idle_seconds = (current_time - last_request_time).total_seconds()
            model_info["last_request_time"] = last_request_time.isoformat()
            model_info["idle_duration_seconds"] = round(idle_seconds, 1)
            model_info["idle_duration_minutes"] = round(idle_seconds / 60, 1)
            
            # 添加闲置状态分类
            if idle_seconds > IDLE_TIMEOUT_MINUTES * 60:
                model_info["idle_status"] = "ready_for_unload"
            elif idle_seconds > IDLE_DEEP_CLEANUP_THRESHOLD:
                model_info["idle_status"] = "deep_idle"
            elif idle_seconds > IDLE_MEMORY_CLEANUP_INTERVAL:
                model_info["idle_status"] = "idle"
            else:
                model_info["idle_status"] = "active"
        else:
            model_info["idle_status"] = "no_requests" if asr_model is not None else "unloaded"
        
        health_status["model"] = model_info
        
        # 检查系统资源使用
        memory = psutil.virtual_memory()
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
        except:
            cpu_percent = 0.0
            
        health_status["system"] = {
            "memory_usage_percent": memory.percent,
            "memory_available_gb": round(memory.available / 1024**3, 2),
            "memory_total_gb": round(memory.total / 1024**3, 2),
            "cpu_usage_percent": round(cpu_percent, 1)
        }
        
        # 添加资源优化配置状态
        health_status["optimization"] = {
            "aggressive_memory_cleanup": AGGRESSIVE_MEMORY_CLEANUP,
            "idle_timeout_minutes": IDLE_TIMEOUT_MINUTES,
            "idle_memory_cleanup_interval": IDLE_MEMORY_CLEANUP_INTERVAL,
            "idle_deep_cleanup_threshold": IDLE_DEEP_CLEANUP_THRESHOLD,
            "enable_idle_cpu_optimization": ENABLE_IDLE_CPU_OPTIMIZATION,
            "force_cleanup_threshold": FORCE_CLEANUP_THRESHOLD
        }
        
        return jsonify(health_status), 200
        
    except Exception as e:
        error_status = {
            "status": "unhealthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "error": str(e)
        }
        return jsonify(error_status), 500

@app.route('/health/simple', methods=['GET'])
def simple_health_check():
    """
    简单健康检查端点 - 仅返回HTTP 200状态
    """
    return "OK", 200

@app.route('/v1/audio/transcriptions', methods=['POST'])
def transcribe_audio():
    """
    兼容 OpenAI 的语音识别接口，支持长音频分片处理。
    """
    # --- -1. API Key 认证 ---
    if API_KEY:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header is missing or invalid. It must be in 'Bearer <key>' format."}), 401
        
        provided_key = auth_header.split(' ')[1]
        if provided_key != API_KEY:
            return jsonify({"error": "Invalid API key."}), 401

    # --- 0. 确保模型加载并更新时间戳 ---
    try:
        # 如果懒加载启用，则按需加载；否则，直接使用已加载的全局模型
        local_asr_model = load_model_if_needed() if ENABLE_LAZY_LOAD else asr_model
        if not local_asr_model:
             # 此情况涵盖了懒加载失败和预加载失败两种场景
             return jsonify({"error": "模型加载失败或尚未加载，无法处理请求"}), 500
    except Exception as e:
        return jsonify({"error": f"模型加载时发生严重错误: {e}"}), 500
    
    # 如果启用了懒加载，则更新最后请求时间
    if ENABLE_LAZY_LOAD:
        global last_request_time
        last_request_time = datetime.datetime.now()


    # --- 1. 基本校验 ---
    if 'file' not in request.files:
        return jsonify({"error": "请求中未找到文件部分"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    if not shutil.which('ffmpeg'):
        return jsonify({"error": "FFmpeg 未安装或未在系统 PATH 中"}), 500
    if not shutil.which('ffprobe'):
        return jsonify({"error": "ffprobe 未安装或未在系统 PATH 中"}), 500

    # 获取请求参数
    model_name = request.form.get('model', 'whisper-1')
    response_format = request.form.get('response_format', 'json')  # 支持 json, text, srt, verbose_json, vtt
    language = request.form.get('language', None)
    prompt = request.form.get('prompt', None)
    temperature = float(request.form.get('temperature', 0))
    
    print(f"接收到请求，模型: '{model_name}', 响应格式: '{response_format}', 语言: '{language}'")

    # --- 0.5 语言白名单校验（Whisper 兼容行为）---
    # 若客户端显式传入 language，我们只接受受支持的 25 种语言，否则直接拒绝
    detected_language = None  # 用于存储自动检测的语言
    if language:
        lang_norm = str(language).strip().lower().replace('_', '-')
        # 兼容像 "en-US" 这种区域码：只取主语言部分
        primary = lang_norm.split('-')[0]
        if primary not in SUPPORTED_LANG_CODES:
            # 与 Whisper 的风格保持一致：返回 400，并在 message 中提示不支持
            return jsonify({
                "error": {
                    "message": f"Unsupported language: {language}",
                    "type": "invalid_request_error",
                    "param": "language",
                    "code": "unsupported_language"
                }
            }), 400

    original_filename = secure_filename(str(file.filename or 'uploaded_file'))
    unique_id = str(uuid.uuid4())
    temp_original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{original_filename}")
    target_wav_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}.wav")
    
    # 用于清理所有临时文件的列表
    temp_files_to_clean = []

    try:
        # --- 2. 保存并统一转换为 16k 单声道 WAV ---
        file.save(temp_original_path)
        temp_files_to_clean.append(temp_original_path)
        
        print(f"[{unique_id}] 正在将 '{original_filename}' 转换为标准 WAV 格式...")
        # 可选前处理滤波器
        ffmpeg_filters = []
        if ENABLE_FFMPEG_DENOISE:
            ffmpeg_filters.append(DENOISE_FILTER)
        ffmpeg_command = [
            'ffmpeg', '-y', '-vn', '-sn', '-dn', '-i', temp_original_path,
            '-ac', '1', '-ar', '16000'
        ]
        if ffmpeg_filters:
            ffmpeg_command += ['-af', ','.join(ffmpeg_filters)]
        ffmpeg_command += [target_wav_path]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg 错误: {result.stderr}")
            return jsonify({"error": "文件转换失败", "details": result.stderr}), 500
        temp_files_to_clean.append(target_wav_path)

        # --- 2.5 自动语言检测和验证（未显式传 language 时）---
        if not language:
            try:
                lid_clip_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_lid.wav")
                temp_files_to_clean.append(lid_clip_path)
                # 取短片段进行快速转写
                clip_seconds = max(5, int(LID_CLIP_SECONDS))
                probe_dur = get_audio_duration(target_wav_path)
                if probe_dur > 0:
                    clip_seconds = min(clip_seconds, int(math.ceil(probe_dur)))
                clip_cmd = [
                    'ffmpeg', '-y', '-i', target_wav_path,
                    '-t', str(clip_seconds),
                    '-ac', '1', '-ar', '16000',
                    lid_clip_path
                ]
                _res = subprocess.run(clip_cmd, capture_output=True, text=True)
                if _res.returncode == 0 and os.path.exists(lid_clip_path):
                    # 仅文本推理（不开时间戳，降低开销）
                    with inference_semaphore:
                        lid_out = safe_transcribe(
                            local_asr_model,
                            lid_clip_path,
                            need_timestamps=False,
                            batch_size=1,
                            num_workers=0,
                        )
                    # 提取文本
                    lid_text = ""
                    if isinstance(lid_out, list) and lid_out:
                        first = lid_out[0]
                        try:
                            if hasattr(first, 'text') and first.text:
                                lid_text = str(first.text)
                            elif hasattr(first, 'segment') and first.segment:
                                lid_text = str(first.segment)
                            else:
                                lid_text = str(first)
                        except Exception:
                            lid_text = str(first)

                    # 用轻量文本语言识别做语言检测
                    if lid_text and lid_text.strip():
                        try:
                            try:
                                from langdetect import detect  # type: ignore
                            except Exception:
                                detect = None  # type: ignore
                            detected = None
                            if detect is not None:
                                detected = detect(lid_text)
                            # 若能检测到语言
                            if detected:
                                det_primary = str(detected).strip().lower().split('-')[0]
                                if det_primary:
                                    if det_primary in SUPPORTED_LANG_CODES:
                                        # 检测到支持的语言，存储用于后续使用
                                        detected_language = det_primary
                                        print(f"[{unique_id}] 自动检测到语言: {detected_language}")
                                    elif ENABLE_AUTO_LANGUAGE_REJECTION:
                                        # 检测到不支持的语言且启用了自动拒绝
                                        return jsonify({
                                            "error": {
                                                "message": f"Unsupported language: {detected}",
                                                "type": "invalid_request_error",
                                                "param": "language",
                                                "code": "unsupported_language"
                                            }
                                        }), 400
                                    else:
                                        # 检测到不支持的语言但未启用自动拒绝，默认为英语
                                        detected_language = "en"
                                        print(f"[{unique_id}] 检测到不支持的语言 {detected}，默认使用英语")
                        except Exception as _e:
                            # 检测失败不影响主流程，默认使用英语
                            print(f"[{unique_id}] 语言自动检测失败，默认使用英语: {_e}")
                            detected_language = "en"
                    else:
                        # 无法提取文本，默认使用英语
                        print(f"[{unique_id}] 无法提取文本进行语言检测，默认使用英语")
                        detected_language = "en"
            except Exception as _e:
                print(f"[{unique_id}] 自动语言检测阶段异常，默认使用英语: {_e}")
                detected_language = "en"

        # --- 3. 音频切片 (Chunking) ---
        # 动态调整chunk大小基于显存使用情况
        heavy_ts_request = response_format in ['srt', 'vtt', 'verbose_json']
        if cuda_available:
            allocated, _, total = get_gpu_memory_usage()
            memory_usage_ratio = allocated / total if total > 0 else 0
            
            if memory_usage_ratio > 0.6:  # 如果显存使用超过60%
                # 减少chunk大小以降低显存压力
                adjusted_chunk_minutes = max(3, CHUNK_MINITE - 2)
                print(f"[{unique_id}] 显存使用较高({memory_usage_ratio*100:.1f}%)，调整chunk大小从 {CHUNK_MINITE} 分钟到 {adjusted_chunk_minutes} 分钟")
                CHUNK_DURATION_SECONDS = adjusted_chunk_minutes * 60
            else:
                CHUNK_DURATION_SECONDS = CHUNK_MINITE * 60
            # 为 ≤8~12GB 显存设备或需要时间戳的请求设置更保守的上限，避免注意力矩阵 OOM
            try:
                vram_gb = total
                cap_env = os.environ.get('CHUNK_SECONDS_CAP', '').strip()
                if cap_env:
                    cap_sec = int(float(cap_env))
                else:
                    if vram_gb <= 8.5:
                        cap_sec = 180 if heavy_ts_request else 240
                    elif vram_gb <= 12.0:
                        cap_sec = 300 if heavy_ts_request else 480
                    else:
                        cap_sec = 600
                if CHUNK_DURATION_SECONDS > cap_sec:
                    print(f"[{unique_id}] 基于GPU显存({vram_gb:.1f}GB){'且需时间戳' if heavy_ts_request else ''}，限制chunk时长为 {cap_sec}s")
                    CHUNK_DURATION_SECONDS = cap_sec
            except Exception:
                pass
        else:
            # CPU模式下使用较小的chunk以避免内存不足
            cpu_chunk_minutes = max(3, CHUNK_MINITE // 2)  # CPU模式减半chunk大小
            print(f"[{unique_id}] CPU模式，调整chunk大小到 {cpu_chunk_minutes} 分钟")
            CHUNK_DURATION_SECONDS = cpu_chunk_minutes * 60
            # CPU 模式也设置上限，尤其在需要时间戳时
            try:
                cap_env = os.environ.get('CHUNK_SECONDS_CAP', '').strip()
                cap_sec = int(float(cap_env)) if cap_env else (180 if heavy_ts_request else 240)
                if CHUNK_DURATION_SECONDS > cap_sec:
                    print(f"[{unique_id}] CPU模式限制chunk时长为 {cap_sec}s")
                    CHUNK_DURATION_SECONDS = cap_sec
            except Exception:
                pass
            
        total_duration = get_audio_duration(target_wav_path)
        if total_duration == 0:
            return jsonify({"error": "无法处理时长为0的音频"}), 400

        # 检查是否需要切片，如果音频时长小于切片阈值，则直接处理
        if total_duration <= CHUNK_DURATION_SECONDS:
            print(f"[{unique_id}] 文件总时长: {total_duration:.2f}s. 小于切片阈值({CHUNK_DURATION_SECONDS}s)，无需切片。")
            chunk_paths = [target_wav_path]
            chunk_info_list = [{'start': 0, 'end': total_duration, 'duration': total_duration}]
            num_chunks = 1
        else:
            # 使用重叠分割策略
            if ENABLE_OVERLAP_CHUNKING:
                print(f"[{unique_id}] 启用重叠分割模式，重叠时长: {CHUNK_OVERLAP_SECONDS}s")
                chunk_info_list = create_overlap_chunks(total_duration, CHUNK_DURATION_SECONDS, CHUNK_OVERLAP_SECONDS)
            else:
                # 传统硬分割
                num_chunks = math.ceil(total_duration / CHUNK_DURATION_SECONDS)
                chunk_info_list = []
                for i in range(num_chunks):
                    start_time = i * CHUNK_DURATION_SECONDS
                    end_time = min(start_time + CHUNK_DURATION_SECONDS, total_duration)
                    chunk_info_list.append({
                        'start': start_time,
                        'end': end_time, 
                        'duration': end_time - start_time
                    })
            
            chunk_paths = []
            num_chunks = len(chunk_info_list)
            print(f"[{unique_id}] 文件总时长: {total_duration:.2f}s. 将切分为 {num_chunks} 个片段。")
            
            # 若启用静音对齐，则预先检测静音区间
            silence_intervals = []
            if ENABLE_SILENCE_ALIGNED_CHUNKING and total_duration > CHUNK_DURATION_SECONDS:
                print(f"[{unique_id}] 检测静音区间用于分割对齐: noise={SILENCE_THRESHOLD_DB}, min_dur={MIN_SILENCE_DURATION}s")
                silence_intervals = detect_silences_with_ffmpeg(target_wav_path)
                print(f"[{unique_id}] 共检测到 {len(silence_intervals)} 段静音区间")

            for i, chunk_info in enumerate(chunk_info_list):
                chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_chunk_{i}.wav")
                chunk_paths.append(chunk_path)
                temp_files_to_clean.append(chunk_path)
                
                start_time = chunk_info['start']
                # 将切片开始对齐到最近静音边界（不超过最大偏移）
                if ENABLE_SILENCE_ALIGNED_CHUNKING and silence_intervals:
                    aligned_start = find_nearest_silence(start_time, silence_intervals, SILENCE_MAX_SHIFT_SECONDS)
                    if aligned_start != start_time:
                        print(f"[{unique_id}] 切片{i+1} 开始时间 {start_time:.2f}s 对齐至静音 {aligned_start:.2f}s")
                        # 同时调整该chunk的结束，保持 duration 不变
                        shift = aligned_start - start_time
                        start_time = max(0.0, aligned_start)
                        chunk_info['start'] = start_time
                        chunk_info['end'] = min(total_duration, chunk_info['end'] + shift)
                        chunk_info['duration'] = chunk_info['end'] - chunk_info['start']
                duration = chunk_info['duration']
                
                print(f"[{unique_id}] 正在创建切片 {i+1}/{num_chunks} ({start_time:.1f}s - {chunk_info['end']:.1f}s)...")
                chunk_command = [
                    'ffmpeg', '-y', '-i', target_wav_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c', 'copy',
                    chunk_path
                ]
                result = subprocess.run(chunk_command, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"[{unique_id}] ⚠️ 创建切片 {i+1} 时出现警告: {result.stderr}")
                    # 继续处理，不中断
            
        # --- 4. 循环转录并合并结果 ---
        all_segments = []
        all_words = []
        chunk_boundaries = []
        # 仅在需要 SRT/VTT/verbose_json 时请求时间戳，减少显存与计算
        need_timestamps = response_format in ['srt', 'vtt', 'verbose_json']
        # 当需要进行长字幕切分且启用了基于词时间戳的切分时，也尝试收集词级时间戳
        collect_word_timestamps = (response_format == 'verbose_json') or (SPLIT_LONG_SUBTITLES and ENABLE_WORD_TIMESTAMPS_FOR_SPLIT)
        full_text_parts = []  # 当不需要时间戳时，直接收集文本

        for i, (chunk_path, chunk_info) in enumerate(zip(chunk_paths, chunk_info_list)):
            print(f"[{unique_id}] 正在转录切片 {i+1}/{num_chunks}...")
            
            # 检查显存使用情况，如果过高则强制清理
            if should_force_cleanup():
                print(f"[{unique_id}] 显存使用过高，执行强制清理...")
                aggressive_memory_cleanup()
            
            # 显示当前显存/内存使用
            if cuda_available:
                allocated, _, total = get_gpu_memory_usage()
                print(f"[{unique_id}] 处理切片 {i+1} 前显存使用: {allocated:.2f}GB / {total:.2f}GB")
            else:
                # 显示CPU内存使用
                memory = psutil.virtual_memory()
                print(f"[{unique_id}] 处理切片 {i+1} 前内存使用: {memory.used/1024**3:.2f}GB / {memory.total/1024**3:.2f}GB ({memory.percent:.1f}%)")
            
            # 对当前切片进行转录
            # 使用 with torch.cuda.amp.autocast() 在半精度下运行推理
            # 推理模式进一步降低内存/开销，并发控制避免 OOM
            with inference_semaphore:
                output = safe_transcribe(
                    local_asr_model,
                    chunk_path,
                    need_timestamps=need_timestamps,
                    batch_size=TRANSCRIBE_BATCH_SIZE,
                    num_workers=TRANSCRIBE_NUM_WORKERS,
                )

            # 立即进行内存清理
            if AGGRESSIVE_MEMORY_CLEANUP:
                aggressive_memory_cleanup()
            else:
                if cuda_available:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                gc.collect()
            
            # 记录chunk边界用于后续合并
            chunk_start_offset = chunk_info['start']
            chunk_boundaries.append(chunk_start_offset)
            
            if need_timestamps:
                if output and getattr(output[0], 'timestamp', None):
                    # 修正并收集 segment 时间戳
                    if 'segment' in output[0].timestamp:
                        for seg in output[0].timestamp['segment']:
                            seg['start'] += chunk_start_offset
                            seg['end'] += chunk_start_offset
                            all_segments.append(seg)
                    # 修正并收集 word 时间戳（仅在 verbose_json 需要）
                    if collect_word_timestamps and 'word' in output[0].timestamp:
                        for word in output[0].timestamp['word']:
                            word['start'] += chunk_start_offset
                            word['end'] += chunk_start_offset
                            all_words.append(word)
                else:
                    # 某些模型/配置可能不返回时间戳，尝试直接文本回退
                    if isinstance(output, list) and output:
                        full_text_parts.append(str(output[0]))
            else:
                # 不需要时间戳，直接取文本
                if isinstance(output, list) and output:
                    # NeMo 返回的元素可能是 Hypothesis 对象，优先提取 .text 或 .segment 字段
                    first = output[0]
                    try:
                        # 优先使用常见属性
                        if hasattr(first, 'text') and first.text:
                            full_text_parts.append(str(first.text))
                        elif hasattr(first, 'segment') and first.segment:
                            full_text_parts.append(str(first.segment))
                        else:
                            full_text_parts.append(str(first))
                    except Exception:
                        full_text_parts.append(str(first))
            
            # 释放临时输出引用
            try:
                del output
            except Exception:
                pass
            # 立即删除已处理的chunk文件以节省磁盘空间和内存
            if num_chunks > 1 and os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                    temp_files_to_clean.remove(chunk_path)
                    print(f"[{unique_id}] 已删除处理完成的切片文件: chunk_{i}")
                except Exception as e:
                    print(f"[{unique_id}] 删除切片文件时出错: {e}")

        print(f"[{unique_id}] 所有切片转录完成，正在合并结果。")
        
        # --- 4.5. 处理重叠区域并合并segments ---
        if ENABLE_OVERLAP_CHUNKING and len(chunk_boundaries) > 1:
            print(f"[{unique_id}] 处理重叠区域，去除重复内容...")
            all_segments = merge_overlapping_segments(all_segments, chunk_boundaries, CHUNK_OVERLAP_SECONDS)
            print(f"[{unique_id}] 重叠处理完成，最终segments数量: {len(all_segments)}")

        # --- 4.6. 字幕后处理：合并/延长过短字幕，避免闪烁 ---
        if MERGE_SHORT_SUBTITLES and all_segments:
            before_cnt = len(all_segments)
            all_segments = enforce_min_subtitle_duration(
                all_segments,
                min_duration=MIN_SUBTITLE_DURATION_SECONDS,
                merge_max_gap=SHORT_SUBTITLE_MERGE_MAX_GAP_SECONDS,
                min_chars=SHORT_SUBTITLE_MIN_CHARS,
                min_gap=SUBTITLE_MIN_GAP_SECONDS,
            )
            print(f"[{unique_id}] 字幕后处理完成：{before_cnt} -> {len(all_segments)} 段（最小时长 {MIN_SUBTITLE_DURATION_SECONDS}s）")

        # --- 4.7. 长字幕拆分（按时长/字符数限制） ---
        if SPLIT_LONG_SUBTITLES and all_segments:
            before_cnt = len(all_segments)
            all_segments = split_and_wrap_long_subtitles(
                segments=all_segments,
                words=all_words if collect_word_timestamps else None,
                max_duration=MAX_SUBTITLE_DURATION_SECONDS,
                max_chars=MAX_SUBTITLE_CHARS_PER_SEGMENT,
                preferred_line_length=PREFERRED_LINE_LENGTH,
                max_lines=MAX_SUBTITLE_LINES,
                punctuation=SUBTITLE_SPLIT_PUNCTUATION,
            )
            print(f"[{unique_id}] 长字幕拆分完成：{before_cnt} -> {len(all_segments)} 段（最大时长 {MAX_SUBTITLE_DURATION_SECONDS}s, 最大字符 {MAX_SUBTITLE_CHARS_PER_SEGMENT}）")

        # --- 5. 格式化最终输出 ---
        # 如果既没有时间戳段，也没有直接文本，则视为失败；
        # 否则即使没有 segments（例如模型只返回纯文本），也应返回文本结果。
        if not all_segments and not full_text_parts:
            return jsonify({"error": "转录失败，模型未返回任何有效内容"}), 500

        # 构建完整的转录文本
        full_text = " ".join([seg['segment'].strip() for seg in all_segments if seg['segment'].strip()])
        
        # 根据 response_format 返回不同格式
        if response_format == 'text':
            if not full_text:
                # 当未启用时间戳且直接收集文本
                full_text = " ".join(full_text_parts) if full_text_parts else ""
            return Response(full_text, mimetype='text/plain')
        elif response_format == 'srt':
            srt_result = segments_to_srt(all_segments)
            return Response(srt_result, mimetype='text/plain')
        elif response_format == 'vtt':
            vtt_result = segments_to_vtt(all_segments)
            return Response(vtt_result, mimetype='text/plain')
        elif response_format == 'verbose_json':
            # 详细的 JSON 格式，包含更多信息
            response_data = {
                "task": "transcribe",
                "language": language or detected_language or "en",
                "duration": total_duration,
                "text": full_text,
                "segments": [
                    {
                        "id": i,
                        "seek": int(seg['start'] * 100),  # 转换为 centiseconds
                        "start": seg['start'],
                        "end": seg['end'],
                        "text": seg['segment'].strip(),
                        "tokens": [],  # NeMo 不提供 tokens，留空
                        "temperature": temperature,
                        "avg_logprob": -0.5,  # 模拟值
                        "compression_ratio": 1.0,  # 模拟值
                        "no_speech_prob": 0.0,  # 模拟值
                        "words": [
                            {
                                "word": word['word'],
                                "start": word['start'],
                                "end": word['end'],
                                "probability": 0.9  # 模拟值
                            }
                            for word in all_words 
                            if word['start'] >= seg['start'] and word['end'] <= seg['end']
                        ] if all_words else []
                    }
                    for i, seg in enumerate(all_segments) if seg['segment'].strip()
                ]
            }
            return jsonify(response_data)
        else:
            # 默认 JSON 格式 (response_format == 'json')
            if not all_segments:
                # 当未启用时间戳，text 来自 direct 输出
                if not full_text:
                    full_text = " ".join(full_text_parts) if full_text_parts else ""
            response_data = {"text": full_text}
            return jsonify(response_data)

    except Exception as e:
        print(f"处理过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "服务器内部错误", "details": str(e)}), 500
    finally:
        # --- 6. 清理所有临时文件 ---
        print(f"[{unique_id}] 清理临时文件...")
        for f_path in temp_files_to_clean:
            if os.path.exists(f_path):
                os.remove(f_path)
        print(f"[{unique_id}] 临时文件已清理。")
        
        # --- 7. 强制清理内存，避免累积 ---
        print(f"[{unique_id}] 执行最终内存清理...")
        if cuda_available:
            allocated_before, _, total = get_gpu_memory_usage()
            print(f"[{unique_id}] 清理前显存使用: {allocated_before:.2f}GB / {total:.2f}GB")
        else:
            memory_before = psutil.virtual_memory()
            print(f"[{unique_id}] 清理前内存使用: {memory_before.used/1024**3:.2f}GB / {memory_before.total/1024**3:.2f}GB")
        
        aggressive_memory_cleanup()
        
        if cuda_available:
            allocated_after, _, total = get_gpu_memory_usage()
            print(f"[{unique_id}] 清理后显存使用: {allocated_after:.2f}GB / {total:.2f}GB")
            if allocated_before > 0:
                print(f"[{unique_id}] 释放显存: {allocated_before - allocated_after:.2f}GB")
        else:
            memory_after = psutil.virtual_memory()
            print(f"[{unique_id}] 清理后内存使用: {memory_after.used/1024**3:.2f}GB / {memory_after.total/1024**3:.2f}GB")
        print(f"[{unique_id}] 内存清理完成。")


def segments_to_vtt(segments: list) -> str:
    """将 NeMo 的分段时间戳转换为 VTT 格式字符串"""
    vtt_content = ["WEBVTT", ""]
    
    for i, segment in enumerate(segments):
        start_time = format_vtt_time(segment['start'])
        end_time = format_vtt_time(segment['end'])
        text = segment['segment'].strip()
        if text and PREFERRED_LINE_LENGTH > 0:
            text = wrap_text_for_display(
                text,
                preferred_line_length=PREFERRED_LINE_LENGTH,
                max_lines=MAX_SUBTITLE_LINES,
            )
        
        if text:  # 仅添加有内容的字幕
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text)
            vtt_content.append("")  # 空行分隔
            
    return "\n".join(vtt_content)


def format_vtt_time(seconds: float) -> str:
    """将秒数格式化为 VTT 时间戳格式 HH:MM:SS.mmm"""
    delta = datetime.timedelta(seconds=seconds)
    # 格式化为 0:00:05.123000
    s = str(delta)
    # 分割秒和微秒
    if '.' in s:
        parts = s.split('.')
        integer_part = parts[0]
        fractional_part = parts[1][:3]  # 取前三位毫秒
    else:
        integer_part = s
        fractional_part = "000"

    # 填充小时位
    if len(integer_part.split(':')) == 2:
        integer_part = "0:" + integer_part
    
    return f"{integer_part}.{fractional_part}"


def wrap_text_for_display(text: str, preferred_line_length: int, max_lines: int) -> str:
    """将单行文本按字数软换行为最多 max_lines 行，尽量在空格或标点处断行。
    若文本超过行数限制，后续仍保留但不强制增加行数（SRT/VTT可多行）。
    """
    if preferred_line_length <= 0 or max_lines <= 0:
        return text
    import re
    words = re.findall(r"\S+|\s+", text)
    lines = []
    current = ""
    for token in words:
        tentative = current + token
        if len(tentative.strip()) <= preferred_line_length or not current:
            current = tentative
        else:
            lines.append(current.strip())
            current = token.lstrip()
            if len(lines) >= max_lines - 1:
                break
    if current.strip():
        lines.append(current.strip())
    return "\n".join(lines)


def split_and_wrap_long_subtitles(
    segments: list,
    words: list | None,
    max_duration: float,
    max_chars: int,
    preferred_line_length: int,
    max_lines: int,
    punctuation: str,
) -> list:
    """按时长与字符数将过长字幕拆分为多条，并对每条文本进行换行。
    - 若提供 words（词级时间戳），优先在词边界拆分；否则退化为按标点/字符切分。
    """
    if not segments:
        return []

    # 建立每段内的 words 索引（若提供）
    words_by_range: list[list] = []
    if words:
        for seg in segments:
            start, end = seg.get('start', 0.0), seg.get('end', 0.0)
            seg_words = [w for w in words if w.get('start', 0.0) >= start and w.get('end', 0.0) <= end]
            words_by_range.append(seg_words)
    else:
        words_by_range = [[] for _ in segments]

    import re
    punct_set = set(punctuation)

    def split_points_by_chars(text: str) -> list[int]:
        points: list[int] = []
        last = 0
        while last + max_chars < len(text):
            cut = last + max_chars
            # 尽量向左回退到最近的空格或标点
            back = cut
            while back > last and text[back - 1] not in punct_set and not text[back - 1].isspace():
                back -= 1
            if back == last:
                back = cut
            points.append(back)
            last = back
        return points

    new_segments: list = []
    for seg, seg_words in zip(segments, words_by_range):
        start = float(seg.get('start', 0.0))
        end = float(seg.get('end', 0.0))
        text = str(seg.get('segment', '')).strip()
        if not text:
            continue

        duration = max(0.0, end - start)
        too_long_by_time = duration > max_duration
        too_long_by_chars = len(text) > max_chars

        if not too_long_by_time and not too_long_by_chars:
            seg_copy = dict(seg)
            seg_copy['segment'] = wrap_text_for_display(text, preferred_line_length, max_lines)
            new_segments.append(seg_copy)
            continue

        # 计算应拆分的片段数（时间/字符双约束）
        parts_by_time = max(1, int(math.ceil(duration / max_duration))) if max_duration > 0 else 1
        parts_by_chars = max(1, int(math.ceil(len(text) / max_chars))) if max_chars > 0 else 1
        parts = max(parts_by_time, parts_by_chars)

        # 基于词时间戳拆分
        if seg_words:
            total_dur = duration if duration > 0 else 1e-6
            target_bounds = [start + i * (total_dur / parts) for i in range(1, parts)]
            cut_times: list[float] = []
            for tb in target_bounds:
                # 找到离 tb 最近的词边界
                best_t = None
                best_d = 1e9
                for w in seg_words:
                    for edge in (w.get('start', 0.0), w.get('end', 0.0)):
                        d = abs(edge - tb)
                        if d < best_d:
                            best_d = d
                            best_t = edge
                if best_t is not None:
                    cut_times.append(best_t)
            cut_times = sorted(t for t in cut_times if start < t < end)

            # 按 cut_times 切片
            times = [start] + cut_times + [end]
            # 将词按时间段分桶，并组装文本
            for i in range(len(times) - 1):
                s_i, e_i = times[i], times[i + 1]
                sub_words = [w for w in seg_words if w.get('start', 0.0) >= s_i and w.get('end', 0.0) <= e_i]
                sub_text = " ".join(w.get('word', '').strip() for w in sub_words if w.get('word'))
                if not sub_text:
                    # 回退到原文本的切片估计
                    ratio_a = (s_i - start) / total_dur
                    ratio_b = (e_i - start) / total_dur
                    a = int(ratio_a * len(text))
                    b = int(ratio_b * len(text))
                    sub_text = text[a:b].strip()
                sub_text = wrap_text_for_display(sub_text, preferred_line_length, max_lines)
                new_segments.append({'start': s_i, 'end': e_i, 'segment': sub_text})
            continue

        # 无词级时间戳时：按字符与标点近似拆分
        # 先按字符上限计算断点
        points = split_points_by_chars(text) if too_long_by_chars else []
        # 加入标点断点（句末优先）
        for m in re.finditer(r"[。！？!?.,;；，]", text):
            idx = m.end()
            # 只在过长时考虑
            if len(text) > max_chars or duration > max_duration:
                points.append(idx)
        points = sorted(set(p for p in points if 0 < p < len(text)))

        # 根据 points 切文本，时间均分
        stops = [0] + points + [len(text)]
        times = [start + i * ((end - start) / (len(stops) - 1)) for i in range(len(stops))]
        for i in range(len(stops) - 1):
            a, b = stops[i], stops[i + 1]
            s_i, e_i = times[i], times[i + 1]
            sub_text = text[a:b].strip()
            if not sub_text:
                continue
            sub_text = wrap_text_for_display(sub_text, preferred_line_length, max_lines)
            new_segments.append({'start': s_i, 'end': e_i, 'segment': sub_text})

    # 最终保证按开始时间排序
    new_segments.sort(key=lambda s: (s.get('start', 0.0), s.get('end', 0.0)))
    return new_segments

def configure_decoding_strategy(model):
    """配置 NeMo 模型的解码策略（若支持）。
    - 对 RNNT/Conformer-Transducer 等模型，尝试开启 beam search。
    - 若模型不支持相应属性，静默跳过。
    """
    try:
        if hasattr(model, 'change_decoding_strategy'):
            if DECODING_STRATEGY == 'beam':
                model.change_decoding_strategy(decoding_cfg={
                    'strategy': 'beam',
                    'beam_size': RNNT_BEAM_SIZE,
                })
                print(f"✅ 启用 Beam Search，beam_size={RNNT_BEAM_SIZE}")
            else:
                # 在低显存环境下，禁用 CUDA graphs 降低一次性显存峰值
                model.change_decoding_strategy(decoding_cfg={
                    'strategy': 'greedy',
                    'allow_cuda_graphs': False,
                    'greedy': {
                        'use_cuda_graph_decoder': False,
                        'max_symbols_per_step': 10,
                        'loop_labels': True,
                    }
                })
                print("✅ 启用 Greedy 解码")
        elif hasattr(model, 'decoder') and hasattr(model.decoder, 'cfg'):
            # 兼容部分模型的 decoder 配置
            decoder_cfg = getattr(model.decoder, 'cfg')
            if DECODING_STRATEGY == 'beam' and hasattr(decoder_cfg, 'beam_size'):
                decoder_cfg.beam_size = RNNT_BEAM_SIZE
                print(f"✅ 配置 decoder.beam_size={RNNT_BEAM_SIZE}")
            # 其余情况按默认
    except Exception as e:
        print(f"⚠️ 设置解码策略时出错: {e}")


def safe_transcribe(model, audio_path: str, need_timestamps: bool, batch_size: int, num_workers: int):
    """执行一次安全的转写：
    - 使用 autocast + inference_mode 降低显存
    - 如遇 CUDA OOM，自动降级为 greedy 解码并重试一次
    """
    global DECODING_STRATEGY
    try:
        if cuda_available:
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                return model.transcribe(
                    [audio_path],
                    timestamps=need_timestamps,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
        else:
            with torch.inference_mode():
                return model.transcribe(
                    [audio_path],
                    timestamps=need_timestamps,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e) or 'CUDA error' in str(e):
            print("⚠️ 检测到 CUDA 内存不足，尝试降级为 greedy 解码并重试一次…")
            aggressive_memory_cleanup()
            # 记录原策略并降级
            original_strategy = DECODING_STRATEGY
            try:
                # 强制切换为 greedy
                os.environ['DECODING_STRATEGY'] = 'greedy'
                DECODING_STRATEGY = 'greedy'
                configure_decoding_strategy(model)
                # 重试
                if cuda_available:
                    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                        return model.transcribe(
                            [audio_path],
                            timestamps=need_timestamps,
                            batch_size=1,  # 进一步收缩批量
                            num_workers=0,
                        )
                else:
                    with torch.inference_mode():
                        return model.transcribe(
                            [audio_path],
                            timestamps=need_timestamps,
                            batch_size=1,
                            num_workers=0,
                        )
            finally:
                # 尝试恢复原策略（若需要）
                os.environ['DECODING_STRATEGY'] = original_strategy
                DECODING_STRATEGY = original_strategy
                try:
                    configure_decoding_strategy(model)
                except Exception:
                    pass
        # 非OOM错误原样抛出
        raise
    except ValueError as e:
        # 处理 NeMo TDT Beam 在开启 timestamps 时不支持 alignment preservation 的情况
        if 'Alignment preservation has not been implemented' in str(e):
            print("⚠️ 检测到 TDT Beam 解码不支持对齐保留，自动切换到 greedy 解码并重试一次…")
            aggressive_memory_cleanup()
            original_strategy = DECODING_STRATEGY
            try:
                # 强制切换为 greedy
                os.environ['DECODING_STRATEGY'] = 'greedy'
                globals()['DECODING_STRATEGY'] = 'greedy'
                configure_decoding_strategy(model)
                # 重试（进一步降低批量与并发）
                if cuda_available:
                    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                        return model.transcribe(
                            [audio_path],
                            timestamps=need_timestamps,
                            batch_size=1,
                            num_workers=0,
                        )
                else:
                    with torch.inference_mode():
                        return model.transcribe(
                            [audio_path],
                            timestamps=need_timestamps,
                            batch_size=1,
                            num_workers=0,
                        )
            finally:
                # 尝试恢复原策略
                os.environ['DECODING_STRATEGY'] = original_strategy
                globals()['DECODING_STRATEGY'] = original_strategy
                try:
                    configure_decoding_strategy(model)
                except Exception:
                    pass
        # 其他 ValueError 继续抛出
        raise

# --- Waitress 服务器启动 ---
if __name__ == '__main__':
    
    # 根据是否启用懒加载来决定是预加载模型还是启动后台监控线程
    if ENABLE_LAZY_LOAD:
        print("懒加载模式已启用。模型将在第一次请求时加载。")
        # 启动后台线程来监控和卸载闲置的模型
        if IDLE_TIMEOUT_MINUTES > 0:
            print(f"将启用模型自动卸载功能，闲置超时: {IDLE_TIMEOUT_MINUTES} 分钟。")
            cleanup_thread = threading.Thread(target=model_cleanup_checker, daemon=True)
            cleanup_thread.start()
        else:
            print("模型自动卸载功能已禁用 (IDLE_TIMEOUT_MINUTES=0)。")
        # 启动后台线程在容器启动时预下载模型文件（仅下载到磁盘，不加载到内存）
        print("在后台启动模型预下载线程（仅下载文件，不加载到内存）...")
        try:
            predownload_thread = threading.Thread(target=predownload_model_artifacts, daemon=True)
            predownload_thread.start()
        except Exception as e:
            print(f"启动模型预下载线程失败: {e}")
    else:
        # 懒加载被禁用，在启动时直接加载模型
        print("懒加载模式已禁用，正在启动时预加载模型...")
        try:
            load_model_if_needed()
        except Exception as e:
            print(f"❌ 启动时预加载模型失败: {e}")
            # 预加载失败时退出，以避免运行一个损坏的服务
            exit(1)

    if API_KEY:
        print(f"API Key 认证已启用。请在请求头中提供 'Authorization: Bearer YOUR_API_KEY'")
    else:
        print("API Key 认证已禁用，任何请求都将被接受。")


    # === 简化配置预设推导 ===
    # 计算可用 GPU 显存（或读取用户提供的 GPU_VRAM_GB），结合 PRESET 设置其它参数
    try:
        detected_vram_gb = None
        if check_cuda_compatibility():
            _, _, total_gb = get_gpu_memory_usage()
            detected_vram_gb = round(total_gb)
    except Exception:
        detected_vram_gb = None

    gpu_vram_gb = None
    try:
        gpu_vram_gb = int(GPU_VRAM_GB_ENV) if GPU_VRAM_GB_ENV else detected_vram_gb
    except Exception:
        gpu_vram_gb = detected_vram_gb

    preset = PRESET if PRESET in ['speed', 'balanced', 'quality', 'simple'] else 'balanced'
    if preset == 'simple':
        preset = 'balanced'

    # 基于预设和显存推导参数（仅当用户未显式覆盖时生效）
    def set_if_default(name: str, current, value):
        # 仅当环境变量未显式设置时替换默认
        if os.environ.get(name) is None:
            return value
        return current

    # CHUNK_MINITE
    if gpu_vram_gb is not None:
        if preset == 'speed':
            CHUNK_MINITE = set_if_default('CHUNK_MINITE', CHUNK_MINITE,  min(20, 10 if gpu_vram_gb < 12 else 15))
        elif preset == 'quality':
            CHUNK_MINITE = set_if_default('CHUNK_MINITE', CHUNK_MINITE,  max(6, 8 if gpu_vram_gb >= 8 else 6))
        else:  # balanced
            CHUNK_MINITE = set_if_default('CHUNK_MINITE', CHUNK_MINITE,  8 if gpu_vram_gb < 8 else 10)

    # 并发与显存占比
    if preset == 'speed':
        MAX_CONCURRENT_INFERENCES = set_if_default('MAX_CONCURRENT_INFERENCES', MAX_CONCURRENT_INFERENCES, 2 if (gpu_vram_gb and gpu_vram_gb >= 16) else 1)
        GPU_MEMORY_FRACTION = set_if_default('GPU_MEMORY_FRACTION', GPU_MEMORY_FRACTION, 0.95)
        DECODING_STRATEGY = set_if_default('DECODING_STRATEGY', DECODING_STRATEGY, 'greedy')
    elif preset == 'quality':
        MAX_CONCURRENT_INFERENCES = set_if_default('MAX_CONCURRENT_INFERENCES', MAX_CONCURRENT_INFERENCES, 1)
        GPU_MEMORY_FRACTION = set_if_default('GPU_MEMORY_FRACTION', GPU_MEMORY_FRACTION, 0.90)
        DECODING_STRATEGY = set_if_default('DECODING_STRATEGY', DECODING_STRATEGY, 'beam')
        RNNT_BEAM_SIZE = set_if_default('RNNT_BEAM_SIZE', RNNT_BEAM_SIZE, 4 if (gpu_vram_gb and gpu_vram_gb >= 8) else 2)
    else:  # balanced
        MAX_CONCURRENT_INFERENCES = set_if_default('MAX_CONCURRENT_INFERENCES', MAX_CONCURRENT_INFERENCES, 1)
        GPU_MEMORY_FRACTION = set_if_default('GPU_MEMORY_FRACTION', GPU_MEMORY_FRACTION, 0.92 if (gpu_vram_gb and gpu_vram_gb >= 12) else 0.90)
        DECODING_STRATEGY = set_if_default('DECODING_STRATEGY', DECODING_STRATEGY, 'greedy')

    # 记录最终预设
    print(f"预设: {preset}  | GPU_VRAM_GB: {gpu_vram_gb if gpu_vram_gb is not None else 'unknown'}")
    print(f"推导: CHUNK_MINITE={CHUNK_MINITE}, MAX_CONCURRENT_INFERENCES={MAX_CONCURRENT_INFERENCES}, GPU_MEMORY_FRACTION={GPU_MEMORY_FRACTION}, DECODING_STRATEGY={DECODING_STRATEGY}")

    # 更新并发信号量以匹配推导值
    try:
        new_max_conc = int(MAX_CONCURRENT_INFERENCES) if isinstance(MAX_CONCURRENT_INFERENCES, (int, float, str)) else 1
        if new_max_conc < 1:
            new_max_conc = 1
        globals()['inference_semaphore'] = threading.Semaphore(new_max_conc)
    except Exception as e:
        print(f"⚠️ 初始化并发信号量失败，使用默认1: {e}")
        globals()['inference_semaphore'] = threading.Semaphore(1)

    print(f"🚀 服务器启动中...")
    print(f"API 端点: POST http://{host}:{port}/v1/audio/transcriptions")
    print(f"服务将使用 {threads} 个线程运行。")
    print("")
    print("=== 显存优化配置 ===")
    print(f"激进显存清理: {'启用' if AGGRESSIVE_MEMORY_CLEANUP else '禁用'}")
    print(f"梯度检查点: {'启用' if ENABLE_GRADIENT_CHECKPOINTING else '禁用'}")
    print(f"强制清理阈值: {FORCE_CLEANUP_THRESHOLD*100:.0f}%")
    print(f"最大chunk内存: {MAX_CHUNK_MEMORY_MB}MB")
    print(f"默认chunk时长: {CHUNK_MINITE} 分钟")
    print("=" * 25)
    print("")
    print("=== 闲置资源优化配置 ===")
    print(f"模型闲置超时: {IDLE_TIMEOUT_MINUTES} 分钟")
    print(f"闲置内存清理间隔: {IDLE_MEMORY_CLEANUP_INTERVAL} 秒")
    print(f"深度清理阈值: {IDLE_DEEP_CLEANUP_THRESHOLD} 秒")
    print(f"闲置CPU优化: {'启用' if ENABLE_IDLE_CPU_OPTIMIZATION else '禁用'}")
    print(f"监控间隔: {IDLE_MONITORING_INTERVAL} 秒")
    # 初始化CUDA兼容性检查
    print("正在检查CUDA兼容性...")
    cuda_available = check_cuda_compatibility()
    
    if cuda_available:
        _, _, total_memory = get_gpu_memory_usage()
        print(f"GPU总显存: {total_memory:.1f}GB")
    else:
        memory = psutil.virtual_memory()
        print(f"系统内存: {memory.total/1024**3:.1f}GB")
    print("=" * 25)
    print("")
    print("=== Tensor Core 配置 ===")
    print(f"Tensor Core: {'启用' if ENABLE_TENSOR_CORE else '禁用'}")
    print(f"cuDNN Benchmark: {'启用' if ENABLE_CUDNN_BENCHMARK else '禁用'}")
    print(f"精度模式: {TENSOR_CORE_PRECISION}")
    if cuda_available:
        print(f"GPU支持: {get_tensor_core_info()}")
    else:
        print("GPU支持: N/A - CUDA不可用或不兼容")
    print("=" * 25)
    print("")
    print("=== 句子完整性优化 ===")
    print(f"重叠分割: {'启用' if ENABLE_OVERLAP_CHUNKING else '禁用'}")
    if ENABLE_OVERLAP_CHUNKING:
        print(f"重叠时长: {CHUNK_OVERLAP_SECONDS}s")
        print(f"边界阈值: {SENTENCE_BOUNDARY_THRESHOLD}")
    print("=" * 25)
    serve(app, host=host, port=port, threads=threads)