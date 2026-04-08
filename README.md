# 高精度日语转写系统（Windows / faster-whisper）

本项目是“工程强化版”的日语实时转写系统：

- 目标不是原始 ASR 流，而是**接近字幕可读文本**
- 重点场景：**直播 / 培训讲话 / 背景音乐存在**
- 支持：**初稿实时显示 + 延迟精修覆盖**

---

## 1. 模块结构

- `audio_capture.py`：WASAPI loopback 系统音频采集（Windows）
- `audio_preprocess.py`：音频预处理链路（单声道、16k、滤波、降噪、归一化）
- `segmenter.py`：混合分段与滑窗工具（chunk + overlap）
- `transcriber.py`：双阶段识别、上下文注入、GPU 回退、文本更新事件
- `postprocess.py`：去重复、清理、标点、可读性断句
- `hotwords.py`：领域热词词典加载与替换
- `correction.py`：二次文本校正层（合并断裂句、移除坏片段）
- `config.py`：参数中心 + 预设（直播 / 培训 / 背景音乐 / 高精度）
- `gui.py`：GUI（初稿区 + 修正版区）
- `main.py`：主编排（线程、队列、写文件、异常管理）
- `start_windows.bat`：Windows 启动脚本

---

## 2. 关键能力

### 双阶段识别（核心）

1. `realtime_mode`：低延迟初稿（快速参数）
2. `quality_mode`：对最近 10~20 秒上下文做高精度重识别（强参数）
3. GUI 展示“初稿 -> 修正版覆盖”

### 滑窗 + 重叠

- 识别时使用 `chunk_length_sec + overlap_seconds`
- 降低长句断裂、重复、漏词

### 上下文注入

- 将前几段修正版作为 `initial_prompt` 的上下文
- 提升长句连贯性和固定短语稳定性

### 音频预处理强化

- 转单声道
- 重采样 16kHz
- RMS 归一化
- 高通 + 低通 + 带通
- 轻量降噪 + 背景音乐抑制

### 日语后处理 + 热词词典

- 去口吃式重复
- 去重复短语
- 标点补全与可读断句
- 领域错词替换（如 報連相 / 外勤先 / 査定 / 評価制度）

---

## 3. 参数与预设

所有关键参数集中在 `config.py`：

- `beam_size`
- `best_of`
- `temperature`
- `vad_filter`
- `chunk_length_sec`
- `overlap_seconds`
- `no_speech_threshold`
- `log_prob_threshold`

推荐预设：

- 直播：`PRESETS["直播场景"]`
- 培训讲话：`PRESETS["培训讲话"]`
- 背景音乐：`PRESETS["背景音乐场景"]`（默认）
- 高精度：`PRESETS["高精度模式"]`

---

## 4. GPU / CPU 说明

- 默认先尝试：`device="cuda"`, `compute_type="float16"`
- 若 CUDA 初始化失败：自动回退 CPU
- 若运行时 GPU 报错（cublas/cudnn/cuda）：当前音频立即 CPU 重试，不丢数据

---

## 5. 安装与运行

```bash
pip install -r requirements.txt
python main.py
```

Windows 也可双击：

```bash
start_windows.bat
```

---

## 6. 热词词典

GUI 可指定热词词典路径（json 或 txt）：

### JSON 示例

```json
{
  "法連想": "報連相",
  "外見先": "外勤先",
  "さて": "査定"
}
```

### TXT 示例

```text
法連想 => 報連相
外見先 => 外勤先
さて => 査定
```

---

## 7. 如何测试“精度提升”

建议用同一段“有背景音乐 + 连续讲话”音频，比较：

1. 旧版本输出
2. 新版本初稿输出
3. 新版本修正版输出（最终 txt）

重点观察：

- 固定词汇（報連相 / 外勤先 / 査定 / 評価制度）命中率
- 重复短语数量
- 长句断裂/漏词情况
- 可读性（是否接近字幕句子）

---

## 8. 下一步可继续提升

1. 引入专业降噪/分离模型（Demucs / RNNoise / DNS）
2. 引入语言模型重评分（N-best rerank）
3. 强化领域语料微调（术语词库 + prompt模板）
4. 增加真正的时间轴对齐与高质量 SRT/VTT 导出
5. 加入自动评估（CER/WER + 术语命中率）


## 9. data discontinuity 修复说明

- 录音线程与识别线程已完全解耦：`capture -> RingBuffer -> segmenter -> transcriber`。
- 采集线程写入 `RingBuffer` 时永不阻塞，识别卡顿不会打断采集。
- 当下游短时拥塞时，环形缓存会覆盖最旧帧并记录 dropped 计数，优先保证“实时连续采集”。
- 分段线程向识别队列采用非阻塞写入，避免反向卡住采集链路。
