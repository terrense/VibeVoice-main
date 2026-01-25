# VibeVoice 模块详细说明

## 目录
1. [Processor模块](#processor模块)
2. [Modular模块](#modular模块)
3. [Schedule模块](#schedule模块)
4. [Scripts模块](#scripts模块)

---

## 一、Processor模块

### 模块概述
Processor模块负责数据预处理和后处理，是模型输入输出的桥梁。

### 1. vibevoice_processor.py - TTS处理器

**主类**: `VibeVoiceProcessor`

**职责**:
- 解析多说话人脚本
- 处理语音样本
- 创建语音提示
- 生成模型输入

**脚本格式支持**:
```
Speaker 0: Hello, this is speaker zero.
Speaker 1: Hi, I'm speaker one.
```

**文件格式支持**:
- JSON: `[{"speaker": "0", "text": "..."}, ...]`
- TXT: 纯文本或格式化脚本

**关键流程**:
1. 解析脚本 → 提取说话人和文本
2. 编码语音样本 → 创建语音提示
3. 构建输入序列 → 系统提示 + 语音输入 + 文本输入 + 语音输出标记
4. Padding和mask生成

**输出结构**:
```python
{
    "input_ids": [batch_size, seq_len],
    "attention_mask": [batch_size, seq_len],
    "speech_tensors": [num_samples, max_length],
    "speech_masks": [num_samples, max_vae_len],
    "speech_input_mask": [batch_size, seq_len],  # 标记语音token位置
}
```

### 2. vibevoice_asr_processor.py - ASR处理器

**主类**: `VibeVoiceASRProcessor`

**职责**:
- 音频预处理
- 构建ASR输入格式
- 后处理转录结果

**输入格式**:
```python
# 系统提示
"You are a helpful assistant that transcribes audio input into text output in JSON format."

# 用户输入
<|speech_start|><|speech_pad|>...<|speech_end|>
This is a X.XX seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content
```

**输出格式**:
```json
[
    {
        "Start time": "0.00",
        "End time": "2.50",
        "Speaker ID": "0",
        "Content": "Hello world"
    }
]
```

**流式处理逻辑**:
- 音频时长 < 60秒: 直接处理
- 音频时长 ≥ 60秒: 自动启用流式模式

### 3. vibevoice_streaming_processor.py - 流式TTS处理器

**主类**: `VibeVoiceStreamingProcessor`

**特点**:
- 基于缓存的提示处理
- 仅支持单样本
- 不实现标准`__call__`方法

**核心方法**: `process_input_with_cached_prompt()`

**使用场景**: 实时TTS生成，文本逐步输入

### 4. vibevoice_tokenizer_processor.py - 音频特征提取器

**主类**: `VibeVoiceTokenizerProcessor`

**基类**: `FeatureExtractionMixin`

**职责**:
- 音频格式转换
- 立体声→单声道
- 音频归一化
- 文件加载

**支持格式**:
- 音频: .wav, .mp3, .flac, .m4a, .ogg
- 张量: .pt, .npy

**归一化流程**:
1. 调整到目标dB FS (-25 dB)
2. 避免削波

### 5. audio_utils.py - 音频工具

**主要功能**:
- FFmpeg音频加载
- 音频归一化
- 并发控制

**关键类**: `AudioNormalizer`

**FFmpeg并发控制**:
```python
# 环境变量控制最大并发数
VIBEVOICE_FFMPEG_MAX_CONCURRENCY=4
```

**用途**: 防止vLLM多请求并发时FFmpeg进程过多导致超时

---

## 二、Modular模块

### 模块概述
Modular模块包含核心模型实现，是VibeVoice的心脏。

### 1. configuration_vibevoice.py - 配置类

**配置层次**:
```
VibeVoiceConfig (TTS)
├── acoustic_tokenizer_config
├── semantic_tokenizer_config
├── decoder_config (Qwen2Config)
└── diffusion_head_config

VibeVoiceASRConfig (ASR)
├── acoustic_tokenizer_config
├── semantic_tokenizer_config
└── decoder_config

VibeVoiceStreamingConfig (流式TTS)
├── acoustic_tokenizer_config
├── decoder_config
└── diffusion_head_config
```

**关键参数**:
- `acoustic_vae_dim`: 声学VAE维度 (64)
- `semantic_vae_dim`: 语义VAE维度 (128)
- `tts_backbone_num_hidden_layers`: TTS专用层数 (20)

### 2. modeling_vibevoice.py - TTS主模型

**架构图**:
```
输入文本 → Tokenizer → Embeddings
                              ↓
语音样本 → Acoustic Encoder → Acoustic Connector ──┐
                                                   ├→ 拼接 → Language Model → Hidden States
语义样本 → Semantic Encoder → Semantic Connector ──┘                              ↓
                                                                          ┌────────┴────────┐
                                                                          ↓                 ↓
                                                                      LM Head         Diffusion Head
                                                                          ↓                 ↓
                                                                      Text Loss      Diffusion Loss
```

**训练损失**:
1. 语言模型损失: 文本生成
2. 扩散损失: 语音质量

**关键组件**:
- `SpeechConnector`: 线性层 → RMSNorm → 线性层
- `VibeVoiceDiffusionHead`: 扩散预测头
- `DPMSolverMultistepScheduler`: 噪声调度器

### 3. modeling_vibevoice_asr.py - ASR模型

**架构图**:
```
音频输入 → Acoustic Encoder → Acoustic Connector ──┐
                                                   ├→ 拼接 → 替换speech_pad位置
音频输入 → Semantic Encoder → Semantic Connector ──┘              ↓
                                                          Text Embeddings
                                                                  ↓
                                                          Language Model
                                                                  ↓
                                                              LM Head
                                                                  ↓
                                                          转录文本 (JSON)
```

**流式处理**:
- 分段大小: 60秒
- 缓存机制: `VibeVoiceTokenizerStreamingCache`
- 最终拼接: 所有段的mean拼接后统一采样

### 4. modeling_vibevoice_streaming.py - 流式TTS模型

**分层架构**:
```
                    ┌─────────────────────────────────┐
                    │   Input Text Tokens             │
                    └─────────────────────────────────┘
                                  ↓
                    ┌─────────────────────────────────┐
                    │   Language Model (下层)         │
                    │   - 仅编码文本                   │
                    │   - 不生成语音                   │
                    └─────────────────────────────────┘
                                  ↓
                          Hidden States
                                  ↓
                    ┌─────────────────────────────────┐
                    │   TTS Language Model (上层)     │
                    │   - 编码文本                     │
                    │   - 生成语音                     │
                    │   + TTS Input Types Embedding   │
                    └─────────────────────────────────┘
                                  ↓
                    ┌─────────────────────────────────┐
                    │   TTS EOS Classifier            │
                    │   (判断是否结束生成)             │
                    └─────────────────────────────────┘
```

**关键特性**:
- 禁用统一`forward()`方法
- 必须分别调用`language_model`和`tts_language_model`
- 支持文本窗口流式输入

### 5. modeling_vibevoice_streaming_inference.py - 流式推理

**生成流程**:
```
1. 预填充阶段
   ├─ 文本窗口1 (5 tokens) → LM → TTS LM
   ├─ 文本窗口2 (5 tokens) → LM → TTS LM
   └─ ...

2. 生成阶段 (每个文本窗口后)
   ├─ 语音token 1 → Diffusion → Audio Chunk 1
   ├─ 语音token 2 → Diffusion → Audio Chunk 2
   ├─ ...
   └─ 语音token 6 → Diffusion → Audio Chunk 6

3. 循环直到
   ├─ EOS检测到
   ├─ 达到最大长度
   └─ 外部停止信号
```

**窗口大小**:
- 文本窗口: 5 tokens (`TTS_TEXT_WINDOW_SIZE`)
- 语音窗口: 6 tokens (`TTS_SPEECH_WINDOW_SIZE`)

**CFG支持**:
- 正向条件: TTS LM输出
- 负向条件: 负提示LM输出
- CFG scale: 可调节

### 6. modular_vibevoice_tokenizer.py - 语音编解码器

**VAE架构**:
```
Encoder:
音频 [B, 1, T] → Conv1d → Block1D × N → Conv1d → Mean [B, D, T']
                                                 → Std (固定或学习)

Decoder:
Latent [B, D, T'] → Conv1d → Block1D × N → Conv1d → 音频 [B, 1, T]
```

**Block1D结构**:
```
输入 → Norm → Mixer (Conv/DepthwiseConv) → LayerScale → 残差
    ↓
    → Norm → FFN → LayerScale → 残差 → 输出
```

**流式支持**:
- `SConv1d`: 流式卷积层
- `SConvTranspose1d`: 流式转置卷积层
- 缓存机制: 保持`context_size`个样本

**采样策略**:
- `gaussian`: 高斯采样
- `laplace`: 拉普拉斯采样
- `cauchy`: 柯西采样
- `none`: 不采样（直接使用mean）

### 7. modular_vibevoice_diffusion_head.py - 扩散头

**架构**:
```
Noisy Latent [B, T, D] → Linear → Hidden [B, T, H]
                                      ↓
Timestep → TimestepEmbedder → t_emb [B, H]
                                      ↓
Condition [B, T, H] → Linear → c_emb [B, T, H]
                                      ↓
                            c = c_emb + t_emb
                                      ↓
                            HeadLayer × N
                                      ↓
                            FinalLayer
                                      ↓
                        Predicted Noise/Velocity
```

**HeadLayer**:
```
输入 → AdaLN (shift, scale, gate) → FFN → gate * output → 残差
```

**初始化策略**:
- AdaLN modulation: 零初始化
- Final layer: 零初始化
- 其他: 正态分布初始化

### 8. modular_vibevoice_text_tokenizer.py - 文本分词器

**继承关系**:
```
Qwen2Tokenizer
    ↓
VibeVoiceTextTokenizer (TTS)
    - 添加语音特殊token
    - <|vision_start|>, <|vision_end|>, <|vision_pad|>

Qwen2TokenizerFast
    ↓
VibeVoiceTextTokenizerFast (TTS快速版)
    - 同上

Qwen2TokenizerFast
    ↓
VibeVoiceASRTextTokenizerFast (ASR)
    - 添加ASR特殊token
    - <|object_ref_start|>, <|object_ref_end|>, <|box_start|>
    - 设置chat_template
```

### 9. streamer.py - 音频流处理

**AudioStreamer架构**:
```
生成线程                     消费线程
    ↓                           ↓
put(audio_chunks) → Queue[0] → get_stream(0) → 音频块1
                  → Queue[1] → get_stream(1) → 音频块2
                  → Queue[N] → get_stream(N) → 音频块N
    ↓
end() → 发送停止信号
```

**AsyncAudioStreamer**:
- 使用`asyncio.Queue`
- 支持异步迭代
- 适用于异步应用

---

## 三、Schedule模块

### 1. dpm_solver.py - DPM求解器

**DPMSolverMultistepScheduler**

**支持的算法**:
- `dpmsolver++`: 推荐用于引导采样
- `sde-dpmsolver++`: 推荐用于无条件采样

**Beta调度策略**:
- `linear`: 线性调度
- `scaled_linear`: 缩放线性
- `cosine`: 余弦调度
- `cauchy`: 柯西调度
- `laplace`: 拉普拉斯调度

**关键方法**:
- `set_timesteps()`: 设置推理时间步
- `step()`: 执行一步去噪
- `add_noise()`: 添加噪声（训练用）
- `convert_model_output()`: 转换模型输出

**预测类型**:
- `epsilon`: 预测噪声
- `sample`: 预测样本
- `v_prediction`: 预测速度

### 2. timestep_sampler.py - 时间步采样器

**UniformSampler**:
- 均匀采样时间步
- 用于标准训练

**LogitNormalSampler**:
- Logit正态分布采样
- 用于重要性采样训练

---

## 四、Scripts模块

### convert_nnscaler_checkpoint_to_transformers.py

**功能**: 转换nnscaler训练检查点为HuggingFace格式

**转换流程**:
1. 加载nnscaler检查点
2. 提取模型状态字典
3. 处理权重绑定
4. 创建HF模型
5. 加载权重
6. 保存为分片格式

**输出文件**:
- `config.json`: 模型配置
- `preprocessor_config.json`: 处理器配置
- `model-*.safetensors`: 模型权重分片

**使用示例**:
```bash
python convert_nnscaler_checkpoint_to_transformers.py \
    --nnscaler_checkpoint_path checkpoint.pt \
    --pytorch_dump_folder_path output_dir \
    --config_path config.json
```

---

## 总结

VibeVoice的模块设计遵循以下原则:

1. **分离关注点**: Processor处理数据，Modular实现模型，Schedule管理扩散
2. **可扩展性**: 配置类支持灵活组合
3. **流式优先**: 核心组件都支持流式处理
4. **HuggingFace兼容**: 遵循HF标准接口

每个模块都有明确的职责，通过清晰的接口协作，构成完整的语音处理系统。
