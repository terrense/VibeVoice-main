# VibeVoice 架构分析文档

## 目录结构

```
vibevoice/
├── __init__.py                 # 主入口，导出核心类
├── configs/                    # 配置文件
│   ├── qwen2.5_1.5b_64k.json
│   └── qwen2.5_7b_32k.json
├── modular/                    # 核心模型模块
│   ├── __init__.py
│   ├── configuration_vibevoice.py              # TTS配置类
│   ├── configuration_vibevoice_streaming.py    # 流式TTS配置类
│   ├── modeling_vibevoice.py                   # TTS主模型
│   ├── modeling_vibevoice_asr.py               # ASR模型
│   ├── modeling_vibevoice_streaming.py         # 流式TTS模型
│   ├── modeling_vibevoice_streaming_inference.py # 流式推理
│   ├── modular_vibevoice_diffusion_head.py     # 扩散头
│   ├── modular_vibevoice_text_tokenizer.py     # 文本分词器
│   ├── modular_vibevoice_tokenizer.py          # 语音编解码器
│   └── streamer.py                             # 音频流处理
├── processor/                  # 数据处理器
│   ├── __init__.py
│   ├── audio_utils.py                          # 音频工具函数
│   ├── vibevoice_asr_processor.py              # ASR处理器
│   ├── vibevoice_processor.py                  # TTS处理器
│   ├── vibevoice_streaming_processor.py        # 流式处理器
│   └── vibevoice_tokenizer_processor.py        # 音频特征提取
├── schedule/                   # 扩散调度器
│   ├── __init__.py
│   ├── dpm_solver.py                           # DPM求解器
│   └── timestep_sampler.py                     # 时间步采样器
└── scripts/                    # 工具脚本
    ├── __init__.py
    └── convert_nnscaler_checkpoint_to_transformers.py
```

## 一、配置类 (Configuration Classes)

### 1. VibeVoiceAcousticTokenizerConfig
**文件**: `vibevoice/modular/configuration_vibevoice.py`

**用途**: 声学编解码器配置

**主要参数**:
- `channels`: 音频通道数 (默认: 1)
- `vae_dim`: VAE潜在空间维度 (默认: 64)
- `causal`: 是否使用因果卷积 (默认: True)
- `encoder_ratios`: 编码器下采样比率 (默认: [8,5,5,4,2,2])
- `decoder_ratios`: 解码器上采样比率
- `encoder_depths`: 编码器深度配置 (默认: "3-3-3-3-3-3-8")

### 2. VibeVoiceSemanticTokenizerConfig
**文件**: `vibevoice/modular/configuration_vibevoice.py`

**用途**: 语义编码器配置

**主要参数**:
- `vae_dim`: VAE潜在空间维度 (默认: 64)
- `fix_std`: 固定标准差 (默认: 0)
- `std_dist_type`: 标准差分布类型 (默认: 'none')

### 3. VibeVoiceDiffusionHeadConfig
**文件**: `vibevoice/modular/configuration_vibevoice.py`

**用途**: 扩散头配置

**主要参数**:
- `hidden_size`: 隐藏层大小 (默认: 768)
- `head_layers`: 扩散头层数 (默认: 4)
- `prediction_type`: 预测类型 (默认: "v_prediction")
- `ddpm_num_steps`: DDPM训练步数 (默认: 1000)
- `ddpm_num_inference_steps`: 推理步数 (默认: 20)
- `ddpm_beta_schedule`: Beta调度策略 (默认: "cosine")

### 4. VibeVoiceConfig
**文件**: `vibevoice/modular/configuration_vibevoice.py`

**用途**: TTS主配置类，组合所有子配置

**子配置**:
- `acoustic_tokenizer_config`: 声学编解码器配置
- `semantic_tokenizer_config`: 语义编码器配置
- `decoder_config`: 语言模型配置 (Qwen2Config)
- `diffusion_head_config`: 扩散头配置

### 5. VibeVoiceASRConfig
**文件**: `vibevoice/modular/configuration_vibevoice.py`

**用途**: ASR模型配置

**子配置**:
- `acoustic_tokenizer_config`
- `semantic_tokenizer_config`
- `decoder_config`

### 6. VibeVoiceStreamingConfig
**文件**: `vibevoice/modular/configuration_vibevoice_streaming.py`

**用途**: 流式TTS配置

**特殊参数**:
- `tts_backbone_num_hidden_layers`: TTS专用的上层Transformer层数 (默认: 20)

## 二、核心模型类 (Core Model Classes)

### 1. TTS模型

#### VibeVoiceModel
**文件**: `vibevoice/modular/modeling_vibevoice.py`

**用途**: TTS基础模型

**主要组件**:
- `language_model`: Qwen2语言模型
- `acoustic_tokenizer`: 声学编解码器
- `semantic_tokenizer`: 语义编码器
- `acoustic_connector`: 声学特征连接器
- `semantic_connector`: 语义特征连接器
- `prediction_head`: 扩散预测头
- `noise_scheduler`: DPM噪声调度器

**主要方法**:
- `forward()`: 前向传播
- `get_input_embeddings()`: 获取输入嵌入
- `set_speech_tokenizers()`: 设置语音编解码器

#### VibeVoiceForConditionalGeneration
**文件**: `vibevoice/modular/modeling_vibevoice.py`

**用途**: TTS条件生成模型（带LM头）

**继承**: `VibeVoicePreTrainedModel`

**主要组件**:
- `model`: VibeVoiceModel实例
- `lm_head`: 语言模型输出头

**主要方法**:
- `forward()`: 前向传播，计算语言模型损失和扩散损失
- `forward_speech_features()`: 处理语音特征
- `tie_weights()`: 绑定输入输出嵌入权重

### 2. ASR模型

#### VibeVoiceASRModel
**文件**: `vibevoice/modular/modeling_vibevoice_asr.py`

**用途**: ASR基础模型

**主要组件**:
- `language_model`: Qwen2语言模型
- `acoustic_tokenizer`: 声学编码器
- `semantic_tokenizer`: 语义编码器
- `acoustic_connector`: 声学特征连接器
- `semantic_connector`: 语义特征连接器

#### VibeVoiceASRForConditionalGeneration
**文件**: `vibevoice/modular/modeling_vibevoice_asr.py`

**用途**: ASR条件生成模型

**继承**: `VibeVoiceASRPreTrainedModel`, `GenerationMixin`

**主要方法**:
- `encode_speech()`: 编码语音输入（支持流式处理长音频）
- `forward()`: 前向传播
- `prepare_inputs_for_generation()`: 准备生成输入

**特性**:
- 支持长音频流式处理（>60秒自动启用）
- 使用缓存避免卷积溢出（>2^32）

### 3. 流式TTS模型

#### VibeVoiceStreamingModel
**文件**: `vibevoice/modular/modeling_vibevoice_streaming.py`

**用途**: 流式TTS基础模型

**架构特点**:
- 分层设计：下层Transformer仅编码文本，上层Transformer编码文本并生成语音
- `language_model`: 下层文本编码器
- `tts_language_model`: 上层TTS专用层
- `tts_input_types`: 标记TTS文本的嵌入

**主要组件**:
- `acoustic_tokenizer`: 声学编解码器
- `acoustic_connector`: 声学特征连接器
- `prediction_head`: 扩散预测头
- `noise_scheduler`: DPM噪声调度器

**注意**: `forward()` 方法被禁用，需要分别调用 `language_model` 和 `tts_language_model`

#### VibeVoiceStreamingForConditionalGenerationInference
**文件**: `vibevoice/modular/modeling_vibevoice_streaming_inference.py`

**用途**: 流式TTS推理模型

**继承**: `VibeVoiceStreamingPreTrainedModel`, `GenerationMixin`

**主要组件**:
- `model`: VibeVoiceStreamingModel实例
- `tts_eos_classifier`: TTS结束分类器

**主要方法**:
- `forward_lm()`: 基础文本LM前向传播
- `forward_tts_lm()`: TTS LM前向传播
- `generate()`: 流式生成（文本窗口+语音扩散采样）
- `sample_speech_tokens()`: 采样语音token

**生成流程**:
1. 文本分窗口预填充
2. 增量LM + TTS LM更新
3. 交错语音token扩散采样
4. 支持实时音频流输出

## 三、编解码器类 (Tokenizer Classes)

### 1. 语音编解码器

#### VibeVoiceAcousticTokenizerModel
**文件**: `vibevoice/modular/modular_vibevoice_tokenizer.py`

**用途**: 声学编解码器（VAE架构）

**主要组件**:
- `encoder`: TokenizerEncoder - 音频→潜在表示
- `decoder`: TokenizerDecoder - 潜在表示→音频
- `fix_std`: 固定标准差
- `std_dist_type`: 分布类型 ('gaussian', 'laplace', 'cauchy', 'none')

**主要方法**:
- `encode()`: 编码音频，返回 `VibeVoiceTokenizerEncoderOutput`
- `decode()`: 解码潜在表示为音频
- 支持流式处理（使用 `VibeVoiceTokenizerStreamingCache`）

#### VibeVoiceSemanticTokenizerModel
**文件**: `vibevoice/modular/modular_vibevoice_tokenizer.py`

**用途**: 语义编码器（仅编码器）

**主要组件**:
- `encoder`: TokenizerEncoder

**主要方法**:
- `encode()`: 编码音频为语义表示

### 2. 文本分词器

#### VibeVoiceTextTokenizer
**文件**: `vibevoice/modular/modular_vibevoice_text_tokenizer.py`

**基类**: `Qwen2Tokenizer`

**特殊Token**:
- `<|vision_start|>`: 语音开始
- `<|vision_end|>`: 语音结束
- `<|vision_pad|>`: 语音扩散pad

**属性**:
- `speech_start_id`
- `speech_end_id`
- `speech_diffusion_id`
- `pad_id`

#### VibeVoiceTextTokenizerFast
**文件**: `vibevoice/modular/modular_vibevoice_text_tokenizer.py`

**基类**: `Qwen2TokenizerFast`

**用途**: 快速版本的文本分词器

#### VibeVoiceASRTextTokenizerFast
**文件**: `vibevoice/modular/modular_vibevoice_text_tokenizer.py`

**基类**: `Qwen2TokenizerFast`

**用途**: ASR专用文本分词器

**特殊Token**:
- `<|object_ref_start|>`: 语音开始
- `<|object_ref_end|>`: 语音结束
- `<|box_start|>`: 语音pad

**属性**:
- `speech_start_id`
- `speech_end_id`
- `speech_pad_id`

### 3. 编解码器内部组件

#### TokenizerEncoder
**文件**: `vibevoice/modular/modular_vibevoice_tokenizer.py`

**用途**: 编码器组件

**架构**:
- 初始卷积层
- 多个下采样Block1D层
- 最终卷积层

**主要方法**:
- `forward()`: 编码音频
- 支持流式处理

#### TokenizerDecoder
**文件**: `vibevoice/modular/modular_vibevoice_tokenizer.py`

**用途**: 解码器组件

**架构**:
- 初始卷积层
- 多个上采样Block1D层
- 最终卷积层

**主要方法**:
- `forward()`: 解码潜在表示
- 支持流式处理

## 四、处理器类 (Processor Classes)

### 1. VibeVoiceProcessor
**文件**: `vibevoice/processor/vibevoice_processor.py`

**用途**: TTS数据处理器

**主要组件**:
- `tokenizer`: 文本分词器
- `audio_processor`: 音频处理器
- `audio_normalizer`: 音频归一化器

**主要方法**:
- `__call__()`: 处理文本和语音样本
- `_process_single()`: 处理单个脚本
- `_create_voice_prompt()`: 创建语音提示
- `prepare_speech_inputs()`: 准备语音输入
- `_parse_script()`: 解析脚本（Speaker X: text格式）
- `save_audio()`: 保存音频

**输入格式**:
- 文本: 字符串或文件路径（.json/.txt）
- 语音样本: 音频文件路径或numpy数组列表

**输出**:
- `input_ids`: Token ID序列
- `attention_mask`: 注意力掩码
- `speech_tensors`: 语音特征
- `speech_masks`: 语音掩码
- `speech_input_mask`: 语音token位置掩码

### 2. VibeVoiceASRProcessor
**文件**: `vibevoice/processor/vibevoice_asr_processor.py`

**用途**: ASR数据处理器

**主要方法**:
- `__call__()`: 处理音频输入
- `_process_single_audio()`: 处理单个音频
- `post_process_transcription()`: 后处理转录文本（解析JSON）

**特性**:
- 自动检测音频时长
- <60秒音频自动禁用流式模式
- 支持上下文信息（热词、元数据）

**输入格式**:
- 音频: 文件路径、numpy数组或torch张量
- 采样率: 24000 Hz

**输出**:
- `input_ids`: Token ID序列
- `attention_mask`: 注意力掩码
- `acoustic_input_mask`: 声学token位置掩码
- `speech_tensors`: 音频特征
- `speech_masks`: 语音掩码

### 3. VibeVoiceStreamingProcessor
**文件**: `vibevoice/processor/vibevoice_streaming_processor.py`

**用途**: 流式TTS数据处理器

**主要方法**:
- `process_input_with_cached_prompt()`: 基于缓存提示处理输入
- `prepare_speech_inputs()`: 准备语音输入

**特性**:
- 仅支持单样本处理
- 使用缓存的KV cache

### 4. VibeVoiceTokenizerProcessor
**文件**: `vibevoice/processor/vibevoice_tokenizer_processor.py`

**基类**: `FeatureExtractionMixin`

**用途**: 音频特征提取器

**主要方法**:
- `__call__()`: 处理音频
- `_ensure_mono()`: 转换为单声道
- `_process_single_audio()`: 处理单个音频
- `_load_audio_from_path()`: 从文件加载音频
- `save_audio()`: 保存音频

**支持格式**:
- 音频: .wav, .mp3, .flac, .m4a, .ogg
- 张量: .pt, .npy

## 五、扩散相关类 (Diffusion Classes)

### 1. VibeVoiceDiffusionHead
**文件**: `vibevoice/modular/modular_vibevoice_diffusion_head.py`

**基类**: `PreTrainedModel`

**用途**: 扩散预测头

**主要组件**:
- `noisy_images_proj`: 噪声图像投影
- `cond_proj`: 条件投影
- `t_embedder`: 时间步嵌入器
- `layers`: HeadLayer列表
- `final_layer`: 最终输出层

**主要方法**:
- `forward()`: 预测噪声/速度
- `initialize_weights()`: 初始化权重

### 2. DPMSolverMultistepScheduler
**文件**: `vibevoice/schedule/dpm_solver.py`

**基类**: `SchedulerMixin`, `ConfigMixin`

**用途**: DPM多步求解器

**主要方法**:
- `set_timesteps()`: 设置时间步
- `step()`: 执行一步去噪
- `add_noise()`: 添加噪声
- `convert_model_output()`: 转换模型输出

**支持的调度策略**:
- `linear`
- `scaled_linear`
- `cosine` (squaredcos_cap_v2)
- `cauchy`
- `laplace`

### 3. 时间步采样器

#### UniformSampler
**文件**: `vibevoice/schedule/timestep_sampler.py`

**用途**: 均匀采样时间步

#### LogitNormalSampler
**文件**: `vibevoice/schedule/timestep_sampler.py`

**用途**: Logit正态分布采样时间步

## 六、辅助类 (Utility Classes)

### 1. 流式缓存

#### VibeVoiceTokenizerStreamingCache
**文件**: `vibevoice/modular/modular_vibevoice_tokenizer.py`

**用途**: 流式卷积缓存（类似注意力KV缓存）

**主要方法**:
- `get()`: 获取缓存状态
- `set()`: 设置缓存状态
- `clear()`: 清除缓存
- `set_to_zero()`: 将缓存置零

### 2. 音频流处理

#### AudioStreamer
**文件**: `vibevoice/modular/streamer.py`

**基类**: `BaseStreamer`

**用途**: 音频流处理器（同步版本）

**主要方法**:
- `put()`: 放入音频块
- `end()`: 结束生成
- `get_stream()`: 获取单个样本流
- `__iter__()`: 批量迭代器

#### AsyncAudioStreamer
**文件**: `vibevoice/modular/streamer.py`

**基类**: `AudioStreamer`

**用途**: 异步音频流处理器

**主要方法**:
- `put()`: 异步放入音频块
- `end()`: 异步结束生成
- `get_stream()`: 异步获取流
- `__aiter__()`: 异步迭代器

### 3. 音频工具

#### AudioNormalizer
**文件**: `vibevoice/processor/audio_utils.py`

**用途**: 音频归一化

**主要方法**:
- `tailor_dB_FS()`: 调整到目标dB FS
- `avoid_clipping()`: 避免削波
- `__call__()`: 归一化音频

**函数**:
- `load_audio_use_ffmpeg()`: 使用FFmpeg加载音频
- `load_audio_bytes_use_ffmpeg()`: 从字节加载音频

### 4. 卷积层

#### SConv1d
**文件**: `vibevoice/modular/modular_vibevoice_tokenizer.py`

**用途**: 支持流式的1D卷积层

**特性**:
- 支持因果/非因果卷积
- 支持流式处理（使用缓存）
- 自动处理padding

**主要方法**:
- `forward()`: 前向传播
- `_forward_streaming()`: 流式前向传播
- `_forward_non_streaming()`: 非流式前向传播

#### SConvTranspose1d
**文件**: `vibevoice/modular/modular_vibevoice_tokenizer.py`

**用途**: 支持流式的1D转置卷积层

**特性**:
- 支持流式处理
- 自动处理padding和unpadding

### 5. 其他组件

#### Block1D
**文件**: `vibevoice/modular/modular_vibevoice_tokenizer.py`

**用途**: 1D卷积块

**组件**:
- `norm`: 归一化层（LayerNorm或RMSNorm）
- `mixer`: 混合层（卷积或深度卷积）
- `ffn`: 前馈网络
- `gamma`: 层缩放参数

#### SpeechConnector
**文件**: `vibevoice/modular/modeling_vibevoice.py`

**用途**: 语音特征连接器

**架构**:
- 线性层 → RMSNorm → 线性层

#### BinaryClassifier
**文件**: `vibevoice/modular/modeling_vibevoice_streaming.py`

**用途**: 二分类器（用于TTS EOS检测）

**架构**:
- 线性层 → ReLU → 线性层

## 七、数据类 (Data Classes)

### 1. VibeVoiceTokenizerEncoderOutput
**文件**: `vibevoice/modular/modular_vibevoice_tokenizer.py`

**用途**: 编码器输出（高斯分布）

**字段**:
- `mean`: 均值
- `std`: 标准差

**方法**:
- `sample()`: 从分布采样

### 2. VibeVoiceCausalLMOutputWithPast
**文件**: `vibevoice/modular/modeling_vibevoice.py`

**用途**: TTS模型输出

**字段**:
- `loss`: 语言模型损失
- `diffusion_loss`: 扩散损失
- `speech_token_num`: 语音token数量
- `logits`: 输出logits
- `past_key_values`: KV缓存
- `hidden_states`: 隐藏状态
- `attentions`: 注意力权重

### 3. VibeVoiceGenerationOutput
**文件**: `vibevoice/modular/modeling_vibevoice.py`

**用途**: 生成输出

**字段**:
- `sequences`: 生成的token序列
- `speech_outputs`: 生成的语音波形列表

## 八、工具脚本 (Utility Scripts)

### convert_nnscaler_checkpoint_to_transformers.py
**文件**: `vibevoice/scripts/convert_nnscaler_checkpoint_to_transformers.py`

**用途**: 转换nnscaler检查点为HuggingFace格式

**主要函数**:
- `convert_vibevoice_nnscaler_checkpoint_to_hf()`: 执行转换

**功能**:
- 加载nnscaler检查点
- 提取模型状态字典
- 创建HuggingFace模型
- 保存为分片格式（.safetensors）
- 保存处理器配置

## 总结

VibeVoice是一个复杂的多模态语音处理系统，主要包含：

1. **TTS系统**: 文本→语音生成
   - 标准TTS: `VibeVoiceForConditionalGeneration`
   - 流式TTS: `VibeVoiceStreamingForConditionalGenerationInference`

2. **ASR系统**: 语音→文本识别
   - `VibeVoiceASRForConditionalGeneration`

3. **核心技术**:
   - VAE编解码器（声学+语义）
   - 扩散模型（DDPM）
   - Transformer语言模型（Qwen2）
   - 流式处理支持

4. **特色功能**:
   - 多说话人TTS
   - 长音频流式处理
   - 实时音频流输出
   - 灵活的配置系统
