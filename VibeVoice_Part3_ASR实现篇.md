# VibeVoice深度解析（三）：ASR实现与代码详解

> 本文深入解析VibeVoice的ASR（自动语音识别）实现，包括长音频流式处理、模型推理和结果后处理。

## 目录
- [一、ASR数据流全景](#一asr数据流全景)
- [二、ASR Processor实现](#二asr-processor实现)
- [三、ASR Model实现](#三asr-model实现)
- [四、长音频流式处理](#四长音频流式处理)

---

## 一、ASR数据流全景

### 1.1 完整流程图

```
音频输入
  ↓
┌─────────────────────────────────────────────────────────┐
│ 1. Processor阶段                                         │
├─────────────────────────────────────────────────────────┤
│ 音频文件/数组                                             │
│   ↓                                                      │
│ 加载音频 → 重采样到24kHz → 归一化                         │
│   ↓                                                      │
│ 计算音频时长 → 决定是否使用流式模式                        │
│   ↓                                                      │
│ 构建输入序列：                                            │
│   [系统提示] + [语音tokens] + [用户提示]                  │
│   ↓                                                      │
│ 输出：input_ids, acoustic_input_mask, speech_tensors     │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Model阶段                                             │
├─────────────────────────────────────────────────────────┤
│ 音频编码：                                                │
│   短音频(<60s): 直接编码                                  │
│   长音频(≥60s): 流式分段编码                              │
│   ↓                                                      │
│ Acoustic Encoder → 声学特征                               │
│ Semantic Encoder → 语义特征                               │
│   ↓                                                      │
│ 特征融合：acoustic + semantic                             │
│   ↓                                                      │
│ 替换speech_pad位置的embedding                             │
│   ↓                                                      │
│ Language Model：理解语音内容                              │
│   ↓                                                      │
│ LM Head → 生成JSON格式转录                                │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Post-processing阶段                                   │
├─────────────────────────────────────────────────────────┤
│ 解析JSON输出                                              │
│   ↓                                                      │
│ 提取结构化信息：                                          │
│   - Start time                                           │
│   - End time                                             │
│   - Speaker ID                                           │
│   - Content                                              │
│   ↓                                                      │
│ 返回转录结果列表                                          │
└─────────────────────────────────────────────────────────┘
  ↓
转录结果
```

### 1.2 输入输出示例

**输入音频**: 3秒音频，包含两个说话人

**Processor输出**:
```python
{
    "input_ids": [
        # 系统提示
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, ...,
        # 语音tokens
        151857,  # <|object_ref_start|>
        151859, 151859, ..., 151859,  # <|box_start|> × 23 (3秒音频)
        151858,  # <|object_ref_end|>
        # 用户提示
        2028, 374, 264, 220, 18, 13, 15, 17, 6923, 4275, ...,
        151645, 77091, 198,  # <|im_start|>assistant\n
    ],
    "acoustic_input_mask": [
        False, False, ..., False,  # 系统提示
        False,  # <|object_ref_start|>
        True, True, ..., True,  # 需要替换为语音特征
        False,  # <|object_ref_end|>
        False, False, ..., False,  # 用户提示
    ],
    "speech_tensors": [72000],  # 3秒 × 24000Hz
    "speech_masks": [23],  # ceil(72000 / 3200)
}
```

**模型输出**:
```json
[
    {
        "Start time": "0.00",
        "End time": "1.50",
        "Speaker ID": "0",
        "Content": "Hello, how are you?"
    },
    {
        "Start time": "1.50",
        "End time": "3.00",
        "Speaker ID": "1",
        "Content": "I'm fine, thank you!"
    }
]
```

---

## 二、ASR Processor实现

### 2.1 VibeVoiceASRProcessor核心代码

#### __call__方法：主入口

```python
def __call__(
    self,
    audio: Optional[Union[str, np.ndarray, torch.Tensor, List]] = None,
    sampling_rate: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    padding: bool = True,
    max_length: Optional[int] = None,
    truncation: bool = False,
    add_generation_prompt: bool = True,
    use_streaming: bool = True,
    context_info: Optional[str] = None,
    **kwargs
) -> BatchEncoding:
    """
    处理音频输入用于ASR
    
    Args:
        audio: 音频输入（文件路径、数组或张量）
        use_streaming: 是否使用流式模式（<60s自动禁用）
        context_info: 上下文信息（热词、元数据等）
    """
    if audio is None:
        raise ValueError("Audio input is required for ASR processing")
    
    # 处理单个/批量输入
    if isinstance(audio, list):
        is_batched = True
        audio_list = audio
    else:
        is_batched = False
        audio_list = [audio]
    
    # 处理每个音频
    all_encodings = []
    for audio_input in audio_list:
        encoding = self._process_single_audio(
            audio_input,
            sampling_rate=sampling_rate,
            add_generation_prompt=add_generation_prompt,
            use_streaming=use_streaming,
            context_info=context_info,
        )
        all_encodings.append(encoding)
    
    # 批量编码
    batch_encoding = self._batch_encode(
        all_encodings,
        padding=padding,
        max_length=max_length,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    
    return batch_encoding
```

#### _process_single_audio：处理单个音频

```python
def _process_single_audio(
    self,
    audio: Union[str, np.ndarray, torch.Tensor],
    sampling_rate: Optional[int] = None,
    add_generation_prompt: bool = True,
    use_streaming: bool = True,
    context_info: Optional[str] = None,
) -> Dict[str, Any]:
    """处理单个音频输入"""
    
    # 1. 加载音频
    if isinstance(audio, str):
        # 从文件加载（使用FFmpeg）
        audio_array, file_sr = load_audio_use_ffmpeg(audio, resample=False)
        
        # 重采样到24kHz
        if file_sr != self.target_sample_rate:
            import librosa
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=file_sr, 
                target_sr=self.target_sample_rate
            )
    elif isinstance(audio, torch.Tensor):
        audio_array = audio.cpu().numpy()
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()
    else:
        audio_array = np.array(audio, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()
    
    # 确保float32
    audio_array = audio_array.astype(np.float32)
    
    # 2. 归一化
    if self.normalize_audio and self.audio_normalizer:
        audio_array = self.audio_normalizer(audio_array)
    
    # 3. 计算音频时长
    audio_duration = len(audio_array) / self.target_sample_rate
    
    # 4. 自动禁用流式模式（短音频）
    if use_streaming and audio_duration < 60.0:
        use_streaming = False
    
    # 5. 计算VAE token长度
    vae_tok_len = math.ceil(len(audio_array) / self.speech_tok_compress_ratio)
    
    # 6. 构建token序列
    # 系统提示
    system_prompt_text = self.tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}],
        tokenize=False
    )
    system_tokens = self.tokenizer.encode(system_prompt_text)
    
    # 语音tokens
    sp_start_token = self.tokenizer.convert_ids_to_tokens(self.speech_start_id)
    sp_pad_token = self.tokenizer.convert_ids_to_tokens(self.speech_pad_id)
    sp_end_token = self.tokenizer.convert_ids_to_tokens(self.speech_end_id)
    
    # 用户提示
    show_keys = ['Start time', 'End time', 'Speaker ID', 'Content']
    if context_info and context_info.strip():
        user_suffix = (
            f"This is a {audio_duration:.2f} seconds audio, "
            f"with extra info: {context_info.strip()}\n\n"
            f"Please transcribe it with these keys: " + ", ".join(show_keys)
        )
    else:
        user_suffix = (
            f"This is a {audio_duration:.2f} seconds audio, "
            f"please transcribe it with these keys: " + ", ".join(show_keys)
        )
    
    user_input_string = ''.join(
        [sp_start_token] + [sp_pad_token] * vae_tok_len + [sp_end_token]
    ) + '\n' + user_suffix
    
    user_tokens = self.tokenizer.apply_chat_template(
        [{"role": "user", "content": user_input_string}],
        tokenize=True
    )
    
    # 组合tokens
    full_tokens = system_tokens + user_tokens
    
    # 7. 创建acoustic_input_mask
    acoustic_input_mask = [
        1 if token == self.speech_pad_id else 0 
        for token in full_tokens
    ]
    
    return {
        "input_ids": full_tokens,
        "acoustic_input_mask": acoustic_input_mask,
        "speech": audio_array,
        "vae_tok_len": vae_tok_len,
    }
```

#### post_process_transcription：后处理

```python
def post_process_transcription(self, text: str) -> List[Dict[str, Any]]:
    """
    后处理转录文本，提取结构化数据
    
    Args:
        text: 模型生成的文本（JSON格式）
    
    Returns:
        转录片段列表
    """
    try:
        # 提取JSON
        if "```json" in text:
            json_start = text.find("```json") + 7
            json_end = text.find("```", json_start)
            json_str = text[json_start:json_end].strip()
        else:
            # 查找JSON数组或对象
            json_start = text.find("[")
            if json_start == -1:
                json_start = text.find("{")
            if json_start != -1:
                # 找到匹配的闭括号
                bracket_count = 0
                json_end = json_start
                for i in range(json_start, len(text)):
                    if text[i] in "[{":
                        bracket_count += 1
                    elif text[i] in "]}":
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_end = i + 1
                            break
                json_str = text[json_start:json_end]
            else:
                json_str = text
        
        # 解析JSON
        result = json.loads(json_str)
        
        # 确保是列表
        if isinstance(result, dict):
            result = [result]
        
        # 清理和标准化
        cleaned_result = []
        for item in result:
            if isinstance(item, dict):
                cleaned_item = {}
                # 键映射
                key_mapping = {
                    "Start time": "start_time",
                    "Start": "start_time",
                    "End time": "end_time",
                    "End": "end_time",
                    "Speaker ID": "speaker_id",
                    "Speaker": "speaker_id",
                    "Content": "text",
                }
                for key, mapped_key in key_mapping.items():
                    if key in item:
                        cleaned_item[mapped_key] = item[key]
                
                if cleaned_item:
                    cleaned_result.append(cleaned_item)
        
        return cleaned_result
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return []
```

---

## 三、ASR Model实现

### 3.1 VibeVoiceASRForConditionalGeneration

#### encode_speech：编码语音

```python
def encode_speech(
    self,
    speech_tensors: torch.FloatTensor,
    speech_masks: Optional[torch.BoolTensor] = None,
    speech_semantic_tensors: Optional[torch.FloatTensor] = None,
    streaming_segment_duration: float = 60.0,
):
    """
    编码语音输入，支持长音频流式处理
    
    Args:
        speech_tensors: 音频数组 [batch_size, samples]
        streaming_segment_duration: 流式分段时长（秒）
    
    Returns:
        组合的语音特征
    """
    # 确保正确的dtype
    speech_tensors = speech_tensors.to(self.config.torch_dtype)
    
    # 确保形状 (batch, samples)
    if speech_tensors.ndim == 1:
        speech_tensors = speech_tensors.unsqueeze(0)
    
    batch_size, total_samples = speech_tensors.shape
    sample_rate = 24000
    
    # 计算分段大小
    segment_samples = int(streaming_segment_duration * sample_rate)
    
    # 决定是否使用流式
    use_streaming = total_samples > segment_samples
    
    with torch.no_grad():
        if not use_streaming:
            # 短音频：直接处理
            encoder_output = self.model.acoustic_tokenizer.encode(
                speech_tensors.unsqueeze(1)
            )
            audio_tokens = encoder_output.sample(
                dist_type=self.model.acoustic_tokenizer.std_dist_type
            )[0]
            acoustic_features = self.model.acoustic_connector(audio_tokens)
            
            # 编码语义特征
            if speech_semantic_tensors is not None:
                semantic_features = self.model.semantic_connector(
                    speech_semantic_tensors
                )
            else:
                semantic_tokens = self.model.semantic_tokenizer.encode(
                    speech_tensors.unsqueeze(1)
                ).mean
                semantic_features = self.model.semantic_connector(semantic_tokens)
        else:
            # 长音频：流式处理
            acoustic_encoder_cache = VibeVoiceTokenizerStreamingCache()
            semantic_encoder_cache = VibeVoiceTokenizerStreamingCache()
            acoustic_mean_segments = []
            semantic_mean_segments = []
            sample_indices = torch.arange(batch_size, device=speech_tensors.device)
            
            # 分段处理
            segments = list(self._iter_segments(total_samples, segment_samples))
            num_segments = len(segments)
            
            for seg_idx, (start, end) in enumerate(segments):
                chunk = speech_tensors[:, start:end].contiguous()
                if chunk.numel() == 0:
                    continue
                
                is_final = (seg_idx == num_segments - 1)
                
                # 编码声学特征
                acoustic_encoder_output = self.model.acoustic_tokenizer.encode(
                    chunk.unsqueeze(1),
                    cache=acoustic_encoder_cache,
                    sample_indices=sample_indices,
                    use_cache=True,
                    is_final_chunk=is_final,
                )
                acoustic_mean_segments.append(acoustic_encoder_output.mean)
                
                # 编码语义特征
                semantic_encoder_output = self.model.semantic_tokenizer.encode(
                    chunk.unsqueeze(1),
                    cache=semantic_encoder_cache,
                    sample_indices=sample_indices,
                    use_cache=True,
                    is_final_chunk=is_final,
                )
                semantic_mean_segments.append(semantic_encoder_output.mean)
            
            # 拼接所有段
            acoustic_mean_full = torch.cat(acoustic_mean_segments, dim=1).contiguous()
            acoustic_encoder_output = VibeVoiceTokenizerEncoderOutput(
                mean=acoustic_mean_full,
                std=self.model.acoustic_tokenizer.fix_std
            )
            audio_tokens = acoustic_encoder_output.sample(
                dist_type=self.model.acoustic_tokenizer.std_dist_type
            )[0]
            acoustic_features = self.model.acoustic_connector(audio_tokens)
            
            semantic_tokens = torch.cat(semantic_mean_segments, dim=1).contiguous()
            semantic_features = self.model.semantic_connector(semantic_tokens)
        
        # 融合特征
        if speech_masks is not None:
            combined_features = acoustic_features[speech_masks] + \
                              semantic_features[speech_masks]
        else:
            combined_features = acoustic_features + semantic_features
    
    return combined_features

def _iter_segments(self, total_length: int, segment_length: int):
    """迭代音频分段"""
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    for start in range(0, total_length, segment_length):
        end = min(start + segment_length, total_length)
        if end > start:
            yield start, end
```

#### forward：前向传播

```python
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    speech_tensors: Optional[torch.FloatTensor] = None,
    speech_masks: Optional[torch.BoolTensor] = None,
    speech_semantic_tensors: Optional[torch.FloatTensor] = None,
    acoustic_input_mask: Optional[torch.BoolTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, CausalLMOutput]:
    """前向传播"""
    
    # 1. 获取文本嵌入
    if inputs_embeds is None and input_ids is not None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
    
    # 2. 编码语音并替换
    if speech_tensors is not None and acoustic_input_mask is not None:
        speech_features = self.encode_speech(
            speech_tensors=speech_tensors,
            speech_masks=speech_masks,
            speech_semantic_tensors=speech_semantic_tensors,
        )
        # 克隆避免in-place操作
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[acoustic_input_mask] = speech_features
    
    # 3. 通过语言模型
    outputs = self.model(
        input_ids=None,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=kwargs.get('use_cache', False),
        return_dict=True,
    )
    
    hidden_states = outputs.last_hidden_state
    logits = self.lm_head(hidden_states)
    
    # 4. 计算损失（训练时）
    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1).to(shift_logits.device)
        )
    
    return VibeVoiceCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
    )
```

---

## 四、长音频流式处理

### 4.1 为什么需要流式处理？

**问题**：
- 卷积层的中间激活值大小与输入长度成正比
- 长音频（>10分钟）会导致激活值超过2^32，引发溢出

**解决方案**：
- 将长音频分段处理
- 使用缓存保持上下文连续性
- 最后拼接所有段的输出

### 4.2 流式处理流程

```
长音频 [1, 14400000]  # 10分钟 × 24000Hz
  ↓
分段 (每段60秒)
  ├─ 段1 [1, 1440000]  # 0-60s
  ├─ 段2 [1, 1440000]  # 60-120s
  ├─ ...
  └─ 段10 [1, 1440000]  # 540-600s
  ↓
逐段编码（使用缓存）
  ├─ 段1 → mean1 [1, 450, 64]
  ├─ 段2 → mean2 [1, 450, 64]
  ├─ ...
  └─ 段10 → mean10 [1, 450, 64]
  ↓
拼接
  mean_full = cat([mean1, ..., mean10], dim=1)  # [1, 4500, 64]
  ↓
统一采样
  audio_tokens = sample(mean_full)
  ↓
连接器
  features = connector(audio_tokens)
```

### 4.3 缓存机制

**VibeVoiceTokenizerStreamingCache**：

```python
class VibeVoiceTokenizerStreamingCache:
    def __init__(self):
        self.cache = {}  # {(layer_id, sample_idx): state_tensor}
    
    def get(self, layer_id, sample_indices):
        """获取缓存状态"""
        states = []
        for idx in sample_indices.tolist():
            key = (layer_id, idx)
            if key not in self.cache:
                return None
            states.append(self.cache[key])
        return torch.stack(states, dim=0)
    
    def set(self, layer_id, sample_indices, states):
        """设置缓存状态"""
        for i, idx in enumerate(sample_indices.tolist()):
            key = (layer_id, idx)
            self.cache[key] = states[i].detach()
```

**SConv1d流式卷积**：

```python
def _forward_streaming(self, x, cache, sample_indices, is_final_chunk):
    """流式前向传播"""
    # 1. 获取缓存
    cached_states = cache.get(self.layer_id, sample_indices)
    if cached_states is None:
        cached_states = torch.zeros(B, C, self.context_size, ...)
    
    # 2. 拼接缓存和输入
    input_with_context = torch.cat([cached_states, x], dim=2)
    
    # 3. 最后一块添加额外padding
    if is_final_chunk:
        extra_padding = get_extra_padding_for_conv1d(...)
        if extra_padding > 0:
            input_with_context = pad1d(input_with_context, (0, extra_padding))
    
    # 4. 卷积
    output = self.conv(input_with_context)
    
    # 5. 更新缓存
    new_cache = input_with_context[:, :, -self.context_size:]
    cache.set(self.layer_id, sample_indices, new_cache)
    
    return output
```

### 4.4 性能对比

| 音频时长 | 非流式内存 | 流式内存 | 加速比 |
|---------|-----------|---------|--------|
| 1分钟 | 2GB | 2GB | 1.0x |
| 5分钟 | 10GB | 2GB | 5.0x |
| 10分钟 | 溢出 | 2GB | ∞ |
| 30分钟 | 溢出 | 2GB | ∞ |

---

## 五、实战示例

### 5.1 基础ASR

```python
from vibevoice import (
    VibeVoiceASRForConditionalGeneration,
    VibeVoiceASRProcessor
)

# 加载模型
model = VibeVoiceASRForConditionalGeneration.from_pretrained(
    "vibevoice-asr-1.5b"
)
processor = VibeVoiceASRProcessor.from_pretrained(
    "vibevoice-asr-1.5b"
)

# 处理音频
inputs = processor(
    audio="meeting.wav",
    return_tensors="pt"
)

# 生成转录
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,
)

# 解码
text = processor.decode(outputs[0], skip_special_tokens=True)

# 后处理
transcription = processor.post_process_transcription(text)

# 打印结果
for segment in transcription:
    print(f"[{segment['start_time']}-{segment['end_time']}] "
          f"Speaker {segment['speaker_id']}: {segment['text']}")
```

### 5.2 长音频ASR

```python
# 处理长音频（自动启用流式）
inputs = processor(
    audio="long_meeting.wav",  # 30分钟音频
    return_tensors="pt",
    use_streaming=True,  # 自动检测
)

# 生成（流式处理在encode_speech中完成）
outputs = model.generate(**inputs)
transcription = processor.post_process_transcription(
    processor.decode(outputs[0])
)
```

### 5.3 带上下文信息的ASR

```python
# 提供热词和元数据
inputs = processor(
    audio="technical_talk.wav",
    context_info="Keywords: AI, machine learning, neural network",
    return_tensors="pt"
)

outputs = model.generate(**inputs)
transcription = processor.post_process_transcription(
    processor.decode(outputs[0])
)
```

---

## 小结

本文深入解析了VibeVoice ASR的实现：

1. ✅ ASR完整数据流：从音频到转录
2. ✅ Processor实现：音频处理、提示构建
3. ✅ Model实现：语音编码、特征融合
4. ✅ 流式处理：长音频分段、缓存机制

**核心要点**：

- 🎯 双编码器架构：声学+语义
- 🎯 流式处理：支持任意长度音频
- 🎯 JSON输出：结构化转录结果
- 🎯 上下文支持：热词、元数据增强

**与TTS的对比**：

| 特性 | TTS | ASR |
|------|-----|-----|
| 输入 | 文本+语音样本 | 音频 |
| 输出 | 音频波形 | JSON文本 |
| 编码器 | 声学编解码器 | 声学+语义编码器 |
| 解码器 | 扩散模型 | 语言模型 |
| 流式 | 文本窗口 | 音频分段 |

下一篇将讲解流式TTS的实现细节！

---

如果觉得本文有帮助，欢迎点赞、收藏、关注！