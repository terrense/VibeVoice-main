# VibeVoice深度解析（二）：TTS实现与代码详解

> 本文深入解析VibeVoice的TTS（文本转语音）实现，从数据处理到模型推理，带你理解每一行关键代码。

## 目录
- [一、TTS数据流全景](#一tts数据流全景)
- [二、Processor：数据预处理](#二processor数据预处理)
- [三、Model：前向传播](#三model前向传播)
- [四、Generation：推理生成](#四generation推理生成)

---

## 一、TTS数据流全景

### 1.1 完整流程图

```
用户输入
  ↓
┌─────────────────────────────────────────────────────────┐
│ 1. Processor阶段                                         │
├─────────────────────────────────────────────────────────┤
│ 文本脚本 + 语音样本                                       │
│   ↓                                                      │
│ 解析脚本 → 提取说话人                                     │
│   ↓                                                      │
│ 编码语音样本 → 创建语音提示                               │
│   ↓                                                      │
│ 构建输入序列：                                            │
│   [系统提示] + [语音输入] + [文本输入] + [语音输出标记]    │
│   ↓                                                      │
│ 输出：input_ids, speech_tensors, masks                   │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Model阶段                                             │
├─────────────────────────────────────────────────────────┤
│ 嵌入层：token → embeddings                               │
│   ↓                                                      │
│ 语音编码：audio → latent → connector → embeddings        │
│   ↓                                                      │
│ 替换：将语音token位置的embedding替换为语音特征            │
│   ↓                                                      │
│ Language Model：生成hidden states                        │
│   ↓                                                      │
│ 分支1：LM Head → 文本logits → 语言模型损失               │
│ 分支2：Diffusion Head → 预测噪声 → 扩散损失              │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Generation阶段（推理）                                 │
├─────────────────────────────────────────────────────────┤
│ 自回归生成语音token                                       │
│   ↓                                                      │
│ 每个token通过扩散模型采样潜在表示                          │
│   ↓                                                      │
│ 解码器：latent → audio                                   │
│   ↓                                                      │
│ 拼接所有音频块 → 完整语音                                 │
└─────────────────────────────────────────────────────────┘
  ↓
输出音频
```

### 1.2 数据格式示例

**输入脚本**：

```
Speaker 0: Hello, welcome to VibeVoice demo.
Speaker 1: This is an amazing text-to-speech system.
Speaker 0: Let's hear how it sounds!
```

**处理后的token序列**：

```
[系统提示tokens]
<|vision_start|> [语音样本0的VAE tokens] <|vision_end|>
<|vision_start|> [语音样本1的VAE tokens] <|vision_end|>
[文本输入标记]
Speaker 0: Hello, welcome to VibeVoice demo.
Speaker 1: This is an amazing text-to-speech system.
Speaker 0: Let's hear how it sounds!
[语音输出标记]
<|vision_start|> [待生成的语音tokens] <|vision_end|>
```

---

## 二、Processor：数据预处理

### 2.1 VibeVoiceProcessor核心代码

#### 初始化

```python
class VibeVoiceProcessor:
    def __init__(self, tokenizer=None, audio_processor=None, 
                 speech_tok_compress_ratio=3200, db_normalize=True):
        self.tokenizer = tokenizer  # 文本分词器
        self.audio_processor = audio_processor  # 音频处理器
        self.speech_tok_compress_ratio = 3200  # 压缩比
        self.db_normalize = db_normalize  # 是否归一化
        
        # 音频归一化器
        if db_normalize:
            self.audio_normalizer = AudioNormalizer()
        
        # 系统提示
        self.system_prompt = (
            " Transform the text provided by various speakers "
            "into speech output, utilizing the distinct voice "
            "of each respective speaker.\n"
        )
```

#### __call__方法：主入口

```python
def __call__(
    self,
    text: Optional[Union[str, List[str]]] = None,
    voice_samples: Optional[List[Union[str, np.ndarray]]] = None,
    padding: bool = True,
    return_tensors: Optional[str] = None,
    **kwargs,
) -> BatchEncoding:
    """
    处理文本和语音样本
    
    Args:
        text: 脚本文本或文件路径
        voice_samples: 语音样本列表（每个说话人一个）
        padding: 是否padding
        return_tensors: 返回格式（'pt'表示PyTorch张量）
    
    Returns:
        BatchEncoding包含：
        - input_ids: token序列
        - attention_mask: 注意力掩码
        - speech_tensors: 语音特征
        - speech_masks: 语音掩码
        - speech_input_mask: 标记语音token位置
    """
    # 1. 处理单个/批量输入
    if isinstance(text, str):
        texts = [text]
        is_batched = False
    else:
        texts = text
        is_batched = True
    
    # 2. 处理语音样本
    if voice_samples is not None:
        if not is_batched:
            voice_samples_list = [voice_samples]
        else:
            voice_samples_list = voice_samples
    else:
        voice_samples_list = [None] * len(texts)
    
    # 3. 处理每个输入
    all_encodings = []
    for text_input, voice_input in zip(texts, voice_samples_list):
        encoding = self._process_single(text_input, voice_input)
        all_encodings.append(encoding)
    
    # 4. 批量编码
    batch_encoding = self._batch_encode(
        all_encodings,
        padding=padding,
        return_tensors=return_tensors,
    )
    
    return batch_encoding
```

#### _process_single：处理单个样本

```python
def _process_single(
    self,
    text: Union[str, TextInput],
    voice_samples: Optional[List[Union[str, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """处理单个脚本"""
    
    # 1. 加载脚本（支持文件路径或直接文本）
    if isinstance(text, str):
        if text.endswith('.json'):
            script = self._convert_json_to_script(text)
        elif text.endswith('.txt'):
            script = self._convert_text_to_script(text)
        else:
            script = text
    
    # 2. 解析脚本，提取说话人和文本
    parsed_lines = self._parse_script(script)
    # 结果：[(speaker_id, text), ...]
    # 例如：[(0, " Hello"), (1, " Hi"), (0, " Bye")]
    
    all_speakers = list(set(speaker_id for speaker_id, _ in parsed_lines))
    
    # 3. 创建系统提示
    system_tokens = self.tokenizer.encode(self.system_prompt)
    
    # 4. 处理语音样本（如果提供）
    if voice_samples:
        voice_tokens, voice_speech_inputs, voice_speech_masks = \
            self._create_voice_prompt(voice_samples[:len(all_speakers)])
    else:
        voice_tokens, voice_speech_inputs, voice_speech_masks = [], [], []
    
    # 5. 构建完整token序列
    full_tokens = system_tokens + voice_tokens
    speech_input_mask = [False] * len(system_tokens) + voice_speech_masks
    
    # 6. 添加文本输入部分
    full_tokens += self.tokenizer.encode(' Text input:\n', add_special_tokens=False)
    speech_input_mask += [False] * len(self.tokenizer.encode(' Text input:\n', add_special_tokens=False))
    
    for speaker_id, speaker_text in parsed_lines:
        speaker_text_tokens = self.tokenizer.encode(
            f" Speaker {speaker_id}:{speaker_text}\n", 
            add_special_tokens=False
        )
        full_tokens += speaker_text_tokens
        speech_input_mask += [False] * len(speaker_text_tokens)
    
    # 7. 添加语音输出标记
    full_tokens += self.tokenizer.encode(' Speech output:\n', add_special_tokens=False)
    full_tokens += [self.tokenizer.speech_start_id]
    speech_input_mask += [False] * (len(self.tokenizer.encode(' Speech output:\n', add_special_tokens=False)) + 1)
    
    return {
        "input_ids": full_tokens,
        "speech_inputs": voice_speech_inputs if voice_speech_inputs else None,
        "speech_input_mask": speech_input_mask,
        "parsed_script": parsed_lines,
        "all_speakers": all_speakers,
    }
```

#### _create_voice_prompt：创建语音提示

```python
def _create_voice_prompt(
    self, 
    speaker_samples: List[Union[str, np.ndarray]]
) -> Tuple[List[int], List[np.ndarray], List[bool]]:
    """
    创建语音提示tokens和处理音频样本
    
    Returns:
        voice_tokens: 语音提示的token序列
        voice_speech_inputs: 音频数组列表
        voice_speech_masks: 标记哪些token是语音
    """
    vae_token_id = self.tokenizer.speech_diffusion_id  # <|vision_pad|>
    
    voice_full_tokens = self.tokenizer.encode(' Voice input:\n', add_special_tokens=False)
    voice_speech_inputs = []
    voice_speech_masks = [False] * len(voice_full_tokens)
    
    for speaker_id, speaker_audio in enumerate(speaker_samples):
        # 1. 编码说话人前缀
        prefix_tokens = self.tokenizer.encode(
            f" Speaker {speaker_id}:", 
            add_special_tokens=False
        )
        
        # 2. 加载音频
        if isinstance(speaker_audio, str):
            wav = self.audio_processor._load_audio_from_path(speaker_audio)
        else:
            wav = np.array(speaker_audio, dtype=np.float32)
        
        # 3. 归一化音频
        if self.db_normalize and self.audio_normalizer:
            wav = self.audio_normalizer(wav)
        
        # 4. 计算VAE token长度
        # 音频长度 / 压缩比 = VAE token数量
        vae_tok_len = math.ceil(wav.shape[0] / self.speech_tok_compress_ratio)
        
        # 5. 构建token序列
        # 格式：Speaker X: <|speech_start|> <|speech_pad|> × N <|speech_end|>
        speaker_tokens = (
            prefix_tokens + 
            [self.tokenizer.speech_start_id] + 
            [vae_token_id] * vae_tok_len +  # 占位符
            [self.tokenizer.speech_end_id] + 
            self.tokenizer.encode('\n', add_special_tokens=False)
        )
        
        # 6. 创建mask（标记哪些是语音token）
        vae_input_mask = (
            [False] * len(prefix_tokens) + 
            [False] +  # speech_start
            [True] * vae_tok_len +  # 这些位置需要替换为语音特征
            [False] +  # speech_end
            [False]  # \n
        )
        
        voice_full_tokens.extend(speaker_tokens)
        voice_speech_masks.extend(vae_input_mask)
        voice_speech_inputs.append(wav)
    
    return voice_full_tokens, voice_speech_inputs, voice_speech_masks
```

### 2.2 关键点解析

**为什么需要speech_input_mask？**

```python
# speech_input_mask标记哪些token位置需要替换为语音特征
# 例如：
input_ids = [101, 102, 103, 104, 105, 106, 107]
speech_input_mask = [False, False, True, True, True, False, False]

# 在模型中：
embeddings = self.embed_tokens(input_ids)  # 获取文本嵌入
speech_features = self.encode_speech(audio)  # 编码语音

# 替换语音token位置
embeddings[speech_input_mask] = speech_features
```

**压缩比3200的含义**：

```python
# 假设音频采样率24000Hz，时长3秒
audio_samples = 24000 * 3 = 72000

# VAE token数量
vae_tokens = ceil(72000 / 3200) = 23

# 每个VAE token代表的音频时长
time_per_token = 3200 / 24000 = 0.133秒 ≈ 133ms
```

---

## 三、Model：前向传播

### 3.1 VibeVoiceForConditionalGeneration

#### forward方法：完整前向传播

```python
def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    speech_tensors: Optional[torch.FloatTensor] = None,
    speech_masks: Optional[torch.BoolTensor] = None,
    acoustic_input_mask: Optional[torch.BoolTensor] = None,
    acoustic_loss_mask: Optional[torch.BoolTensor] = None,
    speeches_loss_input: Optional[torch.FloatTensor] = None,
    speech_semantic_tensors: Optional[torch.FloatTensor] = None,
    ddpm_batch_mul: int = 1,
    **kwargs,
) -> VibeVoiceCausalLMOutputWithPast:
    """
    前向传播
    
    Args:
        input_ids: token序列 [B, L]
        speech_tensors: 音频数组 [N, T_audio]
        speech_masks: 语音掩码 [N, T_vae]
        acoustic_input_mask: 标记输入中的语音token位置 [B, L]
        acoustic_loss_mask: 标记需要计算扩散损失的位置 [B, L]
        speeches_loss_input: 标记哪些语音需要计算损失 [N, T_vae]
    """
    
    # 1. 获取文本嵌入
    x = self.get_input_embeddings()(input_ids)  # [B, L, H]
    
    # 2. 编码语义特征
    semantic_speech_all_connect_features = \
        self.model.semantic_connector(speech_semantic_tensors)
    
    # 3. 编码声学特征
    if speeches_loss_input is not None:
        # 训练模式：部分音频需要计算扩散损失
        speech_all_features, speech_all_connect_features = \
            self.forward_speech_features(
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                speech_type="audio",
                return_unmask=True
            )
        
        # 替换输入中的语音token
        if speech_tensors is not None:
            x[acoustic_input_mask] = (
                speech_all_connect_features[speech_masks] +
                semantic_speech_all_connect_features[speech_masks]
            )
        
        # 选择需要计算损失的语音片段
        target_latent_mask = speeches_loss_input & speech_masks
        speech_features = speech_all_features[target_latent_mask]
        speech_connect_features = speech_all_connect_features[target_latent_mask]
    else:
        # 推理模式：所有音频都用于输入
        speech_features, speech_connect_features = \
            self.forward_speech_features(
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                speech_type="audio",
            )
        
        if speech_tensors is not None:
            x[acoustic_input_mask] = speech_connect_features
    
    # 4. 通过语言模型
    outputs = self.model(
        input_ids=None,
        attention_mask=attention_mask,
        inputs_embeds=x,
        use_cache=False,
        return_dict=True,
    )
    
    hidden_states = outputs.last_hidden_state  # [B, L, H]
    
    # 5. 计算语言模型logits
    logits = self.lm_head(hidden_states)  # [B, L, V]
    
    # 6. 计算扩散损失
    diffusion_loss = None
    if speech_tensors is not None and acoustic_loss_mask.sum().item() > 0:
        # 提取需要计算扩散损失的hidden states
        condition_features = hidden_states[acoustic_loss_mask]  # [M, H]
        
        speech_len, latent_size = speech_features.shape
        
        # 生成噪声
        noise = torch.randn(
            (speech_len * ddpm_batch_mul, latent_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        
        # 随机采样时间步
        timesteps = torch.multinomial(
            torch.ones(self.config.diffusion_head_config.ddpm_num_steps),
            speech_len * ddpm_batch_mul,
            replacement=True,
        ).to(hidden_states.device)
        
        # 重复数据以增加batch size
        speech_features_repeated = speech_features.repeat_interleave(
            ddpm_batch_mul, dim=0
        )
        condition_features_repeated = condition_features.repeat_interleave(
            ddpm_batch_mul, dim=0
        )
        
        # 添加噪声
        noisy_speech_features = self.model.noise_scheduler.add_noise(
            speech_features_repeated, noise, timesteps
        )
        
        # 预测噪声/速度
        model_output = self.model.prediction_head(
            noisy_speech_features, 
            timesteps, 
            condition_features_repeated
        )
        
        # 计算目标
        if self.config.diffusion_head_config.prediction_type == "v_prediction":
            target_for_loss = self.model.noise_scheduler.get_velocity(
                speech_features_repeated, noise, timesteps
            )
        else:  # epsilon
            target_for_loss = noise
        
        # 计算MSE损失
        diffusion_loss = F.mse_loss(
            model_output.float(), 
            target_for_loss.float(), 
            reduction='sum'
        )
        diffusion_loss = diffusion_loss / latent_size / ddpm_batch_mul
    
    return VibeVoiceCausalLMOutputWithPast(
        loss=None,  # 在训练脚本中计算
        diffusion_loss=diffusion_loss,
        speech_token_num=speech_len if speech_tensors is not None else 0,
        logits=logits,
        past_key_values=outputs.past_key_values,
    )
```

#### forward_speech_features：编码语音

```python
def forward_speech_features(
    self, 
    speech_tensors=None, 
    speech_masks=None, 
    speech_type="audio", 
    return_unmask=False
):
    """
    编码语音特征
    
    Args:
        speech_tensors: 音频数组 [N, T_audio]
        speech_masks: 掩码 [N, T_vae]
        speech_type: "audio"或"vae"
        return_unmask: 是否返回未mask的特征
    
    Returns:
        audio_features: VAE潜在表示 [N, T_vae, 64]
        connect_features: 连接后的特征 [N, T_vae, H]
    """
    if speech_tensors is None:
        # 返回零特征
        vae_dim = self.config.acoustic_tokenizer_config.vae_dim
        audio_features = torch.zeros(1, 1, vae_dim).to(
            self.get_input_embeddings().weight
        )
        connect_features = self.model.acoustic_connector(audio_features)
        return audio_features, connect_features
    
    with torch.no_grad():
        if speech_type == "audio":
            # 编码音频
            frames = self.model.acoustic_tokenizer.encode(
                speech_tensors.unsqueeze(1)
            )[0][0]
            # 从分布采样
            audio_tokens = frames.sample(
                self.model.acoustic_tokenizer.std_dist_type
            )[0]
        
        elif speech_type == "vae":
            # 直接使用VAE特征
            vae_dim = self.config.acoustic_tokenizer_config.vae_dim
            speech_mode = speech_tensors.reshape(
                speech_tensors.size(0), -1, vae_dim
            )
            # 添加高斯噪声
            std = torch.randn(...) * self.model.acoustic_tokenizer.fix_std
            audio_tokens = speech_mode + std * torch.randn(speech_mode.shape)
        
        # 计算缩放因子（首次调用时）
        if torch.isnan(self.model.speech_scaling_factor):
            scaling_factor = 1. / audio_tokens[speech_masks].flatten().std()
            bias_factor = -audio_tokens[speech_masks].flatten().mean()
            
            # 分布式同步
            if dist.is_initialized():
                dist.all_reduce(scaling_factor)
                dist.all_reduce(bias_factor)
                world_size = dist.get_world_size()
                self.model.speech_scaling_factor.copy_(scaling_factor / world_size)
                self.model.speech_bias_factor.copy_(bias_factor / world_size)
        
        # 归一化
        audio_features = (audio_tokens + self.model.speech_bias_factor) * \
                        self.model.speech_scaling_factor
    
    # 通过连接器
    connect_features = self.model.acoustic_connector(audio_features)
    
    if return_unmask:
        return audio_features, connect_features
    return audio_features[speech_masks], connect_features[speech_masks]
```

### 3.2 关键点解析

**为什么需要speech_scaling_factor？**

```python
# VAE输出的潜在表示范围可能很大，需要归一化到合适的范围
# 这样可以：
# 1. 稳定训练
# 2. 与文本嵌入的scale匹配
# 3. 提高扩散模型的性能

# 计算方式：
mean = audio_tokens.mean()
std = audio_tokens.std()
normalized = (audio_tokens - mean) / std
```

**ddpm_batch_mul的作用**：

```python
# 扩散模型训练时，增加batch size可以：
# 1. 提高训练效率
# 2. 更好地估计梯度
# 3. 加速收敛

# 实现方式：将每个样本重复多次
speech_features_repeated = speech_features.repeat_interleave(4, dim=0)
# 原始: [10, 64]
# 重复后: [40, 64]
```

---

## 四、Generation：推理生成

### 4.1 生成流程（待续）

由于篇幅限制，生成部分将在下一篇文章详细讲解，包括：

- 自回归生成策略
- 扩散采样过程
- 音频解码和拼接
- 流式生成实现

---

## 小结

本文深入解析了VibeVoice TTS的实现：

1. ✅ 完整的数据流：从文本到音频
2. ✅ Processor的详细实现：脚本解析、语音编码
3. ✅ Model的前向传播：嵌入替换、扩散训练
4. ✅ 关键代码的逐行解读

**核心要点**：

- 🎯 使用VAE压缩音频，降低计算量
- 🎯 通过mask机制灵活控制语音token位置
- 🎯 扩散模型生成高质量语音潜在表示
- 🎯 多任务联合训练（LM + Diffusion）

下一篇将继续讲解ASR实现和流式处理！

---

如果觉得本文有帮助，欢迎点赞、收藏、关注！
