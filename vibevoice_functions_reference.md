# VibeVoice 函数参考手册

## 目录
1. [模型类方法](#模型类方法)
2. [处理器类方法](#处理器类方法)
3. [编解码器方法](#编解码器方法)
4. [工具函数](#工具函数)

---

## 一、模型类方法

### 1. VibeVoiceForConditionalGeneration

#### forward()
```python
def forward(
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    speech_tensors: Optional[torch.FloatTensor] = None,
    speech_masks: Optional[torch.BoolTensor] = None,
    speeches_loss_input: Optional[torch.FloatTensor] = None,
    speech_semantic_tensors: Optional[torch.FloatTensor] = None,
    acoustic_input_mask: Optional[torch.BoolTensor] = None,
    acoustic_loss_mask: Optional[torch.BoolTensor] = None,
    ddpm_batch_mul: int = 1,
    **kwargs
) -> Union[Tuple, VibeVoiceCausalLMOutputWithPast]
```
**用途**: TTS模型前向传播，计算语言模型损失和扩散损失

**返回**: `VibeVoiceCausalLMOutputWithPast` 包含loss、diffusion_loss、logits等

#### forward_speech_features()
```python
def forward_speech_features(
    speech_tensors=None,
    speech_masks=None,
    speech_type="audio",
    return_unmask=False
) -> tuple
```
**用途**: 处理语音特征，编码音频为潜在表示

**返回**: (audio_features, connect_features)

### 2. VibeVoiceASRForConditionalGeneration

#### encode_speech()
```python
def encode_speech(
    speech_tensors: torch.FloatTensor,
    speech_masks: Optional[torch.BoolTensor] = None,
    speech_semantic_tensors: Optional[torch.FloatTensor] = None,
    streaming_segment_duration: float = 60.0,
)
```
**用途**: 编码语音输入，支持长音频流式处理

**参数**:
- `streaming_segment_duration`: 流式分段时长（秒）

**返回**: 组合的语音特征

#### prepare_inputs_for_generation()
```python
def prepare_inputs_for_generation(
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    speech_tensors=None,
    speech_masks=None,
    speech_semantic_tensors=None,
    acoustic_input_mask=None,
    **kwargs,
)
```
**用途**: 为生成步骤准备输入（遵循Qwen2-VL模式）

**特性**: 仅在第一次前向传播时包含语音输入

### 3. VibeVoiceStreamingForConditionalGenerationInference

#### forward_lm()
```python
def forward_lm(
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]
```
**用途**: 基础文本LM的单次前向传播

**返回**: `BaseModelOutputWithPast` 包含last_hidden_state和past_key_values

#### forward_tts_lm()
```python
def forward_tts_lm(
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    lm_last_hidden_state: Optional[torch.FloatTensor] = None,
    tts_text_masks: Optional[torch.BoolTensor] = None,
    **kwargs,
) -> Union[Tuple, VibeVoiceCausalLMOutputWithPast]
```
**用途**: TTS LM的单次前向传播

**特性**: 
- 使用lm_last_hidden_state替换尾部嵌入
- 通过tts_text_masks添加类型嵌入
- 预测EOS（二分类器）

#### generate()
```python
@torch.no_grad()
def generate(
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    audio_streamer: Optional[Union[AudioStreamer, AsyncAudioStreamer]] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    speech_tensors: Optional[torch.FloatTensor] = None,
    speech_masks: Optional[torch.BoolTensor] = None,
    speech_input_mask: Optional[torch.BoolTensor] = None,
    tts_text_ids: Optional[torch.LongTensor] = None,
    return_speech: bool = True,
    cfg_scale: float = 1.0,
    stop_check_fn: Optional[Callable[[], bool]] = None,
    **kwargs,
) -> Union[torch.LongTensor, VibeVoiceGenerationOutput]
```
**用途**: 流式生成（文本窗口+语音扩散采样）

**特性**:
- 窗口化文本预填充
- 交错语音token扩散采样
- 支持实时音频流输出
- 支持外部停止检查

**返回**: `VibeVoiceGenerationOutput` 包含sequences和speech_outputs

---

## 二、处理器类方法

### 1. VibeVoiceProcessor

#### __call__()
```python
def __call__(
    text: Optional[Union[str, List[str], TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
    voice_samples: Optional[Union[List[Union[str, np.ndarray]], List[List[Union[str, np.ndarray]]]]] = None,
    padding: Union[bool, str, PaddingStrategy] = True,
    truncation: Union[bool, str, TruncationStrategy] = False,
    max_length: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_attention_mask: bool = True,
    **kwargs,
) -> BatchEncoding
```
**用途**: 处理文本和语音样本

**输入格式**:
- text: 脚本字符串、文件路径(.json/.txt)或列表
- voice_samples: 音频文件路径或numpy数组列表

**返回**: `BatchEncoding` 包含input_ids、attention_mask、speech_tensors等

#### from_pretrained()
```python
@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
```
**用途**: 从预训练模型加载处理器

#### save_pretrained()
```python
def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs)
```
**用途**: 保存处理器配置

#### prepare_speech_inputs()
```python
def prepare_speech_inputs(
    speech_inputs: List[np.ndarray],
    return_tensors: Optional[Union[str, TensorType]] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]
```
**用途**: 准备语音输入（padding和mask）

**返回**: 包含padded_speeches和speech_masks的字典

#### save_audio()
```python
def save_audio(
    audio: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    output_path: str = "output.wav",
    sampling_rate: Optional[int] = None,
    normalize: bool = False,
    batch_prefix: str = "audio_",
) -> str
```
**用途**: 保存音频到文件

### 2. VibeVoiceASRProcessor

#### __call__()
```python
def __call__(
    audio: Optional[Union[str, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, torch.Tensor]]]] = None,
    sampling_rate: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    padding: bool = True,
    max_length: Optional[int] = None,
    truncation: bool = False,
    add_generation_prompt: bool = True,
    use_streaming: bool = True,
    context_info: Optional[str] = None,
    **kwargs
) -> BatchEncoding
```
**用途**: 处理音频输入用于ASR

**参数**:
- `use_streaming`: 是否使用流式模式（<60s自动禁用）
- `context_info`: 上下文信息（热词、元数据）

**返回**: `BatchEncoding` 包含input_ids、acoustic_input_mask、speech_tensors等

#### post_process_transcription()
```python
def post_process_transcription(self, text: str) -> List[Dict[str, Any]]
```
**用途**: 后处理转录文本，提取结构化数据

**返回**: 转录片段列表，每个包含start_time、end_time、speaker_id、text

### 3. VibeVoiceTokenizerProcessor

#### __call__()
```python
def __call__(
    audio: Union[str, np.ndarray, List[float], List[np.ndarray], List[List[float]], List[str]] = None,
    sampling_rate: Optional[int] = None,
    return_tensors: Optional[str] = None,
    **kwargs,
)
```
**用途**: 处理音频（格式转换、归一化）

**支持格式**: 文件路径、numpy数组、列表

#### preprocess_audio()
```python
def preprocess_audio(
    audio_path_or_array: Union[str, np.ndarray],
    normalize: Optional[bool] = None,
) -> np.ndarray
```
**用途**: 预处理音频（向后兼容方法）

#### save_audio()
```python
def save_audio(
    audio: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    output_path: str = "output.wav",
    sampling_rate: Optional[int] = None,
    normalize: bool = False,
    batch_prefix: str = "audio_",
)
```
**用途**: 保存音频到WAV文件

---

## 三、编解码器方法

### 1. VibeVoiceAcousticTokenizerModel

#### encode()
```python
def encode(
    self,
    audio: torch.Tensor,
    cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
    sample_indices: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    is_final_chunk: bool = False,
    debug: bool = False,
) -> VibeVoiceTokenizerEncoderOutput
```
**用途**: 编码音频为潜在表示

**参数**:
- `cache`: 流式缓存
- `use_cache`: 是否使用流式模式
- `is_final_chunk`: 是否为最后一个块

**返回**: `VibeVoiceTokenizerEncoderOutput` 包含mean和std

#### decode()
```python
def decode(
    self,
    latent: torch.Tensor,
    cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
    sample_indices: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    debug: bool = False,
) -> torch.Tensor
```
**用途**: 解码潜在表示为音频

**返回**: 音频张量

### 2. VibeVoiceTokenizerEncoderOutput

#### sample()
```python
def sample(self, dist_type: str = 'gaussian') -> Tuple[torch.Tensor, torch.Tensor]
```
**用途**: 从分布采样

**参数**:
- `dist_type`: 分布类型 ('gaussian', 'laplace', 'cauchy', 'none')

**返回**: (采样值, 标准差)

---

## 四、工具函数

### 1. 音频处理函数

#### load_audio_use_ffmpeg()
```python
def load_audio_use_ffmpeg(
    file: str,
    resample: bool = False,
    target_sr: int = 24000
) -> tuple
```
**文件**: `vibevoice/processor/audio_utils.py`

**用途**: 使用FFmpeg加载音频

**返回**: (audio_data, sample_rate)

#### load_audio_bytes_use_ffmpeg()
```python
def load_audio_bytes_use_ffmpeg(
    data: bytes,
    *,
    resample: bool = False,
    target_sr: int = 24000
) -> tuple
```
**文件**: `vibevoice/processor/audio_utils.py`

**用途**: 从字节流解码音频（避免文件IO）

**返回**: (audio_data, sample_rate)

### 2. AudioNormalizer类方法

#### tailor_dB_FS()
```python
def tailor_dB_FS(self, audio: np.ndarray) -> tuple
```
**用途**: 调整音频到目标dB FS水平

**返回**: (normalized_audio, rms, scalar)

#### avoid_clipping()
```python
def avoid_clipping(self, audio: np.ndarray, scalar: Optional[float] = None) -> tuple
```
**用途**: 避免音频削波

**返回**: (normalized_audio, scalar)

#### __call__()
```python
def __call__(self, audio: np.ndarray) -> np.ndarray
```
**用途**: 归一化音频（调整dB FS + 避免削波）

### 3. 卷积工具函数

#### get_extra_padding_for_conv1d()
```python
def get_extra_padding_for_conv1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding_total: int = 0
) -> int
```
**文件**: `vibevoice/modular/modular_vibevoice_tokenizer.py`

**用途**: 计算卷积所需的额外padding

#### pad1d()
```python
def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = 'zero',
    value: float = 0.
) -> torch.Tensor
```
**文件**: `vibevoice/modular/modular_vibevoice_tokenizer.py`

**用途**: 1D padding（处理reflect模式的小输入）

#### unpad1d()
```python
def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]) -> torch.Tensor
```
**文件**: `vibevoice/modular/modular_vibevoice_tokenizer.py`

**用途**: 移除padding

### 4. 扩散相关函数

#### betas_for_alpha_bar()
```python
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
) -> torch.Tensor
```
**文件**: `vibevoice/schedule/dpm_solver.py`

**用途**: 创建beta调度

**支持类型**: cosine, exp, cauchy, laplace

#### rescale_zero_terminal_snr()
```python
def rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor
```
**文件**: `vibevoice/schedule/dpm_solver.py`

**用途**: 重缩放betas使终端SNR为零

### 5. 转换脚本函数

#### convert_vibevoice_nnscaler_checkpoint_to_hf()
```python
def convert_vibevoice_nnscaler_checkpoint_to_hf(
    checkpoint_path: str,
    pytorch_dump_folder_path: str,
    config_path: str = None,
)
```
**文件**: `vibevoice/scripts/convert_nnscaler_checkpoint_to_transformers.py`

**用途**: 转换nnscaler检查点为HuggingFace格式

**功能**:
- 加载nnscaler检查点
- 提取模型状态
- 创建HF模型
- 保存为分片格式

---

## 五、流式处理方法

### 1. VibeVoiceTokenizerStreamingCache

#### get()
```python
def get(self, layer_id: str, sample_indices: torch.Tensor) -> Optional[torch.Tensor]
```
**用途**: 获取缓存状态

#### set()
```python
def set(self, layer_id: str, sample_indices: torch.Tensor, states: torch.Tensor)
```
**用途**: 设置缓存状态

#### clear()
```python
def clear(
    layer_id: Optional[str] = None,
    sample_indices: Optional[torch.Tensor] = None
)
```
**用途**: 清除缓存

### 2. AudioStreamer

#### put()
```python
def put(self, audio_chunks: torch.Tensor, sample_indices: torch.Tensor)
```
**用途**: 放入音频块到队列

#### end()
```python
def end(self, sample_indices: Optional[torch.Tensor] = None)
```
**用途**: 结束生成信号

#### get_stream()
```python
def get_stream(self, sample_idx: int)
```
**用途**: 获取单个样本的音频流

---

这份文档涵盖了VibeVoice项目中最重要的函数和方法。每个函数都包含了签名、用途和关键参数说明。
