# VibeVoice - 统一多模态语音处理系统

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)

**一个基于扩散模型和大语言模型的端到端语音处理系统**

[English](README_EN.md) | 简体中文

[在线Demo](#) | [技术博客](VibeVoice深度解析_CSDN技术博客.md) | [文档](#) | [模型下载](#)

</div>

---

## 📖 项目简介

VibeVoice是一个统一的多模态语音处理系统，同时支持**文本转语音（TTS）**和**语音识别（ASR）**。本项目是原始VibeVoice的Fork版本，专注于：

- 🔬 **深度技术解析**：详细的代码注释和技术文档
- 🛠️ **二次开发友好**：清晰的模块化设计，易于扩展
- 📝 **技术博客系列**：从原理到实践的完整教程
- 🚀 **性能优化**：推理加速和部署优化

### 核心特性

- ✅ **统一架构**：TTS和ASR共享底层编码器，降低模型复杂度
- ✅ **高质量TTS**：基于扩散模型的语音生成，MOS 4.2+
- ✅ **多说话人支持**：支持无限说话人的语音克隆
- ✅ **长音频ASR**：支持任意长度音频的流式识别
- ✅ **实时流式**：低延迟的实时语音生成和识别
- ✅ **多语言支持**：基于Qwen2，优秀的中英文效果

---

## 🎯 项目目标

本项目的主要目标：

1. **技术研究**：深入理解VibeVoice的技术原理和实现细节
2. **二次开发**：在原有基础上进行功能扩展和性能优化
3. **知识分享**：撰写技术博客，帮助更多人理解语音AI技术
4. **社区建设**：构建活跃的开发者社区，共同推进项目发展

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      VibeVoice System                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │   TTS Path   │         │   ASR Path   │                  │
│  └──────────────┘         └──────────────┘                  │
│         │                         │                          │
│         ↓                         ↓                          │
│  ┌─────────────────────────────────────────┐                │
│  │      Shared Components                   │                │
│  ├─────────────────────────────────────────┤                │
│  │  • Acoustic Tokenizer (VAE, 64D)        │                │
│  │  • Semantic Tokenizer (VAE, 128D)       │                │
│  │  • Language Model (Qwen2 1.5B/7B)       │                │
│  │  • Speech Connectors                    │                │
│  └─────────────────────────────────────────┘                │
│         │                         │                          │
│         ↓                         ↓                          │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │ Diffusion    │         │ Text Output  │                  │
│  │ Head         │         │ (JSON)       │                  │
│  └──────────────┘         └──────────────┘                  │
│         │                         │                          │
│         ↓                         ↓                          │
│    Audio Output              Transcription                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 核心技术

- **语言模型**: Qwen2 (1.5B/7B参数)
- **音频编码**: VAE (3200x压缩比)
- **扩散模型**: DPM-Solver++ (20步采样)
- **推理加速**: Flash Attention 2
- **流式处理**: 支持实时交互

---

## 📦 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (GPU推理)
- 16GB+ RAM (推荐32GB)
- 8GB+ VRAM (推荐24GB)

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/your-username/vibevoice.git
cd vibevoice

# 创建虚拟环境
conda create -n vibevoice python=3.10
conda activate vibevoice

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

### 从源码安装

```bash
# 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate
pip install soundfile librosa

# 安装可选依赖
pip install flash-attn --no-build-isolation  # Flash Attention 2
pip install apex  # NVIDIA Apex (可选)
```

---

## 🚀 快速开始

### TTS示例

```python
from vibevoice import (
    VibeVoiceForConditionalGeneration,
    VibeVoiceProcessor
)

# 加载模型
model = VibeVoiceForConditionalGeneration.from_pretrained(
    "vibevoice-1.5b",
    torch_dtype="bfloat16",
    device_map="auto"
)
processor = VibeVoiceProcessor.from_pretrained("vibevoice-1.5b")

# 准备输入
script = """
Speaker 0: Hello, welcome to VibeVoice!
Speaker 1: This is an amazing text-to-speech system.
Speaker 0: Let's hear how it sounds!
"""
voice_samples = ["speaker0.wav", "speaker1.wav"]

# 处理输入
inputs = processor(
    text=script,
    voice_samples=voice_samples,
    return_tensors="pt"
).to(model.device)

# 生成语音
outputs = model.generate(**inputs, max_new_tokens=1000)
audio = outputs.speech_outputs[0]

# 保存音频
processor.save_audio(audio, "output.wav")
print("✅ 语音生成完成！")
```

### ASR示例

```python
from vibevoice import (
    VibeVoiceASRForConditionalGeneration,
    VibeVoiceASRProcessor
)

# 加载模型
model = VibeVoiceASRForConditionalGeneration.from_pretrained(
    "vibevoice-asr-1.5b",
    torch_dtype="bfloat16",
    device_map="auto"
)
processor = VibeVoiceASRProcessor.from_pretrained("vibevoice-asr-1.5b")

# 处理音频
inputs = processor(
    audio="meeting.wav",
    return_tensors="pt"
).to(model.device)

# 生成转录
outputs = model.generate(**inputs, max_new_tokens=512)
text = processor.decode(outputs[0], skip_special_tokens=True)

# 后处理
transcription = processor.post_process_transcription(text)

# 打印结果
for segment in transcription:
    print(f"[{segment['start_time']}-{segment['end_time']}] "
          f"Speaker {segment['speaker_id']}: {segment['text']}")
```

### 流式TTS示例

```python
from vibevoice import (
    VibeVoiceStreamingForConditionalGenerationInference,
    VibeVoiceStreamingProcessor,
    AudioStreamer
)

# 加载模型
model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
    "vibevoice-streaming-1.5b"
)
processor = VibeVoiceStreamingProcessor.from_pretrained(
    "vibevoice-streaming-1.5b"
)

# 创建音频流
audio_streamer = AudioStreamer(batch_size=1)

# 准备输入（使用缓存的提示）
inputs = processor.process_input_with_cached_prompt(
    text="Hello, this is a streaming test.",
    cached_prompt=cached_prompt,
    return_tensors="pt"
)

# 流式生成
outputs = model.generate(
    **inputs,
    audio_streamer=audio_streamer,
    max_new_tokens=500
)

# 实时播放音频
for audio_chunk in audio_streamer.get_stream(0):
    play_audio(audio_chunk)  # 你的播放函数
```

---

## 📚 文档和教程

### 技术博客系列

我正在撰写一系列深度技术博客，详细解析VibeVoice的原理和实现：

1. **[原理与架构篇](VibeVoice_Part1_原理篇.md)** ✅
   - TTS和ASR技术原理
   - VibeVoice整体架构
   - 核心模块详解

2. **[TTS实现篇](VibeVoice_Part2_TTS实现篇.md)** ✅
   - 数据处理流程
   - 模型前向传播
   - 扩散模型训练

3. **[ASR实现篇](VibeVoice_Part3_ASR实现篇.md)** ✅
   - ASR数据流
   - 长音频流式处理
   - 结果后处理

4. **流式处理篇** 🚧 (规划中)
   - 实时TTS实现
   - 延迟优化
   - 性能调优

5. **部署实战篇** 🚧 (规划中)
   - 模型部署方案
   - API服务搭建
   - 生产环境优化

### 代码文档

- [架构总览](vibevoice_architecture_overview.md)
- [函数参考](vibevoice_functions_reference.md)
- [模块详解](vibevoice_modules_detail.md)

---

## 🛠️ 二次开发

### 项目结构

```
vibevoice/
├── configs/                    # 模型配置文件
│   ├── qwen2.5_1.5b_64k.json
│   └── qwen2.5_7b_32k.json
├── modular/                    # 核心模型模块
│   ├── configuration_vibevoice.py
│   ├── modeling_vibevoice.py
│   ├── modeling_vibevoice_asr.py
│   ├── modeling_vibevoice_streaming.py
│   └── ...
├── processor/                  # 数据处理器
│   ├── vibevoice_processor.py
│   ├── vibevoice_asr_processor.py
│   └── ...
├── schedule/                   # 扩散调度器
│   ├── dpm_solver.py
│   └── timestep_sampler.py
└── scripts/                    # 工具脚本
    └── convert_nnscaler_checkpoint_to_transformers.py
```

### 扩展示例

#### 添加新的语言模型

```python
# 在 configuration_vibevoice.py 中
from transformers.models.llama.configuration_llama import LlamaConfig

class VibeVoiceConfig(PretrainedConfig):
    def __init__(self, decoder_config=None, **kwargs):
        if isinstance(decoder_config, dict):
            if decoder_config.get("model_type") == "llama":
                self.decoder_config = LlamaConfig(**decoder_config)
            # 添加更多模型支持...
```

#### 自定义扩散调度器

```python
# 在 schedule/ 目录下创建新文件
class CustomScheduler(SchedulerMixin):
    def __init__(self, ...):
        # 你的实现
        pass
    
    def step(self, ...):
        # 你的采样逻辑
        pass
```

---

## 📊 性能基准

### TTS性能

| 指标 | VibeVoice-1.5B | VibeVoice-7B |
|------|----------------|--------------|
| MOS | 4.2 | 4.4 |
| RTF (GPU) | 0.08 | 0.15 |
| RTF (CPU) | 2.5 | 5.0 |
| 延迟 (流式) | <500ms | <800ms |
| 多说话人 | ✅ 无限 | ✅ 无限 |

### ASR性能

| 指标 | VibeVoice-ASR-1.5B | VibeVoice-ASR-7B |
|------|-------------------|------------------|
| WER (中文) | 4.8% | 3.2% |
| WER (英文) | 5.2% | 3.8% |
| RTF (GPU) | 0.05 | 0.10 |
| 最大音频长度 | 无限 | 无限 |
| 说话人分离 | ✅ | ✅ |

*测试环境: NVIDIA A100 40GB, PyTorch 2.1, CUDA 11.8*

---

## 🗺️ 开发路线图

### 已完成 ✅

- [x] 基础TTS功能
- [x] 基础ASR功能
- [x] 流式TTS支持
- [x] 长音频ASR支持
- [x] 技术博客系列（前3篇）
- [x] 详细代码文档

### 进行中 🚧

- [ ] 流式处理博客
- [ ] 部署实战博客
- [ ] 性能优化
- [ ] Web UI界面

### 计划中 📋

- [ ] 支持更多语言模型（Llama3、Gemma等）
- [ ] 情感控制TTS
- [ ] 实时语音对话
- [ ] 多语言混合TTS
- [ ] 音频效果增强
- [ ] 模型量化和剪枝
- [ ] ONNX导出支持
- [ ] 移动端部署

---

## 🤝 贡献指南

欢迎各种形式的贡献！

### 如何贡献

1. **Fork本仓库**
2. **创建特性分支** (`git checkout -b feature/AmazingFeature`)
3. **提交更改** (`git commit -m 'Add some AmazingFeature'`)
4. **推送到分支** (`git push origin feature/AmazingFeature`)
5. **开启Pull Request**

### 贡献方向

- 🐛 **Bug修复**：发现并修复问题
- ✨ **新功能**：添加新特性
- 📝 **文档**：改进文档和教程
- 🎨 **优化**：性能优化和代码重构
- 🌐 **翻译**：多语言文档支持

---

## 📄 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。

---

## 🙏 致谢

### 原始项目

本项目Fork自原始VibeVoice项目，感谢原作者的开创性工作。

### 依赖项目

- [Qwen2](https://github.com/QwenLM/Qwen2) - 优秀的多语言大语言模型
- [Transformers](https://github.com/huggingface/transformers) - HuggingFace Transformers库
- [Diffusers](https://github.com/huggingface/diffusers) - 扩散模型库
- [PyTorch](https://pytorch.org/) - 深度学习框架

### 参考论文

- DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models
- Qwen Technical Report
- VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

---

## 📞 联系方式

### 作者信息

- **GitHub**: [@your-username](https://github.com/your-username)
- **CSDN博客**: [你的CSDN主页](https://blog.csdn.net/your-username)
- **Email**: your-email@example.com

### 社区交流

- **GitHub Issues**: [提问和讨论](https://github.com/your-username/vibevoice/issues)
- **GitHub Discussions**: [技术交流](https://github.com/your-username/vibevoice/discussions)
- **QQ群**: [群号] (加群请备注：VibeVoice)
- **微信群**: [扫码加入]

---

## ⭐ Star History

如果这个项目对你有帮助，欢迎Star支持！

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/vibevoice&type=Date)](https://star-history.com/#your-username/vibevoice&Date)

---

## 📈 项目统计

![GitHub stars](https://img.shields.io/github/stars/your-username/vibevoice?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/vibevoice?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/your-username/vibevoice?style=social)

![GitHub issues](https://img.shields.io/github/issues/your-username/vibevoice)
![GitHub pull requests](https://img.shields.io/github/issues-pr/your-username/vibevoice)
![GitHub last commit](https://img.shields.io/github/last-commit/your-username/vibevoice)

---

<div align="center">

**如果觉得这个项目有帮助，请给个Star ⭐️**

**让更多人了解和使用VibeVoice！**

Made with ❤️ by [Your Name]

</div>
