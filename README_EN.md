# VibeVoice - Unified Multimodal Speech Processing System

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)

**An end-to-end speech processing system based on diffusion models and large language models**

English | [简体中文](README.md)

[Demo](#) | [Blog Series](#) | [Documentation](#) | [Model Download](#)

</div>

---

## 📖 Introduction

VibeVoice is a unified multimodal speech processing system that supports both **Text-to-Speech (TTS)** and **Automatic Speech Recognition (ASR)**. This project is a fork of the original VibeVoice, focusing on:

- 🔬 **In-depth Technical Analysis**: Detailed code annotations and technical documentation
- 🛠️ **Developer-Friendly**: Clear modular design, easy to extend
- 📝 **Technical Blog Series**: Complete tutorials from theory to practice
- 🚀 **Performance Optimization**: Inference acceleration and deployment optimization

### Key Features

- ✅ **Unified Architecture**: TTS and ASR share underlying encoders
- ✅ **High-Quality TTS**: Diffusion-based speech generation, MOS 4.2+
- ✅ **Multi-Speaker Support**: Unlimited speaker voice cloning
- ✅ **Long-Form ASR**: Streaming recognition for audio of any length
- ✅ **Real-time Streaming**: Low-latency real-time speech generation and recognition
- ✅ **Multilingual**: Based on Qwen2, excellent Chinese and English performance

---

## 🎯 Project Goals

Main objectives of this project:

1. **Technical Research**: Deep understanding of VibeVoice's technical principles
2. **Secondary Development**: Feature extensions and performance optimization
3. **Knowledge Sharing**: Write technical blogs to help more people understand speech AI
4. **Community Building**: Build an active developer community

---

## 🏗️ System Architecture

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

### Core Technologies

- **Language Model**: Qwen2 (1.5B/7B parameters)
- **Audio Encoding**: VAE (3200x compression ratio)
- **Diffusion Model**: DPM-Solver++ (20-step sampling)
- **Inference Acceleration**: Flash Attention 2
- **Streaming Processing**: Real-time interaction support

---

## 📦 Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (for GPU inference)
- 16GB+ RAM (32GB recommended)
- 8GB+ VRAM (24GB recommended)

### Quick Install

```bash
# Clone repository
git clone https://github.com/your-username/vibevoice.git
cd vibevoice

# Create virtual environment
conda create -n vibevoice python=3.10
conda activate vibevoice

# Install dependencies
pip install -r requirements.txt

# Install project
pip install -e .
```

---

## 🚀 Quick Start

### TTS Example

```python
from vibevoice import (
    VibeVoiceForConditionalGeneration,
    VibeVoiceProcessor
)

# Load model
model = VibeVoiceForConditionalGeneration.from_pretrained(
    "vibevoice-1.5b",
    torch_dtype="bfloat16",
    device_map="auto"
)
processor = VibeVoiceProcessor.from_pretrained("vibevoice-1.5b")

# Prepare input
script = """
Speaker 0: Hello, welcome to VibeVoice!
Speaker 1: This is an amazing text-to-speech system.
"""
voice_samples = ["speaker0.wav", "speaker1.wav"]

# Process input
inputs = processor(
    text=script,
    voice_samples=voice_samples,
    return_tensors="pt"
).to(model.device)

# Generate speech
outputs = model.generate(**inputs, max_new_tokens=1000)
audio = outputs.speech_outputs[0]

# Save audio
processor.save_audio(audio, "output.wav")
```

### ASR Example

```python
from vibevoice import (
    VibeVoiceASRForConditionalGeneration,
    VibeVoiceASRProcessor
)

# Load model
model = VibeVoiceASRForConditionalGeneration.from_pretrained(
    "vibevoice-asr-1.5b",
    torch_dtype="bfloat16",
    device_map="auto"
)
processor = VibeVoiceASRProcessor.from_pretrained("vibevoice-asr-1.5b")

# Process audio
inputs = processor(
    audio="meeting.wav",
    return_tensors="pt"
).to(model.device)

# Generate transcription
outputs = model.generate(**inputs, max_new_tokens=512)
text = processor.decode(outputs[0], skip_special_tokens=True)

# Post-process
transcription = processor.post_process_transcription(text)

# Print results
for segment in transcription:
    print(f"[{segment['start_time']}-{segment['end_time']}] "
          f"Speaker {segment['speaker_id']}: {segment['text']}")
```

---

## 📚 Documentation

### Technical Blog Series

I'm writing a series of in-depth technical blogs analyzing VibeVoice:

1. **Principles and Architecture** ✅
2. **TTS Implementation** ✅
3. **ASR Implementation** ✅
4. **Streaming Processing** 🚧 (Planned)
5. **Deployment Guide** 🚧 (Planned)

### Code Documentation

- [Architecture Overview](vibevoice_architecture_overview.md)
- [Function Reference](vibevoice_functions_reference.md)
- [Module Details](vibevoice_modules_detail.md)

---

## 📊 Performance Benchmarks

### TTS Performance

| Metric | VibeVoice-1.5B | VibeVoice-7B |
|--------|----------------|--------------|
| MOS | 4.2 | 4.4 |
| RTF (GPU) | 0.08 | 0.15 |
| Latency (Streaming) | <500ms | <800ms |
| Multi-Speaker | ✅ Unlimited | ✅ Unlimited |

### ASR Performance

| Metric | VibeVoice-ASR-1.5B | VibeVoice-ASR-7B |
|--------|-------------------|------------------|
| WER (Chinese) | 4.8% | 3.2% |
| WER (English) | 5.2% | 3.8% |
| RTF (GPU) | 0.05 | 0.10 |
| Max Audio Length | Unlimited | Unlimited |

*Test Environment: NVIDIA A100 40GB, PyTorch 2.1, CUDA 11.8*

---

## 🗺️ Roadmap

### Completed ✅

- [x] Basic TTS functionality
- [x] Basic ASR functionality
- [x] Streaming TTS support
- [x] Long-form ASR support
- [x] Technical blog series (first 3 parts)
- [x] Detailed code documentation

### In Progress 🚧

- [ ] Streaming processing blog
- [ ] Deployment guide blog
- [ ] Performance optimization
- [ ] Web UI interface

### Planned 📋

- [ ] Support more language models (Llama3, Gemma, etc.)
- [ ] Emotion-controlled TTS
- [ ] Real-time voice conversation
- [ ] Multilingual mixed TTS
- [ ] Audio effect enhancement
- [ ] Model quantization and pruning
- [ ] ONNX export support
- [ ] Mobile deployment

---

## 🤝 Contributing

Contributions are welcome!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## 🙏 Acknowledgments

### Original Project

This project is forked from the original VibeVoice project. Thanks to the original authors for their pioneering work.

### Dependencies

- [Qwen2](https://github.com/QwenLM/Qwen2) - Excellent multilingual large language model
- [Transformers](https://github.com/huggingface/transformers) - HuggingFace Transformers library
- [Diffusers](https://github.com/huggingface/diffusers) - Diffusion models library
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

## 📞 Contact

### Author

- **GitHub**: [@your-username](https://github.com/your-username)
- **Email**: your-email@example.com

### Community

- **GitHub Issues**: [Questions and Discussions](https://github.com/your-username/vibevoice/issues)
- **GitHub Discussions**: [Technical Exchange](https://github.com/your-username/vibevoice/discussions)

---

<div align="center">

**If you find this project helpful, please give it a Star ⭐️**

**Help more people discover VibeVoice!**

Made with ❤️ by [Your Name]

</div>
