#!/usr/bin/env python3
"""
VibeVoice ASR 提示词构建机制演示代码

本脚本演示了 VibeVoice ASR 处理器中提示词构建的完整流程，
包括特殊token处理、chat_template应用和acoustic_input_mask创建。
"""

import math
import numpy as np
from typing import Dict, Any, Optional, List


class MockTokenizer:
    """模拟的tokenizer类，用于演示提示词构建机制"""
    
    def __init__(self):
        # 模拟特殊token IDs
        self.speech_start_id = 151644  # <|object_ref_start|>
        self.speech_end_id = 151645    # <|object_ref_end|>
        self.speech_pad_id = 151646    # <|box_start|>
        self.pad_id = 151643           # <|image_pad|>
        
        # 模拟chat template
        self.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
        
        # 模拟词汇表
        self.token_to_id = {
            "<|object_ref_start|>": self.speech_start_id,
            "<|object_ref_end|>": self.speech_end_id,
            "<|box_start|>": self.speech_pad_id,
            "<|image_pad|>": self.pad_id,
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
            "system": 1587,
            "user": 882,
            "assistant": 77091,
        }
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
    
    def convert_tokens_to_ids(self, token: str) -> int:
        """将token转换为ID"""
        return self.token_to_id.get(token, 0)
    
    def convert_ids_to_tokens(self, token_id: int) -> str:
        """将ID转换为token"""
        return self.id_to_token.get(token_id, "<unk>")
    
    def encode(self, text: str) -> List[int]:
        """简化的编码函数（实际实现会更复杂）"""
        # 这里只是演示，实际的tokenizer会进行复杂的subword分词
        words = text.split()
        return list(range(1000, 1000 + len(words)))  # 模拟token IDs
    
    def apply_chat_template(self, messages: List[Dict], tokenize: bool = False, add_generation_prompt: bool = False) -> str:
        """应用chat template"""
        result = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        if add_generation_prompt:
            result += "<|im_start|>assistant\n"
        
        if tokenize:
            return self.encode(result)
        else:
            return result


class PromptConstructionDemo:
    """提示词构建演示类"""
    
    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.speech_tok_compress_ratio = 3200  # 语音压缩比
        self.target_sample_rate = 24000        # 目标采样率
        
        # 系统提示词
        self.SYSTEM_PROMPT = "You are a helpful assistant that transcribes audio input into text output in JSON format."
    
    def demonstrate_prompt_construction(
        self, 
        audio_duration: float = 5.23,
        context_info: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        演示完整的提示词构建流程
        
        Args:
            audio_duration: 音频时长（秒）
            context_info: 上下文信息
            
        Returns:
            构建结果字典
        """
        print("=" * 60)
        print("VibeVoice ASR 提示词构建流程演示")
        print("=" * 60)
        
        # 1. 计算VAE token长度
        audio_samples = int(audio_duration * self.target_sample_rate)
        vae_tok_len = math.ceil(audio_samples / self.speech_tok_compress_ratio)
        
        print(f"\n1. 音频参数计算:")
        print(f"   音频时长: {audio_duration:.2f} 秒")
        print(f"   音频样本数: {audio_samples:,}")
        print(f"   VAE token长度: {vae_tok_len}")
        print(f"   压缩比: {self.speech_tok_compress_ratio}")
        
        # 2. 构建系统提示词
        system_message = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        system_prompt_text = self.tokenizer.apply_chat_template(system_message, tokenize=False)
        system_tokens = self.tokenizer.encode(system_prompt_text)
        
        print(f"\n2. 系统提示词构建:")
        print(f"   原始内容: {self.SYSTEM_PROMPT}")
        print(f"   Chat template处理后:")
        print(f"   {repr(system_prompt_text)}")
        print(f"   Token数量: {len(system_tokens)}")
        
        # 3. 构建语音占位符
        sp_start_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.speech_start_id)
        sp_pad_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.speech_pad_id)
        sp_end_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.speech_end_id)
        
        speech_placeholder = ''.join(
            [sp_start_token] + [sp_pad_token] * vae_tok_len + [sp_end_token]
        )
        
        print(f"\n3. 语音占位符构建:")
        print(f"   开始token: {sp_start_token} (ID: {self.tokenizer.speech_start_id})")
        print(f"   填充token: {sp_pad_token} (ID: {self.tokenizer.speech_pad_id})")
        print(f"   结束token: {sp_end_token} (ID: {self.tokenizer.speech_end_id})")
        print(f"   完整占位符: {speech_placeholder[:50]}...")
        
        # 4. 构建用户指令
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
        
        user_input_string = speech_placeholder + '\n' + user_suffix
        
        print(f"\n4. 用户指令构建:")
        print(f"   输出字段: {show_keys}")
        if context_info:
            print(f"   上下文信息: {context_info}")
        print(f"   用户指令: {user_suffix}")
        
        # 5. 应用chat template到用户输入
        user_message = [{"role": "user", "content": user_input_string}]
        user_tokens = self.tokenizer.apply_chat_template(user_message, tokenize=True)
        
        print(f"\n5. 用户输入Chat Template处理:")
        user_prompt_text = self.tokenizer.apply_chat_template(user_message, tokenize=False)
        print(f"   处理后文本: {repr(user_prompt_text[:100])}...")
        print(f"   Token数量: {len(user_tokens)}")
        
        # 6. 组合完整token序列
        full_tokens = system_tokens + user_tokens
        
        print(f"\n6. 完整token序列:")
        print(f"   系统token数: {len(system_tokens)}")
        print(f"   用户token数: {len(user_tokens)}")
        print(f"   总token数: {len(full_tokens)}")
        
        # 7. 创建acoustic_input_mask
        acoustic_input_mask = [
            1 if token == self.tokenizer.speech_pad_id else 0 
            for token in full_tokens
        ]
        
        speech_token_count = sum(acoustic_input_mask)
        
        print(f"\n7. Acoustic Input Mask:")
        print(f"   语音token数量: {speech_token_count}")
        print(f"   预期VAE token数: {vae_tok_len}")
        print(f"   Mask示例: {acoustic_input_mask[:20]}...")
        
        # 8. 生成完整的对话示例
        complete_conversation = system_prompt_text + self.tokenizer.apply_chat_template(
            user_message, tokenize=False, add_generation_prompt=True
        )
        
        print(f"\n8. 完整对话格式:")
        print("   " + "─" * 50)
        print(f"   {complete_conversation}")
        print("   " + "─" * 50)
        
        return {
            "input_ids": full_tokens,
            "acoustic_input_mask": acoustic_input_mask,
            "vae_tok_len": vae_tok_len,
            "speech_placeholder": speech_placeholder,
            "complete_conversation": complete_conversation,
            "audio_duration": audio_duration,
            "context_info": context_info
        }
    
    def analyze_special_tokens(self):
        """分析特殊token的使用"""
        print("\n" + "=" * 60)
        print("特殊Token分析")
        print("=" * 60)
        
        special_tokens = [
            ("语音开始", "<|object_ref_start|>", self.tokenizer.speech_start_id),
            ("语音结束", "<|object_ref_end|>", self.tokenizer.speech_end_id),
            ("语音填充", "<|box_start|>", self.tokenizer.speech_pad_id),
            ("填充token", "<|image_pad|>", self.tokenizer.pad_id),
        ]
        
        for name, token, token_id in special_tokens:
            print(f"{name:8}: {token:20} (ID: {token_id})")
        
        print(f"\n特殊token的作用:")
        print(f"- <|object_ref_start|>: 标记语音内容的开始位置")
        print(f"- <|object_ref_end|>:   标记语音内容的结束位置")
        print(f"- <|box_start|>:       语音内容的占位符，数量等于VAE token长度")
        print(f"- <|image_pad|>:       序列填充token，用于批处理对齐")
    
    def demonstrate_context_variations(self):
        """演示不同上下文信息的效果"""
        print("\n" + "=" * 60)
        print("上下文信息变化演示")
        print("=" * 60)
        
        context_examples = [
            None,
            "说话人: 张三",
            "关键词: 人工智能, 机器学习",
            "说话人: 张三, 李四; 会议主题: 技术讨论; 音频质量: 较差",
        ]
        
        for i, context in enumerate(context_examples, 1):
            print(f"\n示例 {i}: {context or '无上下文信息'}")
            result = self.demonstrate_prompt_construction(
                audio_duration=3.5, 
                context_info=context
            )
            print(f"生成的用户指令长度: {len(result['complete_conversation'])} 字符")


def main():
    """主函数：运行所有演示"""
    demo = PromptConstructionDemo()
    
    # 1. 基本提示词构建演示
    demo.demonstrate_prompt_construction(
        audio_duration=5.23,
        context_info="说话人: 张三; 关键词: 深度学习, 神经网络"
    )
    
    # 2. 特殊token分析
    demo.analyze_special_tokens()
    
    # 3. 上下文信息变化演示
    demo.demonstrate_context_variations()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()