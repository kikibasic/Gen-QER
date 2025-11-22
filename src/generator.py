import os
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

class LLMGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = None
        self.hf_model = None
        self.tokenizer = None
        
        if 'gpt' in model_name:
            # OpenAI Client
            api_key = os.environ.get('OPENAI_KEY')
            if not api_key:
                raise ValueError("OPENAI_KEY environment variable is not set.")
            self.client = OpenAI(api_key=api_key)
        else:
            # HuggingFace Model
            print(f"Loading local LLM: {model_name}...")
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype='auto'
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, messages: List[dict]) -> str:
        """Unified generation method for both OpenAI and Local LLMs."""
        if 'gpt' in self.model_name:
            response = self.client.chat.completions.create(
                model=self.model_name if 'gpt' in self.model_name else 'gpt-3.5-turbo',
                messages=messages
            )
            return response.choices[0].message.content
        
        elif 'Qwen' in self.model_name:
            return self._chat_qwen(messages)
        elif '01-ai' in self.model_name:
            return self._chat_yi(messages)
        else:
            # Default HF
            return self._chat_default(messages)

    def _chat_qwen(self, messages: List[dict]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages[:-1], 
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.hf_model.device)
        with torch.no_grad():
            generated_ids = self.hf_model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _chat_yi(self, messages: List[dict]) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages[:-1], 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors='pt'
        )
        with torch.no_grad():
            output_ids = self.hf_model.generate(input_ids.to('cuda'))
        return self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    def _chat_default(self, messages: List[dict]) -> str:
        # Generic HF implementation if needed
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
        output = self.hf_model.generate(input_ids, max_new_tokens=512)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)