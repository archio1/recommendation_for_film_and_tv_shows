from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from huggingface_hub import login
import torch
from src.utils.helpers import load_config, get_quantization_config

class FinetunedRecommender:
    def __init__(self, config_path):
        """Initialization of fine-tuned model with LoRA."""
        config = load_config(config_path)
        
        login(token=config['model']['huggingface_token'])
        
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model_name'])
        
        quantization_config = get_quantization_config(config)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config['model']['base_model_name'],
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model = PeftModel.from_pretrained(
            self.base_model,
            config['model']['lora_path'],
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        self.model = self.model.merge_and_unload()
        
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        
        self.model.eval()
        
        self.generation_params = config['generation']

    def generate(self, prompt_text, use_second_params=False):
        """Generate a recommendation based on the query text."""
        messages = [{"role": "user", "content": prompt_text}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        params = self.generation_params.copy()
        if use_second_params:
            params['top_p'] = self.generation_params['top_p_second']
        
        outputs = self.pipe(
            prompt,
            max_new_tokens=params['max_new_tokens'],
            do_sample=params['do_sample'],
            temperature=params['temperature'],
            top_k=params['top_k'],
            top_p=params['top_p'],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        return outputs[0]["generated_text"]