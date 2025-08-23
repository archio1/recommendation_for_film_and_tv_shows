import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
from src.utils.helpers import load_config, load_tokenizer, get_quantization_config

class FinetunedRecommender:
    def __init__(self, config_path):
        """Initialize fine-tuned model with LoRA."""
        self.config = load_config(config_path)

        self.tokenizer = load_tokenizer(
            model_name=self.config['model']['base_model_name'],
            huggingface_token=self.config['model']['huggingface_token']
        )

        self.model = self._load_base_model()

        self.model = self._apply_lora()

        self.pipe = self._create_pipeline()

        model_device = next(self.model.parameters()).device
        print(f"Модель размещена на устройстве: {model_device}")

        self.model.eval()

        self.generation_params = self.config['generation']

    def _load_base_model(self):
        """Load base causal LM model using config parameters."""
        return AutoModelForCausalLM.from_pretrained(
            self.config['model']['base_model_name'],
            quantization_config=get_quantization_config(self.config),
            torch_dtype=getattr(torch, self.config['model']['torch_dtype']),
            device_map=self.config['model']['device_map']
        )

    def _apply_lora(self):
        """Apply LoRA adapter to the base model and merge it."""
        model = PeftModel.from_pretrained(
            self.model,
            self.config['model']['lora_path'],
            device_map=self.config['model']['device_map'],
            torch_dtype=getattr(torch, self.config['model']['torch_dtype'])
        )
        return model.merge_and_unload()

    def _create_pipeline(self):
        """Create text generation pipeline."""
        return pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, prompt_text, use_second_params=False):
        """Generate recommendation and trim to complete entries."""
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

        generated_text = outputs[0]["generated_text"]

        # Убираем заголовок ассистента
        generated_text = re.sub(
            r'<\|start_header_id\|>assistant<\|end_header_id\|>\n?',
            '',
            generated_text,
            flags=re.IGNORECASE
        ).strip()

        lines = generated_text.splitlines()
        trimmed_lines = [ln for ln in lines if ln.strip().endswith('.')]
        return '\n'.join(trimmed_lines)

    def analyze_token_probabilities(self, prompt_text, max_tokens=50, top_k=5):
        """Analyze token probabilities for generated sequence."""
        messages = [{"role": "user", "content": prompt_text}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated_ids = []
        all_top_tokens = []
        all_top_probs = []

        with torch.no_grad():
            for step in range(max_tokens):
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, top_k)
                top_tokens = [self.tokenizer.decode(idx) for idx in top_indices[0]]
                all_top_tokens.append(top_tokens)
                all_top_probs.append(top_probs[0].tolist())
                next_token = top_indices[0][0]  # Greedy choice
                generated_ids.append(next_token)
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones(1, 1, dtype=torch.long).to(self.model.device)], dim=-1)

        generated_text = self.tokenizer.decode(generated_ids)
        print(f"Generated text (greedy): {generated_text}")
        print("\nToken probabilities for each step:")
        for step, (tokens, probs) in enumerate(zip(all_top_tokens, all_top_probs), 1):
            print(f"Step {step}, Top predicted tokens:")
            for token, prob in zip(tokens, probs):
                print(f"Token: '{token}', Probability: {prob:.4f}")
        return generated_text, all_top_tokens, all_top_probs