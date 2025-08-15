import yaml
from transformers import BitsAndBytesConfig
import torch

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_quantization_config(config):
    return BitsAndBytesConfig(
        load_in_4bit=config['quantization']['load_in_4bit'],
        bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, config['quantization']['bnb_4bit_compute_dtype']),
        bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant']
    )