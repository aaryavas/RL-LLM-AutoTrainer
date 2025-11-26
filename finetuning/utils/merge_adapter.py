import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_adapter(adapter_path, output_path, base_model_name=None):
    logger.info(f"Loading adapter from {adapter_path}")
    
    if base_model_name is None:
        # Try to read from adapter_config.json
        import json
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                base_model_name = config.get("base_model_name_or_path")
    
    if not base_model_name:
        raise ValueError("Base model name not found. Please specify base_model_name.")
        
    logger.info(f"Base model: {base_model_name}")
    
    # Load base model
    # Note: We need to load in full precision or bf16 for merging, usually not 4-bit if we want to save a clean model
    # But if memory is an issue, we might have to be careful. 
    # SmolLM2-1.7B is small enough to load in fp16/bf16 on most GPUs.
    
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info("Loading PEFT model...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("Merging adapter...")
    model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    
    # Save tokenizer as well
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)
    
    logger.info("Merge complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, default=None)
    args = parser.parse_args()
    
    merge_adapter(args.adapter_path, args.output_path, args.base_model_name)
