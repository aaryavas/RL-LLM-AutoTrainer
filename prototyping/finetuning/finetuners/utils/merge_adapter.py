"""
Adapter Merge Utility for VB-LoRA models.

Merges PEFT adapters back into the base model to create a standalone model
that can be used for further training (e.g., ORPO) or inference.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_base_model_from_adapter(adapter_path: str) -> Optional[str]:
    """
    Extract base model name from adapter config.

    Args:
        adapter_path: Path to the adapter directory

    Returns:
        Base model name or None if not found
    """
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get("base_model_name_or_path")
    return None


def merge_adapter(
    adapter_path: str,
    output_path: str,
    base_model_name: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    save_tokenizer: bool = True,
) -> str:
    """
    Merge PEFT adapter weights into the base model.

    Args:
        adapter_path: Path to the PEFT adapter directory
        output_path: Path to save the merged model
        base_model_name: Base model name (auto-detected from adapter config if not provided)
        torch_dtype: Data type for model loading
        device_map: Device mapping strategy
        save_tokenizer: Whether to save the tokenizer alongside the model

    Returns:
        Path to the saved merged model
    """
    logger.info(f"Loading adapter from {adapter_path}")

    # Get base model name
    if base_model_name is None:
        base_model_name = get_base_model_from_adapter(adapter_path)

    if not base_model_name:
        raise ValueError(
            "Base model name not found in adapter config. "
            "Please specify base_model_name parameter."
        )

    logger.info(f"Base model: {base_model_name}")

    # Load base model
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Load PEFT model
    logger.info("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge adapter weights into base model
    logger.info("Merging adapter weights...")
    model = model.merge_and_unload()

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save merged model
    logger.info(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)

    # Save tokenizer
    if save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(output_path)
        logger.info("Tokenizer saved")

    # Save merge metadata
    metadata = {
        "base_model": base_model_name,
        "adapter_path": str(adapter_path),
        "merge_dtype": str(torch_dtype),
    }
    with open(os.path.join(output_path, "merge_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Merge complete.")

    # Explicit cleanup to free memory for subsequent steps (e.g., vLLM)
    del model
    if "base_model" in locals():
        del base_model

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_path


def merge_for_orpo(
    vblora_model_path: str,
    output_dir: Optional[str] = None,
    base_model_name: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Merge VB-LoRA adapter for use with ORPO training.

    This is a convenience function that handles the merge process
    specifically for the VB-LoRA -> ORPO pipeline.

    Args:
        vblora_model_path: Path to the VB-LoRA fine-tuned model (adapter)
        output_dir: Directory to save merged model (default: vblora_model_path + "_merged")
        base_model_name: Base model name (auto-detected if not provided)

    Returns:
        Tuple of (merged_model_path, base_model_name)
    """
    # Determine output path
    if output_dir is None:
        output_dir = str(
            Path(vblora_model_path).parent / (Path(vblora_model_path).name + "_merged")
        )

    # Check if already merged
    if os.path.exists(os.path.join(output_dir, "config.json")):
        logger.info(f"Merged model already exists at {output_dir}")
        # Get base model name from merge metadata if available
        metadata_path = os.path.join(output_dir, "merge_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                base_model_name = metadata.get("base_model", base_model_name)
        return output_dir, base_model_name

    # Get base model name from adapter config
    if base_model_name is None:
        base_model_name = get_base_model_from_adapter(vblora_model_path)

    if not base_model_name:
        raise ValueError("Could not determine base model name")

    # Merge
    merged_path = merge_adapter(
        adapter_path=vblora_model_path,
        output_path=output_dir,
        base_model_name=base_model_name,
    )

    return merged_path, base_model_name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge PEFT adapter into base model")
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the PEFT adapter directory",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the merged model"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=None,
        help="Base model name (auto-detected if not provided)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model dtype for loading",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    merge_adapter(
        args.adapter_path,
        args.output_path,
        args.base_model_name,
        torch_dtype=dtype_map[args.dtype],
    )
