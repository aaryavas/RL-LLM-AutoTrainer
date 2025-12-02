"""
Model loading and VB-LoRA configuration for fine-tuning.
"""

import logging
from typing import List, Optional

import torch
from peft import VBLoRAConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Handles loading and configuring models for VB-LoRA fine-tuning.
    Supports quantization and PEFT integration.
    """

    def __init__(
        self,
        model_name_or_path: str,
        bits: int = 4,
        double_quant: bool = True,
        quant_type: str = "nf4",
        use_auth_token: bool = False,
        trust_remote_code: bool = False,
        device_map: str = "auto",
        max_memory_MB: int = 80000,
    ):
        """
        Initialize model loader.

        Args:
            model_name_or_path: Model name or path
            bits: Quantization bits (4, 8, 16, or 32)
            double_quant: Use double quantization
            quant_type: Quantization type ('nf4' or 'fp4')
            use_auth_token: Use HuggingFace auth token
            trust_remote_code: Trust remote code
            device_map: Device mapping strategy
            max_memory_MB: Maximum memory per GPU in MB
        """
        self.model_name_or_path = model_name_or_path
        self.bits = bits
        self.double_quant = double_quant
        self.quant_type = quant_type
        self.use_auth_token = use_auth_token
        self.trust_remote_code = trust_remote_code
        self.device_map = device_map
        self.max_memory_MB = max_memory_MB

    def load_model(
        self,
        use_fp16: bool = False,
        use_bf16: bool = False,
    ) -> AutoModelForCausalLM:
        """
        Load model with quantization configuration.

        Args:
            use_fp16: Use FP16 precision
            use_bf16: Use BF16 precision

        Returns:
            Loaded model
        """
        logger.info(f"Loading model: {self.model_name_or_path}")

        # Determine compute dtype
        compute_dtype = self._get_compute_dtype(use_fp16, use_bf16)

        # Configure quantization
        quantization_config = self._create_quantization_config(compute_dtype)

        # Get max memory configuration
        max_memory = self._get_max_memory()

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=quantization_config if self.bits in [4, 8] else None,
            torch_dtype=compute_dtype,
            device_map=self.device_map,
            max_memory=max_memory,
            trust_remote_code=self.trust_remote_code,
            use_auth_token=self.use_auth_token,
        )

        # Set model attributes for parallelism
        setattr(model, "model_parallel", True)
        setattr(model, "is_parallelizable", True)

        logger.info("Model loaded successfully")

        return model

    def apply_vblora(
        self,
        model,
        vblora_config: dict,
        gradient_checkpointing: bool = True,
    ):
        """
        Apply VB-LoRA to the model.

        Args:
            model: Base model
            vblora_config: VB-LoRA configuration dictionary
            gradient_checkpointing: Enable gradient checkpointing

        Returns:
            Model with VB-LoRA applied
        """
        logger.info("Applying VB-LoRA to model")

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )

        # Find target modules
        target_modules = vblora_config.get("target_modules", ["all"])
        if "all" in target_modules:
            target_modules = self._find_all_linear_names(model)
            logger.info(f"Auto-detected target modules: {target_modules}")

        # Create VB-LoRA config
        peft_config = VBLoRAConfig(
            r=vblora_config["r"],
            vblora_dropout=vblora_config["vblora_dropout"],
            target_modules=target_modules,
            num_vectors=vblora_config["num_vectors"],
            vector_length=vblora_config["vector_length"],
            save_only_topk_weights=vblora_config.get("save_only_topk_weights", True),
            task_type=vblora_config.get("task_type", "CAUSAL_LM"),
        )

        # Apply PEFT
        model = get_peft_model(model, peft_config)

        # Print trainable parameters
        model.print_trainable_parameters()

        logger.info("VB-LoRA applied successfully")

        return model

    def _get_compute_dtype(self, use_fp16: bool, use_bf16: bool) -> torch.dtype:
        """Determine compute dtype based on precision settings."""
        if use_bf16:
            return torch.bfloat16
        elif use_fp16:
            return torch.float16
        else:
            return torch.float32

    def _create_quantization_config(
        self,
        compute_dtype: torch.dtype,
    ) -> Optional[BitsAndBytesConfig]:
        """Create quantization configuration."""
        if self.bits not in [4, 8]:
            return None

        return BitsAndBytesConfig(
            load_in_4bit=self.bits == 4,
            load_in_8bit=self.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.double_quant,
            bnb_4bit_quant_type=self.quant_type,
        )

    def _get_max_memory(self) -> Optional[dict]:
        """Get max memory configuration for devices."""
        if not torch.cuda.is_available():
            return None

        n_gpus = torch.cuda.device_count()
        max_memory = {i: f"{self.max_memory_MB}MB" for i in range(n_gpus)}

        # Handle distributed training
        import os

        if os.environ.get("LOCAL_RANK") is not None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            return {"": max_memory[local_rank]}

        return max_memory

    def _find_all_linear_names(self, model) -> List[str]:
        """
        Find all linear layer names in the model.

        Args:
            model: Model to search

        Returns:
            List of linear layer names
        """
        import bitsandbytes as bnb

        # Determine linear class based on quantization
        if self.bits == 4:
            linear_cls = bnb.nn.Linear4bit
        elif self.bits == 8:
            linear_cls = bnb.nn.Linear8bitLt
        else:
            linear_cls = torch.nn.Linear

        # Find all linear layers
        linear_names = set()
        for name, module in model.named_modules():
            if isinstance(module, linear_cls):
                parts = name.split(".")
                linear_names.add(parts[0] if len(parts) == 1 else parts[-1])

        # Remove output layer
        if "lm_head" in linear_names:
            linear_names.remove("lm_head")

        return list(linear_names)
