"""
Helper utility functions for VB-LoRA fine-tuning.
"""

import json
import torch
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def ensure_dir(directory: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        directory: Path to directory

    Returns:
        Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save (must be JSON serializable)
        file_path: Path to output file
        indent: JSON indentation level
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)

    logger.info(f"Saved JSON to {file_path}")


def load_json(file_path: str) -> Any:
    """
    Load data from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded JSON from {file_path}")
    return data


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": 0,
        "cuda_devices": [],
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "compute_capability": torch.cuda.get_device_capability(i),
            }
            for i in range(torch.cuda.device_count())
        ]

    return info


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to save logs to
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        ensure_dir(Path(log_file).parent)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return {
        "trainable": trainable,
        "total": total,
        "trainable_percent": 100 * trainable / total if total > 0 else 0,
    }
