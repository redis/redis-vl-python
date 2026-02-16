"""Pretrained LLM Router configurations.

This module provides pre-built router configurations that can be loaded
without needing to re-embed references.
"""

from pathlib import Path

PRETRAINED_DIR = Path(__file__).parent


def get_pretrained_path(name: str) -> Path:
    """Get path to a pretrained configuration.
    
    Args:
        name: Pretrained config name (e.g., "default", "cost_optimized")
        
    Returns:
        Path to the pretrained JSON file.
    """
    path = PRETRAINED_DIR / f"{name}.json"
    if not path.exists():
        available = [f.stem for f in PRETRAINED_DIR.glob("*.json")]
        raise ValueError(
            f"Pretrained config '{name}' not found. "
            f"Available: {available}"
        )
    return path
