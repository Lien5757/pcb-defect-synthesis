import json
import os
from typing import Dict, List, Any


def load_prompt_config(config_path: str = 'config/prompts.json') -> Dict[str, Dict[str, str]]:
    """Load prompt configuration from JSON file.

    Args:
        config_path: Path to prompt config JSON file.

    Returns:
        Dictionary mapping dataset names to class-prompt mappings.

    Raises:
        FileNotFoundError: If config file not found.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Prompt config not found at {config_path}")

    with open(config_path) as f:
        return json.load(f)


def get_class_to_prompt_map(data: str, config_path: str = 'config/prompts.json') -> Dict[str, str]:
    """Get class-to-prompt mapping for data preprocessing.

    Args:
        data: Dataset name (e.g., 'exp4', 'exp5').
        config_path: Path to prompt config JSON file.

    Returns:
        Dictionary mapping class names to prompts.

    Raises:
        ValueError: If dataset not found in config.
    """
    config = load_prompt_config(config_path)
    if data not in config:
        raise ValueError(f"Unknown dataset: {data}. Available: {list(config.keys())}")

    return config[data]


def get_prompt_list(data: str, config_path: str = 'config/prompts.json') -> List[str]:
    """Get list of prompts for a dataset (used by inference).

    Args:
        data: Dataset name (e.g., 'exp4', 'exp5').
        config_path: Path to prompt config JSON file.

    Returns:
        List of prompt strings.

    Raises:
        ValueError: If dataset not found in config.
    """
    class_to_prompt = get_class_to_prompt_map(data, config_path)
    return list(class_to_prompt.values())
