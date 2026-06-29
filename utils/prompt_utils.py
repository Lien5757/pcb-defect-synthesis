from typing import List
from utils.prompt_loader import get_prompt_list


def set_prompts(batch_size: int, data: str = 'exp2') -> List[str]:
    """Generate batch of prompts from config for given dataset.

    Args:
        batch_size: Number of prompts to generate.
        data: Dataset name (e.g., 'exp4', 'exp5').

    Returns:
        List of prompts repeated to match batch size.

    Raises:
        ValueError: If dataset not found in config.
    """
    prompt_list = get_prompt_list(data)

    repeat_factor = batch_size // len(prompt_list)
    prompt_batch = prompt_list * repeat_factor

    remainder = batch_size % len(prompt_list)
    if remainder > 0:
        prompt_batch += prompt_list[:remainder]

    return prompt_batch


def set_prompts_given(batch_size: int, prompt: str) -> List[str]:
    """Generate batch of identical custom prompts.

    Args:
        batch_size: Number of prompts to generate.
        prompt: Custom prompt text to repeat.

    Returns:
        List with prompt repeated batch_size times.
    """
    return [prompt] * batch_size