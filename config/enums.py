from enum import Enum


class PromptMode(str, Enum):
    """Prompt generation mode for inference."""

    MULTI = "multi"
    SINGLE = "single"
