import json
import os
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration for SD Inpainting."""
    data_dir: str
    project_name: str = "exp1"
    is_transform: bool = False
    num_epochs: int = 500
    batch_size: int = 1
    lr: float = 5e-7
    weight_decay: float = 1e-6
    use_warmup: bool = True
    warmup_ratio: float = 0.05
    use_weighted_sampler: bool = True
    min_delta: float = 1e-4
    save_interval: int = 200
    plot_interval: int = 10

    @staticmethod
    def from_json(config_path: str) -> "TrainingConfig":
        """Load config from JSON file with validation.

        Args:
            config_path: Path to JSON configuration file.

        Returns:
            TrainingConfig instance.

        Raises:
            FileNotFoundError: If config file does not exist.
            json.JSONDecodeError: If config file is not valid JSON.
            ValueError: If required parameters are missing.
            TypeError: If parameter types are invalid.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Please ensure the file exists at the specified path."
            )

        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Config file is not valid JSON: {config_path}\n"
                f"Error: {str(e)}",
                e.doc,
                e.pos
            )
        except IOError as e:
            raise IOError(f"Cannot read config file: {config_path}\nError: {str(e)}")

        # Validate required parameters
        if 'data_dir' not in config_dict:
            raise ValueError(
                f"Config file must contain 'data_dir' parameter.\n"
                f"Config file: {config_path}\n"
                f"Current keys: {list(config_dict.keys())}"
            )

        try:
            config = TrainingConfig(**config_dict)
        except TypeError as e:
            raise TypeError(
                f"Invalid parameters in config file: {config_path}\n"
                f"Error: {str(e)}"
            )

        logger.info(f"Config loaded from: {config_path}")
        return config

    def to_json(self, output_path: str) -> None:
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_args(data_dir: str, **kwargs) -> "TrainingConfig":
        """Create config from keyword arguments, using defaults for missing keys."""
        return TrainingConfig(data_dir=data_dir, **kwargs)

    def update_from_args(self, **kwargs) -> None:
        """Update config with CLI args (only non-None values)."""
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
