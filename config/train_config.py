import json
import os
from dataclasses import dataclass, asdict


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

    @staticmethod
    def from_json(config_path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        return TrainingConfig(**config_dict)

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
