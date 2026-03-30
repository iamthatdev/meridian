from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import yaml
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    url: str
    pool_size: int = 5
    max_overflow: int = 10

@dataclass
class ModelConfig:
    rw_model_id: str
    math_model_id: str
    fallback_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"

@dataclass
class LoRAConfig:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    gradient_checkpointing: bool = True

@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    max_seq_length_rw: int = 4096
    max_seq_length_math: int = 2048
    warmup_ratio: float = 0.05
    early_stopping_patience: int = 2

    def __post_init__(self):
        # Convert learning_rate from string to float if needed
        if isinstance(self.learning_rate, str):
            self.learning_rate = float(self.learning_rate)

@dataclass
class PathConfig:
    data_dir: Path = Path("data")
    training_dir: Path = Path("data/training")
    generated_dir: Path = Path("data/generated")
    validated_dir: Path = Path("data/validated")
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("outputs/logs")

@dataclass
class Config:
    app_env: Literal["local", "production"]
    database: DatabaseConfig
    models: ModelConfig
    lora: LoRAConfig
    quantization: QuantizationConfig
    training: TrainingConfig
    paths: PathConfig
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, env: Literal["local", "production"] = "local") -> "Config":
        config_file = Path(f"configs/{env}.yaml")
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)

        return cls(
            app_env=env,
            database=DatabaseConfig(**config_dict["database"]),
            models=ModelConfig(**config_dict["models"]),
            lora=LoRAConfig(**config_dict.get("lora", {})),
            quantization=QuantizationConfig(**config_dict.get("quantization", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            paths=PathConfig(**{k: Path(v) for k, v in config_dict.get("paths", {}).items()}),
            log_level=config_dict.get("log_level", "INFO")
        )

def load_config(env: Literal["local", "production"] = None) -> Config:
    """Load configuration from environment variable or YAML file."""
    import os
    if env is None:
        env = os.getenv("APP_ENV", "local")
    return Config.from_yaml(env)