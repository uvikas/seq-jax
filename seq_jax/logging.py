from dataclasses import dataclass, field


@dataclass
class WandbConfig:
    project: str = "seq-jax"
    entity: str = "my-gcp-project"
    mode: str = "online"


@dataclass
class LoggingConfig:
    save_freq: int = 1000
    wandb: WandbConfig = field(default_factory=WandbConfig)


def make_meter(cfg: LoggingConfig):
    pass
