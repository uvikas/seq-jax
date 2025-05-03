from dataclasses import dataclass


@dataclass
class LoggingConfig:
    logdir: str = "/tmp/bc/"
    save_freq: int = 1000

    wandb_project: str
    wandb_entity: str = "my-gcp-project"
    wandb_mode: str = "online"


def make_meter(cfg: LoggingConfig):
    pass
