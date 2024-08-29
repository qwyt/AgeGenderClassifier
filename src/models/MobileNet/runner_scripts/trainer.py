import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.loggers import WandbLogger
from typing import Dict, Any
import yaml

from src.models.MobileNet.callbacks import (
    EarlyStoppingCB,
    BestMetricsCallback,
    LRMonitorCallback,
)
from src.models.MobileNet.data_loader import create_dataloaders
from src.models.MobileNet.classifier import AgeGenderClassifier

PROJECT_NAME = "ag_classifier_main"


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config: Dict[str, Any], sweep_run=False, serialize_final=False):
    if sweep_run is False:
        wandb.finish()
        run = wandb.init(config=config, project=PROJECT_NAME)

    print(f"\n - - - \n(Sweep) config:\n{dict(config)}\n\n - - - \n")

    data = create_dataloaders(config)
    model = AgeGenderClassifier(config)
    wandb_logger = WandbLogger(project=PROJECT_NAME)

    callbacks = [
        # MyEarlyStopping(monitor="val_total_loss", patience=10, mode="min"),
        BestMetricsCallback(),
        Timer(duration=None, interval="epoch"),
        LRMonitorCallback(),
    ]

    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        callbacks=callbacks,
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
    )

    trainer.fit(model, datamodule=data)

    if serialize_final:
        run_name = wandb.run.name
        accuracy = trainer.callback_metrics.get("val_gender_acc", 0)
        epochs_run = trainer.current_epoch + 1

        wandb_run_name = wandb.run.name
        if "prefix" in config:
            save_path = f"{wandb_run_name}_{config['prefix']}_{run_name}_{epochs_run}_{accuracy:.4f}_.pth"
        else:
            save_path = f"{wandb_run_name}_{run_name}_{epochs_run}_{accuracy:.4f}.pth"

        save_model(model, save_path)

    if not sweep_run:
        wandb.finish()


def save_model(
    model: pl.LightningModule, save_path: str = "model_checkpoint.pth"
) -> None:
    """Saves the model state dict and full configuration."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config,  # Save the full configuration
        },
        f"model_store/{save_path}",
    )


def load_model(path: str = "model_checkpoint.pth") -> AgeGenderClassifier:
    """Loads a saved model checkpoint and returns an initialized AgeGenderClassifier."""
    checkpoint = torch.load(f"model_store/{path}")
    config = checkpoint.get("config", {})

    if not config:
        print(
            "Warning: No configuration found in the checkpoint. Using default configuration."
        )
        config = {"model_type": "mobilenet_v3_large"}  # Example default

    model = AgeGenderClassifier(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


if __name__ == "__main__":
    wandb.finish()

    # config = load_config("config/model/dynamic_aug_debug.yaml")
    # train(config, serialize_final=True)
    #
    #
    # config = load_config('config/model/swept-sweep-34_improved_only_valid.yaml')
    # train(config, serialize_final=True)
    #

    # config = load_config('config/model/swept-sweep-34_improved_DYNAMIC_AUG+mobile_net_large.yaml')
    # config = load_config('config/model/swept-sweep-34_improved_DYNAMIC_AUG_v3_large.yaml')
    # train(config, serialize_final=True)
    # config = load_config('config/model/no_weights_swept-sweep-34_improved_DYNAMIC_AUG.yaml')
    # train(config, serialize_final=True)
    config = load_config("config/model/dynamic_aug_debug.yaml")
    train(config, serialize_final=True)
