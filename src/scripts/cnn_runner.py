import pytorch_lightning as pl
import torch

import draft.models.MobileNet.data_loader
from draft.models.MobileNet.classifier import (
    GenderClassifier,
    ModelType,
    MyEarlyStopping,
    ProgressBarToggleCallback,
)


def main():

    DS_PATH = "dataset/folds/fold_0"

    # Your existing code for parameter selection remains the same
    selected_params = {
        "batch_size": 64,
        "base_lr": 0.001,
        "weight_decay": 0.01,
        "lr_mult": 10,
        "lr_scheduler": "one_cycle",
        "freeze_epochs": 2,
        "pct_start": 0.3,
        "aug_mode": None,
        # 'aug_mode': 'mult_1.25_more_trans'
    }
    batch_size = selected_params["batch_size"]
    base_lr = selected_params["base_lr"]
    weight_decay = selected_params["weight_decay"]
    lr_mult = selected_params["lr_mult"]
    lr_scheduler_type = selected_params["lr_scheduler"]
    freeze_epochs = selected_params["freeze_epochs"]
    pct_start = selected_params["pct_start"]

    balancing = None

    aug_mode = selected_params["aug_mode"]

    num_epochs = 1
    freeze_epochs = 0

    data = draft.models.MobileNet.data_loader.create_dataloaders(
        64, aug_mode=aug_mode, path=DS_PATH
    )
    model = GenderClassifier(
        model_type=ModelType.MOBILENET_V3_LARGE,
        weight_decay=weight_decay,
        base_lr=base_lr,
        lr_mult=lr_mult,
        lr_scheduler_type=lr_scheduler_type,
        pct_start=pct_start,
    )

    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=4)

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    from pytorch_lightning.callbacks import Timer

    monitor: str = "val_loss"
    patience = 6
    es_mode = "min"

    # checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)
    callbacks = [
        Timer(duration=None, interval="epoch"),
        MyEarlyStopping(monitor=monitor, mode=es_mode, patience=patience),
        ProgressBarToggleCallback(),
    ]

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        # callbacks=[early_stop_callback, checkpoint_callback],
        accelerator="gpu",
        # log_every_n_steps=1,
        devices=1,
        enable_progress_bar=True,
    )
    trainer.fit(model, datamodule=data)

    # Print final metrics
    print("Final metrics:")
    for k, v in trainer.callback_metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
