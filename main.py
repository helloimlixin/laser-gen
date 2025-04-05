import os
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    trainer = pl.Trainer(
        max_epochs=cfg.model.max_epochs,
        devices=cfg.model.devices,
        callbacks=[ModelCheckpoint(monitor="val_loss", save_last=True), TQDMProgressBar(refresh_rate=1)],
        logger=TensorBoardLogger("lightning_logs", name="laser-gen"),
    )
    trainer.fit(model=cfg.model, datamodule=cfg.datamodule)
    trainer.test(model=cfg.model, datamodule=cfg.datamodule)

if __name__ == "__main__":
    main()
