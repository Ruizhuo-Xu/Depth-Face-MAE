import yaml
import argparse

from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor

from models import models
from datasets import datasets

seed_everything(42, workers=True)

def build_dataloader(cfg):
    train_set = datasets.make(cfg["train_set"])
    val_set = datasets.make(cfg["val_set"])
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    return train_loader, val_loader

def main(cfg):
    wandb_logger_cfg = cfg["trainer"].get("wandb_logger", None)
    wandb_logger = WandbLogger(**wandb_logger_cfg) if wandb_logger_cfg else None
    if wandb_logger:
        wandb_logger.experiment.config.update(cfg)
        
    train_loader, val_loader = build_dataloader(cfg["dataset"])
    model = models.make(cfg["model"], args={"steps_per_epoch": len(train_loader)})
        
    callbacks = [LearningRateMonitor(logging_interval="step")]
    ckpt_cfg = cfg["trainer"].get("checkpoint", None)
    if ckpt_cfg:
        ckpt_callback = ModelCheckpoint(**ckpt_cfg)
        callbacks.append(ckpt_callback)
        
    trainer_cfg = cfg["trainer"]["args"]
    trainer = Trainer(
        **trainer_cfg,
        accelerator="gpu",
        logger=wandb_logger,
        callbacks=callbacks
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Config file loaded: {args.config}")
        
    main(cfg)