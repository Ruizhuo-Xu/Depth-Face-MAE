from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from models import models
from datasets import datasets


if __name__ == "__main__":
    path = "/home/rz/code/Depth-Face-MAE/data/lock3dface.json"
    train_set_cfg = {
        "name": "ImgDataset",
        "args": {
            "anno_file": path,
            "split": "train",
        }
    }
    val_set_cfg = {
        "name": "ImgDataset",
        "args": {
            "anno_file": path,
            "split": "val",
            "test_mode": True,
        }
    }
    train_set = datasets.make(train_set_cfg)
    val_set = datasets.make(val_set_cfg)
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
    
    model_cfg = {
        "name": "ModelForCls",
        "args": {
            "model_spec": {
                "name": "ViT",
                "args": {
                    "img_size": 128,
                    "patch_size": 16,
                    "in_chans": 1,
                    "embed_dim": 256,
                    "depth": 12,
                    "num_heads": 8,
                    "mlp_ratio": 4,
                }
            },
            "optimizer_spec": {
                "name": "adamw",
                "args": {"lr": 3.e-4, "weight_decay": 0.05}
            },
            "num_classes": 509,
            "validation_on_gallery": True,
            "is_lock3dface": True,
            "gallery_path": path
        }
    }

    model = models.make(model_cfg)
    wandb_logger = WandbLogger(project="Depth-Face-MAE", name="baseline")
    trainer = Trainer(
        accelerator="gpu",
        devices=[3],
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        max_epochs=50,
        precision="16-mixed",
        logger=wandb_logger
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)