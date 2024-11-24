import torch.nn as nn
from lightning import LightningModule
from torchmetrics.classification import Accuracy
import timm.optim.optim_factory as optim_factory

from .models import make, register
from .util import optimizer, gallery, lr_sched

lock3dface_subsets = ["NU", "FE", "PS", "OC", "TM"]

@register("ModelForCls")
class ModelForCls(LightningModule):
    def __init__(self, model_spec, optimizer_spec, num_classes,
                 lr_sched_spec=None, steps_per_epoch=None, *args, **kwargs):
        super().__init__()
        self.model = make(model_spec)
        self.optimizer_spec = optimizer_spec
        self.num_classes = num_classes
        self.kwargs = kwargs
        self.lr_sched_spec = lr_sched_spec
        self.steps_per_epoch = steps_per_epoch
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.metrics = nn.ParameterDict()
        self.metrics["total_accuracy"] = Accuracy(task="multiclass", num_classes=num_classes)
        if self.kwargs.get("is_lock3dface", False):
            for k in lock3dface_subsets:
                self.metrics[f"{k}_accuracy"] = Accuracy(task="multiclass", num_classes=num_classes)
        
    def training_step(self, batch, batch_idx):
        x, label, _ = batch
        loss, preds = self.model(x, label)
        train_acc = self.train_acc(preds, label)
        self.log("train/acc", train_acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.global_step)
        return loss
    
    def on_validation_epoch_start(self) -> None:
        self.gallery_feats, self.gallery_labels = \
            gallery.extract_gallery_features(self.model, self.kwargs["anno_file"], self.device)
    
    def validation_step(self, batch, batch_idx):
        x, label, infos = batch
        if self.kwargs.get("validation_on_gallery", False):
            feats = self.model(x, only_return_feats=True)
            is_lock3dface = self.kwargs.get("is_lock3dface", False)
            results = gallery.evaluation(feats, label, self.gallery_feats, self.gallery_labels,
                                         self.metrics, is_lock3dface, img_infos=infos)
            for k, v in self.metrics.items():
                self.log(f"val/{k}", v, prog_bar=True, logger=True,
                         on_step=False, on_epoch=True)
        else:
            loss, preds = self.model(x, label)
            self.log("val/loss", loss, prog_bar=True, logger=True)
            return loss
        return None
    
    def configure_optimizers(self):
        if "weight_decay" in self.optimizer_spec["args"]:
            param_groups = optim_factory.param_groups_weight_decay(self, self.optimizer_spec["args"]["weight_decay"])
            self.optimizer_spec["args"].pop("weight_decay")
        optim = optimizer.make_optimizer(param_groups, self.optimizer_spec)
        self.lr_scheduler = None
        if self.lr_sched_spec is not None:
            self.lr_sched_spec["args"].update({"steps_per_epoch": self.steps_per_epoch})
            self.lr_scheduler = lr_sched.make_lr_scheduler(optim, self.lr_sched_spec)
        return optim


@register("ModelForMAE")
class ModelForMAE(LightningModule):
    def __init__(self, model_spec, optimizer_spec, mask_ratio,
                 lr_sched_spec=None, steps_per_epoch=None, *args, **kwargs):
        super().__init__()
        self.model = make(model_spec)
        self.optimizer_spec = optimizer_spec
        self.mask_ratio = mask_ratio
        self.kwargs = kwargs
        self.lr_sched_spec = lr_sched_spec
        self.steps_per_epoch = steps_per_epoch
        
    def training_step(self, batch, batch_idx):
        x, label, _ = batch
        loss, preds = self.model(x, self.mask_ratio)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.global_step)
        return loss
    
    def configure_optimizers(self):
        if "weight_decay" in self.optimizer_spec["args"]:
            param_groups = optim_factory.param_groups_weight_decay(self, self.optimizer_spec["args"]["weight_decay"])
            self.optimizer_spec["args"].pop("weight_decay")
        optim = optimizer.make_optimizer(param_groups, self.optimizer_spec)
        self.lr_scheduler = None
        if self.lr_sched_spec is not None:
            self.lr_sched_spec["args"].update({"steps_per_epoch": self.steps_per_epoch})
            self.lr_scheduler = lr_sched.make_lr_scheduler(optim, self.lr_sched_spec)
        return optim
        