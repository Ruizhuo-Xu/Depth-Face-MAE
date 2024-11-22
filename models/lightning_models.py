import torch.nn as nn
from lightning import LightningModule
from torchmetrics.classification import Accuracy

from .models import make, register
from .util import optimizer, gallery

lock3dface_subsets = ["NU", "FE", "PS", "OC", "TM"]

@register("ModelForCls")
class ModelForCls(LightningModule):
    def __init__(self, model_spec, optimizer_spec, num_classes, *args, **kwargs):
        super().__init__()
        self.model = make(model_spec)
        self.optimizer_spec = optimizer_spec
        self.num_classes = num_classes
        self.kwargs = kwargs
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
        return loss
    
    def on_validation_epoch_start(self) -> None:
        self.gallery_feats, self.gallery_labels = \
            gallery.extract_gallery_features(self.model, self.kwargs["gallery_path"], self.device)
    
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
        optim = optimizer.make_optimizer(self.parameters(), self.optimizer_spec)
        return optim
        