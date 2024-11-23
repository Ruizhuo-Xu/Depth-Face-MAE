import math
from torch.optim.lr_scheduler import MultiStepLR

def make_lr_scheduler(optimizer, scheduler_spec):
    Scheduler = {
        'MultiStepLr': MultiStepLR,
        'CosineDecayWithWarmup': CosineDecayWithWarmup,
    }[scheduler_spec['name']]
    scheduler = Scheduler(optimizer, **scheduler_spec['args'])
    return scheduler

class CosineDecayWithWarmup():
    def __init__(self,
                 optimizer,
                 warmup_fraction,
                 max_epochs,
                 lr,
                 steps_per_epoch,
                 min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_fraction * max_epochs * steps_per_epoch
        self.max_steps = max_epochs * steps_per_epoch
        self.base_lr = lr
        self.min_lr = min_lr
    
    def step(self, step):
        if step < self.warmup_steps:
            lr = self.base_lr * step / self.warmup_steps 
        else:
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)))
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
