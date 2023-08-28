import timm.scheduler


class LightningCosineLRScheduler(timm.scheduler.CosineLRScheduler):
    """Pytorch Lightning Wrapper for timm LR scheduler. Keeps track of current number of epochs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_epoch = 0

    def step(self):
        super().step(self.current_epoch)
        self.current_epoch += 1
