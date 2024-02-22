# src/modules/training/trainers.py
class AutoEncoderTrainer:
    """Trainer for autoencoder models."""
    def __init__(self, trainer, logdir):
        self.trainer = trainer
        self.logdir = logdir

    def train_autoencoder(self, model, data):
        try:
            self.trainer.fit(model, data)
        except Exception:
            self._handle_training_exception()
            raise
        if not self.opt.no_test and not self.trainer.interrupted:
            self.trainer.test(model, data)

    def _handle_training_exception(self):
        if self.opt.debug and self.trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()