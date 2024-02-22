# src/modules/training/configurators.py
class TrainerConfigurator:
    """Configurator for PyTorch Lightning trainers."""
    def __init__(self, opt, unknown):
        self.opt = opt
        self.unknown = unknown

    def configure_trainer(self):
        trainer_config = self._merge_trainer_configs()
        trainer = self._initialize_trainer(trainer_config)
        return trainer

    def _merge_trainer_configs(self):
        # Merge trainer configurations from lightning_config and command-line options
        configs = [OmegaConf.load(cfg) for cfg in self.opt.base]
        cli = OmegaConf.from_dotlist(self.unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # Default to ddp
        trainer_config["accelerator"] = "ddp"
        for k in self._nondefault_trainer_args():
            trainer_config[k] = getattr(self.opt, k)
        if "gpus" not in trainer_config:
            del trainer_config["accelerator"]
        return trainer_config

    def _nondefault_trainer_args(self):
        # Get non-default trainer arguments
        parser = argparse.ArgumentParser()
        parser = pl.Trainer.add_argparse_args(parser)
        args = parser.parse_args([])
        return sorted(k for k in vars(args) if getattr(self.opt, k) != getattr(args, k))

    def _initialize_trainer(self, trainer_config):
        # Initialize trainer from configurations
        trainer_opt = argparse.Namespace(**trainer_config)
        trainer = pl.Trainer.from_argparse_args(trainer_opt)
        return trainer