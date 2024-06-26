# train_autoencoder.py
import argparse, os, sys, datetime, glob
import pytorch_lightning as pl
import logging
import torch
import torch.multiprocessing as mp

import signal
from packaging import version
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from src.util.misc import log_opts, set_submodule_paths, set_cache_directories
set_submodule_paths(submodule_dir="submodules")
from ldm.util import instantiate_from_config

def get_parser(**parser_kwargs):
    """
    Returns an ArgumentParser object with predefined arguments for command-line parsing.

    Args:
        **parser_kwargs: Additional keyword arguments to be passed to the ArgumentParser constructor.

    Returns:
        argparse.ArgumentParser: An ArgumentParser object with predefined arguments.

    Raises:
        argparse.ArgumentTypeError: If the value provided for a boolean argument is not a valid boolean value.
    """
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--transformers_cache",type=str,default=".cache/transformers_cache", help="transformers cache directory",)
    parser.add_argument("--torch_home", type=str, default=".cache/torch_home", help="torch home directory")
    parser.add_argument("--logging_level", type=str, default="INFO", help="logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("-n", "--name", type=str, const=True, default="test", nargs="?", help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir")
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml", default=list(), help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.")
    parser.add_argument("-t", "--train", type=str2bool, const=True, default=False, nargs="?", help="train")
    parser.add_argument("--no-test", type=str2bool, const=True, default=False, nargs="?", help="disable test")
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-d", "--debug", type=str2bool, nargs="?", const=True, default=False, help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=64, help="seed for seed_everything")
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name")
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="directory for logs")
    parser.add_argument("--scale_lr", type=str2bool,nargs="?",const=True,default=True,help="scale base-lr by ngpu * batch_size * n_accumulate")
    return parser

def nondefault_trainer_args(opt):
    """Return the non-default trainer arguments.

    This function takes an `opt` object as input and compares its attributes with the default
    arguments of the `Trainer` class. It returns a sorted list of attribute names that have
    different values between `opt` and the default arguments.

    Args:
        opt (object): An object containing the arguments.

    Returns:
        list: A sorted list of attribute names that have different values between `opt` and
        the default arguments.
    """
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def parse_args():
    """
    Parse command line arguments.

    Returns:
        opt (argparse.Namespace): Parsed command line arguments.
        unknown (list): List of unknown command line arguments.
    """
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()
    return opt, unknown

def get_nowname(opt, now):
    """Get the name for the current execution.

    This function determines the name for the current execution based on the provided options and current time.

    Args:
        opt (object): The options object containing the execution parameters.
        now (str): The current time.

    Raises:
        ValueError: If the specified resume file cannot be found.

    Returns:
        tuple: A tuple containing the updated options object and the name for the current execution.
    """
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
    
    return opt, nowname

def merge_configs(opt, unknown):
    """
    Merge multiple configurations into a single configuration.

    Args:
        opt (List[str]): List of configuration file paths.
        unknown (List[str]): List of command-line arguments.

    Returns:
        OmegaConf.DictConfig: Merged configuration.
    """
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    return config

def set_trainer_config(opt, lightning_config):
    """Set the trainer configuration based on the provided options.

    Args:
        opt (object): The options object containing the trainer configuration.
        lightning_config (object): The lightning configuration object.

    Returns:
        tuple: A tuple containing the trainer configuration and a boolean indicating whether to use CPU.
    """
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["strategy"] = "ddp"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["strategy"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        logging.info(f"Running on GPUs {gpuinfo}")
        cpu = False

    return trainer_config, cpu

def get_logger_cfgs(opt, logdir, nowname, lightning_config):
    """
    Get the logger configurations for the training process.

    Args:
        opt: The options for the training process.
        logdir: The directory to save the logs.
        nowname: The name of the logger.
        lightning_config: The configuration for the lightning logger.

    Returns:
        The logger configurations.

    """
    # default logger configs
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "offline": opt.debug,
                "id": nowname,
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            }
        },
    }
    default_logger_cfg = default_logger_cfgs["testtube"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg

def get_model_checkpoint_cfgs(ckptdir, model, lightning_config):
    """
    Get the configuration for model checkpoint callbacks.

    Args:
        ckptdir (str): The directory to save the checkpoints.
        model: The model object.
        lightning_config: The configuration object for PyTorch Lightning.

    Returns:
        dict: The configuration for model checkpoint callbacks.
    """
    
    default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
                'save_weights_only': True
            }
        }
    if hasattr(model, "monitor"):
        logging.info(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg =  OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    logging.info(f"Using ModelCheckpoint with {modelckpt_cfg}")
    return modelckpt_cfg

def get_callbacks_cfgs(opt, now, logdir, ckptdir, cfgdir, config, lightning_config, trainer_opt, modelckpt_cfg):
    """
    Returns the configuration for callbacks used in the training process.

    Args:
        opt (type): The options for the callbacks.
        now (type): The current time.
        logdir (type): The directory for logging.
        ckptdir (type): The directory for saving checkpoints.
        cfgdir (type): The directory for configuration files.
        config (type): The configuration settings.
        lightning_config (type): The configuration settings for the lightning module.
        trainer_opt (type): The options for the trainer.
        modelckpt_cfg (type): The configuration for model checkpointing.

    Returns:
        type: The configuration for the callbacks.
    """
    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "src.util.callbacks.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "image_logger": {
            "target": "src.util.callbacks.ImageLogger",
            "params": {
                "batch_frequency": 750,
                "max_images": 4,
                "clamp": True
            }
        },
        "learning_rate_logger": {
            "target": "src.util.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        },
        "cuda_callback": {
            "target": "src.util.callbacks.CUDACallback"
        },
    }
    if version.parse(pl.__version__) >= version.parse('1.4.0'):
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        logging.warning(
            'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint':
                {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                    'params': {
                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        'save_top_k': -1,
                        'every_n_train_steps': 10001,
                        'save_weights_only': False
                    }
                    }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
        callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
    elif 'ignore_keys_callback' in callbacks_cfg:
        del callbacks_cfg['ignore_keys_callback']
        
    return callbacks_cfg

def get_data(config):
    """
    Get the data for training.

    Args:
        config (object): The configuration object.

    Returns:
        object: The prepared data object.
    """
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    logging.info("#### Data #####")
    for k in data.datasets:
        logging.info(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    return data

def configure_learning_rate(config, model, lightning_config, cpu, opt):
    """
    Configures the learning rate for the model based on the provided configuration.

    Args:
        config: The configuration object.
        model: The model object.
        lightning_config: The lightning configuration object.
        cpu: A boolean indicating whether to use CPU or not.
        opt: The optimization object.

    Returns:
        The updated model object with the learning rate configured.
    """  
    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        ngpu = lightning_config.trainer.devices # len(lightning_config.trainer.gpus.strip(",").split(','))
    else:
        ngpu = 1
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    logging.info(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        logging.info(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        logging.info("++++ NOT USING LR SCALING ++++")
        logging.info(f"Setting learning rate to {model.learning_rate:.2e}")
        
    return model

# from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning import LightningDistributedModule
# class CustomDDPPlugin(DDPPlugin):
#     def configure_ddp(self):
#         self.pre_configure_ddp()
#         self._model = self._setup_model(LightningDistributedModule(self.model))
#         self._register_ddp_hooks()
#         self._model._set_static_graph() # THIS IS THE MAGIC LINE

def main():
    """
    Main function for training the autoencoder model.

    This function performs the following steps:
    1. Sets up the necessary paths and configurations.
    2. Instantiates the model.
    3. Sets up the trainer and callbacks.
    4. Trains the model if specified.
    5. Tests the model if specified.
    """
        
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    opt, unknown = parse_args()
    set_cache_directories(opt)

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    
    opt, nowname = get_nowname(opt, now)
    logdir = os.path.join(opt.logdir, nowname)
    logging.basicConfig(level=getattr(logging, opt.logging_level))
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)
    log_opts(opt)
    try:
        config = merge_configs(opt, unknown)
        lightning_config = config.pop("lightning", OmegaConf.create())
        
        # merge trainer cli with config
        trainer_config, cpu = set_trainer_config(opt, lightning_config)
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # get logger configs
        logger_cfg = get_logger_cfgs(opt, logdir, nowname, lightning_config)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
        
        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        modelckpt_cfg = get_model_checkpoint_cfgs(ckptdir, model, lightning_config)
        logging.info(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # get callbacks
        callbacks_cfg = get_callbacks_cfgs(opt, now, logdir, ckptdir, cfgdir, config, lightning_config, trainer_opt, modelckpt_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir

        # data
        data = get_data(config)

        # configure learning rate
        model = configure_learning_rate(config, model, lightning_config, cpu, opt)

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                logging.info("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if "test" in data.datasets:
            if not opt.no_test and not trainer.interrupted:
                trainer.test(model, data)
    
    except Exception as e:
        logging.error(e, exc_info=True) 
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            logging.info(f"{trainer.profiler.summary()}")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    mp.set_start_method('spawn')
    main()