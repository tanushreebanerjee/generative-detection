# src/util.py
import os
import sys
import logging

def log_args(args):
    """Log arguments."""
    logging.info("Arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

def add_submodules_to_sys_path():
    """Add submodules to sys.path."""
    submodule_dir = os.environ["SUBMODULE_DIR"]
    submodule_names = os.listdir(submodule_dir)
    for submodule_name in submodule_names:
        submodule_path = os.path.join(submodule_dir, submodule_name)
        sys.path.append(submodule_path)
        
def set_env_variables():
    """Set environment variables for data, model, output, and tensorboard directories."""
    os.environ["DATA_DIR"] = "data"
    os.environ["MODEL_DIR"] = "models"
    os.environ["OUTPUT_DIR"] = "results"
    os.environ["LOGGING_DIR"] = ".logs"
    os.environ["TENSORBOARD_DIR"] = f"{os.environ['LOGGING_DIR']}/tensorboard"
    os.environ["CONFIG_DIR"] = "configs"
    os.environ["SUBMODULE_DIR"] = "submodules"
    os.environ["CACHE_DIR"] = ".cache"
    
    os.environ["TRANSFORMERS_CACHE"] = f"{os.environ['CACHE_DIR']}/transformers"
    os.environ["TORCH_HOME"] = f"{os.environ['CACHE_DIR']}/torch"