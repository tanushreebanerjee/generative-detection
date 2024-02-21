# src/util.py
import os
import sys
import argparse
import logging

def log_args(args):
    """Log arguments."""
    logging.info("Arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

def add_submodules_to_sys_path():
    """Add submodules to sys.path."""
    submodule_dir = "submodules"
    submodule_names = os.listdir(submodule_dir)
    for submodule_name in submodule_names:
        submodule_path = os.path.join(submodule_dir, submodule_name)
        sys.path.append(submodule_path)
        
def set_env_variables():
    """Set environment variables for data, model, output, and tensorboard directories."""
    os.environ["TRANSFORMERS_CACHE"] = ".cache/transformers"
    os.environ["TORCH_HOME"] = ".cache/torch"

    os.environ["DATA_DIR"] = "data"

    os.environ["MODEL_DIR"] = "models"

    os.environ["OUTPUT_DIR"] = "results"

    os.environ["TENSORBOARD_DIR"] = ".logs/tensorboard"
    
def create_dirs():
    """Create directories for data, model, and output."""
    os.makedirs(os.environ["DATA_DIR"], exist_ok=True)
    os.makedirs(os.environ["MODEL_DIR"], exist_ok=True)
    os.makedirs(os.environ["OUTPUT_DIR"], exist_ok=True)
