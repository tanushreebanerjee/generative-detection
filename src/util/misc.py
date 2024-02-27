# src/util/misc.py
import os
import sys
import logging
import json

def log_opts(opts):
    """Log the options."""
    logging.info("Options: %s", json.dumps(vars(opts), indent=2))

def set_submodule_paths(submodule_dir):
    """Set the paths for the submodules."""
    for submodule in os.listdir(submodule_dir):
        submodule_path = os.path.join(submodule_dir, submodule)
        if os.path.isdir(submodule_path):
            sys.path.append(submodule_path)
     
def set_cache_directories(opts):
    """Set environment variables for cache directories."""
    os.environ["TRANSFORMERS_CACHE"] = opts.transformers_cache
    os.environ["TORCH_HOME"] = opts.torch_home