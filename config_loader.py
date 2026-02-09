"""
Config Loader - Dynamically loads configuration based on REVAL_CONFIG environment variable

This module allows reval.py to specify which config file to use via the REVAL_CONFIG
environment variable. If not set, it uses a default config.

Usage in scripts:
    from config_loader import *
    
Instead of:
    from config import *
"""

import os
import sys
import importlib.util

# Check if a specific config file is specified via environment variable
config_path = os.environ.get('REVAL_CONFIG')

if not config_path:
    # Use default config if not specified
    # Look for configs/AWI-CM3-v3.3.py as default, or first available .py file
    default_config = os.path.join(os.path.dirname(__file__), 'configs', 'AWI-CM3-v3.3.py')
    if os.path.exists(default_config):
        config_path = default_config
        print(f"No REVAL_CONFIG specified, using default: {config_path}")
    else:
        # Find first available config file
        configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
        if os.path.exists(configs_dir):
            config_files = [f for f in os.listdir(configs_dir) if f.endswith('.py')]
            if config_files:
                config_path = os.path.join(configs_dir, sorted(config_files)[0])
                print(f"No REVAL_CONFIG specified, using: {config_path}")
            else:
                raise FileNotFoundError("No config files found in configs/ directory")
        else:
            raise FileNotFoundError("configs/ directory not found")

# Load the specified config file
if config_path:
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("dynamic_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config from {config_path}")
    
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Export all variables from the config module to this namespace
    for attr in dir(config_module):
        if not attr.startswith('_'):
            globals()[attr] = getattr(config_module, attr)
