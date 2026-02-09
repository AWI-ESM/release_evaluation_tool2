"""
Config Loader - Dynamically loads configuration from a specified config file.

Config is resolved in this order:
  1. Command-line argument:  python part5_sea_ice_thickness.py ../configs/HR_tuning.py
  2. Environment variable:   export REVAL_CONFIG=configs/HR_tuning.py

Usage in scripts:
    from config_loader import *
"""

import os
import sys
import importlib.util

# Resolve config path: CLI argument > environment variable
config_path = None

# Check for command-line argument (first .py arg that looks like a config file)
for arg in sys.argv[1:]:
    if arg.endswith('.py') and os.path.exists(arg):
        config_path = os.path.abspath(arg)
        break

# Fall back to environment variable
if not config_path:
    config_path = os.environ.get('REVAL_CONFIG')

if not config_path:
    print("ERROR: No config specified.")
    print("  Usage:  python script.py <path/to/config.py>")
    print("  Or set: export REVAL_CONFIG=<path/to/config.py>")
    sys.exit(1)

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
