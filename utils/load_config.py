import json
import os
from pathlib import Path

def load_config():
    """
    Load configuration from config/config.json relative to the project root.
    
    Returns:
        dict: Configuration dictionary
    """
    # Get the project root directory (parent of utils)
    utils_dir = Path(__file__).resolve().parent
    project_root = utils_dir.parent
    config_path = project_root / 'config' / 'config.json'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config