import spacy
from spacy.cli.init_config import init_config, save_config
from pathlib import Path


# Paths
config_path = Path("../config.cfg")
if config_path.exists():
    config_path.unlink()

# Create the config
cfg = init_config(lang = "gd", pipeline=["tagger"])

# save
save_config(cfg, config_path)
print(f"Config file created at {config_path}")