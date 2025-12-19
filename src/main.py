import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import logging

# Set PyTorch Memory Config to reduce fragmentation
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Deprecated
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Adjust path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eo_core.process import main_hydra

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    main_hydra(cfg)

if __name__ == "__main__":
    main()