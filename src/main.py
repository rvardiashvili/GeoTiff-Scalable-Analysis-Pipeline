import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys

# Adjust path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ben_v2.process import main_hydra

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    main_hydra(cfg)

if __name__ == "__main__":
    main()