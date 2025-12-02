"""
Benchmarking tool for the BigEarthNet v2.0 analysis pipeline.
"""
import time
import argparse
from pathlib import Path
import pandas as pd
import psutil
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf, DictConfig

from .process import main_hydra
from .utils import get_device
from .generate_viewer import generate_viewer

def get_folder_size(folder_path: Path) -> float:
    """Calculate the total size of a folder in megabytes."""
    total_size = sum(f.stat().st_size for f in folder_path.glob('**/*') if f.is_file())
    return total_size / (1024 * 1024)

def monitor_current_process(interval=0.1):
    """Monitor CPU and memory usage of the current process."""
    process = psutil.Process()
    cpu_percents = []
    mem_rss = []

    # Start monitoring
    cpu_percents.append(process.cpu_percent(interval=None)) # non-blocking first call
    mem_rss.append(process.memory_info().rss)

    return {
        "initial_cpu_percent": cpu_percents[0],
        "initial_memory_mb": mem_rss[0] / (1024 * 1024),
    }

def benchmark(input_dir: str, output_dir: str):
    """
    Run the pipeline on all tiles in a directory and generate a performance report.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"❌ Error: Input directory '{input_dir}' not found.")
        return

    output_path.mkdir(exist_ok=True)

    results = []

    tile_folders = [d for d in input_path.iterdir() if d.is_dir()]

    print(f"--- Starting Benchmark ---")
    print(f"Found {len(tile_folders)} tiles to process.")

    # Initialize Hydra
    # config_path is relative to this file: src/eo_core/benchmark.py -> ../../configs
    rel_config_path = "../../configs"

    # Clear any existing GlobalHydra instance to avoid conflicts
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    except:
        pass

    with hydra.initialize(version_base=None, config_path=rel_config_path):
        # Load base config to initialize model
        # We provide dummy paths just to satisfy the mandatory "???" fields in config.yaml
        print("Loading configuration and model...")
        cfg_template = hydra.compose(config_name="config", overrides=["input_path=/tmp", "output_path=/tmp"])

        # Instantiate model once using Hydra
        model = hydra.utils.instantiate(cfg_template.model)
        model.to(get_device()).eval()
        print(f"Model loaded: {cfg_template.model.pretrained_model_name_or_path}")

        for i, tile_folder in enumerate(tile_folders):
            print(f"\n--- Processing tile {i+1}/{len(tile_folders)}: {tile_folder.name} ---")

            tile_output_path = output_path # main_hydra handles creating the subdir

            # Monitor resource usage before processing
            process_start_metrics = monitor_current_process()

            start_time = time.time()

            # Create specific config for this tile
            # We copy the template and update the paths
            cfg = cfg_template.copy()
            cfg.input_path = str(tile_folder)
            cfg.output_path = str(output_path)

            # Run the pipeline with the pre-loaded model
            main_hydra(cfg, model=model)

            end_time = time.time()

            # Monitor resource usage after processing
            process_end_metrics = monitor_current_process()

            duration = end_time - start_time

            input_size_mb = get_folder_size(tile_folder)
            tile_out_dir = output_path / tile_folder.name
            output_size_mb = get_folder_size(tile_out_dir) if tile_out_dir.exists() else 0
            file_count = len(list(tile_folder.glob('**/*')))

            result = {
                "tile_name": tile_folder.name,
                "processing_time_seconds": duration,
                "input_size_mb": input_size_mb,
                "output_size_mb": output_size_mb,
                "file_count": file_count,
                "cpu_at_start": process_start_metrics["initial_cpu_percent"],
                "memory_mb_at_start": process_start_metrics["initial_memory_mb"],
                "cpu_at_end": process_end_metrics["initial_cpu_percent"],
                "memory_mb_at_end": process_end_metrics["initial_memory_mb"],
            }
            results.append(result)

            print(f"--- Finished in {duration:.2f} seconds ---")

    # Generate report
    report_path = output_path / "benchmark_report.csv"
    df = pd.DataFrame(results)
    df.to_csv(report_path, index=False)

    print(f"\n--- Benchmark Complete ---")
    print(f"Report saved to {report_path}")

    # Generate the HTML viewer
    generate_viewer(str(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BigEarthNetv2.0 benchmark.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing Sentinel-2 tile folders.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory for benchmark results.')
    args = parser.parse_args()

    benchmark(args.input_dir, args.output_dir)
