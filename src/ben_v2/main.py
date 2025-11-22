"""
Main entry point for the scalable BigEarthNet analysis pipeline.

This script orchestrates the large-scale segmentation process defined in 
segmentator.py for a single, user-specified Sentinel-2 tile folder.
"""
import time
from pathlib import Path
import argparse

from . import config
from .process import main as segmentator_main
from . import extra_generators
from .generate_viewer import generate_single_node_viewer

def main(TILE_FOLDER: str, OUTPUT_FOLDER: str, model=None):
    """
    Orchestrates the scalable analysis of a single input scene.
    """
    print(f"\n--- Starting Scalable Geo Segmentation ({config.MODEL_NAME}) ---")

    start_time = time.time()
    
    tile_path = Path(TILE_FOLDER)
    output_path = Path(OUTPUT_FOLDER)

    if not tile_path.is_dir():
        print(f"❌ Error: Input directory '{TILE_FOLDER}' not found or is not a directory.")
        print("Please ensure the TILE_FOLDER variable points to the correct scene directory.")
        return

    # Direct call to the scalable segmentation pipeline for the single tile
    print(f"\n=======================================================")
    print(f"  STARTING TILE: {tile_path.name}")
    print(f"=======================================================")
    
    # The segmentator_main function handles all I/O, chunking, inference, and saving.
    segmentator_main(tile_folder=str(tile_path), output_directory=str(output_path), extra_data_generators=[extra_generators.calculate_ndvi])

    # Generate viewer for the single tile
    generate_single_node_viewer(tile_path.name, str(output_path))

    total_time = time.time() - start_time

    print("\n--- Pipeline Complete ---")
    print(f"Total Wall Time: {total_time:.2f} seconds")

