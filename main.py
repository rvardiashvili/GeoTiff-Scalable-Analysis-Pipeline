"""
Main entry point for the BigEarthNet analysis pipeline.

This script initializes the model, discovers scenes, and orchestrates the
analysis pipeline for each scene using the PyTorch DataLoader for efficient I/O.
Finally, it generates an HTML report summarizing the results.
"""
import time
from pathlib import Path

import torch
import numpy as np

import config
# Import BigEarthNetv2_0_ImageClassifier from utils to safely handle potential ImportError
from utils import BigEarthNetv2_0_ImageClassifier
from pipeline import SceneAnalyzer
from reporting import generate_html_report, generate_html_report_data

def analyze_scenes(root_dir: str, model: torch.nn.Module, max_scenes: int, max_patches: int):
    """Manages the analysis for all scenes found in the root directory."""
    scene_paths = [p for p in Path(root_dir).iterdir() if p.is_dir()]
    if max_scenes > 0:
        scene_paths = scene_paths[:max_scenes]

    final_scenes_results, total_patches, gpu_times = [], 0, []

    for scene_path in scene_paths:
        print(f"\n🔹 Analyzing scene: {scene_path.name}")
        analyzer = SceneAnalyzer(scene_path, model, max_patches)

        # FIX: Call the new, synchronous run() method, which executes the DataLoader pipeline.
        # It returns (scene_gpu_times: List[float], scene_patches_processed: int)
        scene_gpu_times, scene_patches_processed = analyzer.run()
        
        # Aggregate results
        scene_results = generate_html_report_data(
            {"name": scene_path.name, "patches": analyzer.raw_results}
        )
        final_scenes_results.append(scene_results)
        
        # Accumulate metrics
        total_patches += scene_patches_processed
        gpu_times.extend(scene_gpu_times)

    # Return the collected data: (scene_results, total_patches, all_gpu_batch_times)
    return final_scenes_results, total_patches, gpu_times

def main():
    """Initializes model and runs the full analysis pipeline."""
    # --- Configuration/Environment Setup ---
    # This path must point to your BigEarthNet-S2 root directory
    ROOT_DIR = "/home/rati/bsc_thesis/BigEarthNetv2.0/ben_subset/BigEarthNet-S2"
    # -------------------------------------------------------------------

    if not Path(ROOT_DIR).exists():
        print(f"❌ Error: Root directory not found at {ROOT_DIR}. Cannot run analysis.")
        return

    print("--- BigEarthNet Analysis Pipeline ---")
    print(f"Loading model '{config.MODEL_NAME}'...")
    
    # Check if BigEarthNetv2_0_ImageClassifier is the placeholder or the actual class
    if isinstance(BigEarthNetv2_0_ImageClassifier, type):
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(config.REPO_ID).to(config.DEVICE)
        model.eval()
        print("Model loaded successfully.")
    else:
        print("Model class is placeholder. Analysis skipped.")
        return

    start_time = time.time()

    scenes_results, total_patches, gpu_times = analyze_scenes(
        ROOT_DIR, model, max_scenes=config.MAX_SCENES, max_patches=config.MAX_PATCHES
    )

    total_time = time.time() - start_time
    # Note: GPU times are returned as lists of batch times.
    avg_gpu_time = np.mean(gpu_times) if gpu_times else 0.0

    print("\n--- Analysis Complete ---")
    print(f"Total Patches Processed: {total_patches}")
    print(f"Total Processing Time: {total_time:.2f} seconds")
    print(f"Average GPU Batch Time ({config.GPU_BATCH_SIZE} patches): {avg_gpu_time:.4f} seconds")

    if scenes_results:
        # Note: The generate_html_report function takes 4 arguments now.
        generate_html_report(
            scenes_results,          
            total_time,              
            avg_gpu_time,            
            "report.html"            
        )
        print("Report generated successfully.")
    else:
        print("No scenes were processed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n--- An unexpected error occurred in main execution ---")
        print(f"Error: {type(e).__name__}: {e}")
