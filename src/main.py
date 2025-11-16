"""
Main entry point for the BigEarthNet v2.0 analysis pipeline.
"""

import argparse
from ben_v2.main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BigEarthNetv2.0 segmentation pipeline or benchmark.')
    parser.add_argument('--tile_folder', type=str, help='Path to the Sentinel-2 tile folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder.')
    parser.add_argument('--benchmark', action='store_true', help='Run in benchmark mode.')
    parser.add_argument('--input_dir', type=str, help='Path to the directory of tiles for benchmarking.')

    args = parser.parse_args()
    print(args)

    if args.benchmark:
        if not args.input_dir:
            print("❌ Error: --input_dir is required for benchmark mode.")
        else:
            from ben_v2.benchmark import benchmark
            benchmark(args.input_dir, args.output_folder)
    elif args.tile_folder:
        main(args.tile_folder, args.output_folder)
    else:
        print("❌ Error: Either --tile_folder or --benchmark with --input_dir must be specified.")
