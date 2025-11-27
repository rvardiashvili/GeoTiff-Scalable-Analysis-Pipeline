# Usage Guide

This guide explains how to run the GeoSpatial Inference Pipeline (GSIP) using the available model configurations.

## Basic Command Structure

The general command to run the pipeline is:

```bash
python src/main.py model=<model_config_name> input_path=<path_to_tile> output_path=<path_to_output_dir>
```

## Available Models

### 1. ResNet-50 (Sentinel-2 Only)
**Config:** `resnet_s2`
**Description:** Standard classification model using 10 Sentinel-2 bands.
**Usage:**
```bash
python src/main.py model=resnet_s2 input_path=/path/to/S2_tile output_path=./results
```

### 2. ResNet-50 (Sentinel-1 & Sentinel-2)
**Config:** `resnet_all`
**Description:** Multi-modal classification using 12 bands (2 S1 + 10 S2). **Requires matching Sentinel-1 data** to be present or configured in the data loader.
**Usage:**
```bash
python src/main.py model=resnet_all input_path=/path/to/S2_tile output_path=./results
```

### 3. ConvNeXt V2 (Sentinel-2)
**Config:** `convnext_s2`
**Description:** Modern ConvNeXt architecture for classification (10 bands).
**Usage:**
```bash
python src/main.py model=convnext_s2 input_path=/path/to/S2_tile output_path=./results
```

### 4. Prithvi-100M Flood Segmentation
**Config:** `prithvi_segmentation`
**Description:** A geospatial foundation model fine-tuned for flood segmentation.
**Note:** This model outputs a segmentation map. The current pipeline will save the raw outputs, but the visualization (Class Map) might need interpretation as "Background" vs "Flood".
**Usage:**
```bash
python src/main.py model=prithvi_segmentation input_path=/path/to/S2_tile output_path=./results
```

## Benchmarking Mode

To run performance benchmarks across multiple tiles:

```bash
python src/main.py --benchmark --input_dir /path/to/all_tiles/ --output_folder ./benchmark_results
```
*Note: Ensure you configure the default model in `configs/config.yaml` or pass `model=...` override if benchmarking a specific model.*

## Understanding Outputs

Check the `out/` directory for results:
*   `*_class.tif`: The final class/segmentation map.
*   `*_maxprob.tif`: Confidence map.
*   `preview.png`: A quick look RGB image of the classification.
