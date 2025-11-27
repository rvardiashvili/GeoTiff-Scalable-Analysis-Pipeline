# Project Structure

This document outlines the directory layout and purpose of key files in the **GeoSpatial Inference Pipeline (GSIP)**.

```
/home/rati/bsc_thesis/BigEarthNetv2.0/main/
├── configs/                  # Hydra configuration files
│   ├── config.yaml           # Main entry point for configuration
│   ├── data_source/          # Data source settings (Sentinel-2, etc.)
│   ├── model/                # Model architectures and adapters
│   │   ├── convnext_s2.yaml  # Config for ConvNeXt (S2 only)
│   │   ├── prithvi_flood_segmentation.yaml # Config for Prithvi Flood Segmentation
│   │   ├── resnet_all.yaml   # Config for ResNet50 (S1 + S2)
│   │   ├── resnet_s2.yaml    # Config for ResNet50 (S2 only)
│   │   └── ...               # Other model configurations
│   └── pipeline/             # Inference pipeline parameters (tiling, batching)
│       └── inference_params.yaml
│
├── src/
│   ├── main.py               # CLI Entry Point
│   ├── requirements.txt      # Python dependencies
│   └── eo_core/              # Core Package Code (GeoSpatial Inference Pipeline logic)
│       ├── adapters/         # Model Adapters (Interface for various ML models)
│       ├── data.py           # Data I/O and preprocessing utilities
│       ├── inference_engine.py # Core logic for model inference
│       ├── memory_utils.py   # Utilities for dynamic memory management
│       ├── process.py        # Main orchestration of tiling and inference pipeline
│       ├── utils.py          # General utility functions
│       └── ...               # Other modules like benchmark, fusion, download, viewer
│
├── out/                      # Default Output Directory
├── docs/                     # Documentation Directory
│   ├── TECHNICAL_REFERENCE.md # Technical Reference Manual
│   ├── PROJECT_STRUCTURE.md  # This File
│   ├── API_REFERENCE.md      # Detailed Function Documentation
│   ├── USAGE.md              # Usage Guide
│   └── DEVELOPMENT.md        # Developer Guide
└── README.md                 # Project Overview
```

## Key Directories

*   **`configs/`**: Controls every aspect of the pipeline without changing code. Models, batch sizes, and file paths are defined here.
*   **`docs/`**: Contains all detailed documentation for the project.
*   **`src/eo_core/adapters/`**: The "plug-and-play" layer. If you want to add a new model type (e.g., a YOLO detector or a specific segmentation net), you write a new Adapter here without touching the core pipeline logic.
*   **`src/eo_core/`**: Contains the heavy lifting logic. `process.py` is the orchestrator, managing the multiprocessing and tiling logic described in `docs/TECHNICAL_REFERENCE.md`.
