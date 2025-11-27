# GeoSpatial Inference Pipeline (GSIP)

This project provides a scalable, multi-modal deep learning pipeline for robust and efficient inference on gigapixel-scale Earth Observation (EO) data, focusing on tasks like land cover classification and segmentation.

## 📚 Documentation

Detailed documentation is organized in the `docs/` directory:

| Document | Description |
| :--- | :--- |
| [**Usage Guide**](docs/USAGE.md) | Detailed instructions on how to run models, configure inputs, and interpret outputs. |
| [**Technical Reference**](docs/TECHNICAL_REFERENCE.md) | Deep dive into the **ERF-Aware Tiling**, **Memory Auto-Configuration**, and mathematical formulas. |
| [**API Reference**](docs/API_REFERENCE.md) | Function-level documentation for the codebase (Engine, Adapters, Data Loading). |
| [**Development Guide**](docs/DEVELOPMENT.md) | Instructions for contributors: adding new models, creating adapters, and modifying the core. |
| [**Project Structure**](docs/PROJECT_STRUCTURE.md) | An overview of the file tree and the purpose of each directory. |

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r src/requirements.txt
pip install -e .
```

### 2. Run a Model

**Sentinel-2 Classification (ResNet-50):**
```bash
python src/main.py model=resnet_s2 input_path=/path/to/tile output_path=./out
```

**Flood Segmentation (Prithvi-100M):**
```bash
python src/main.py model=prithvi_segmentation input_path=/path/to/tile output_path=./out
```

## Key Features

-   **Multi-Model Support:** Easily switch between ResNet, ConvNeXt, and Segmentation models via Config Adapters.
-   **Scalable Tiling:** Processes massive images using an artifact-free Overlap-Tile strategy.
-   **Smart Memory Management:** Automatically calculates safe processing chunk sizes based on available RAM.
-   **Multi-Modal:** Supports fusion of Optical (S2) and Radar (S1) data.
-   **Uncertainty Quantification:** Outputs Entropy and Confidence maps.