# Garment Segmentation for 2D-to-3D Structure Inference

This project builds upon and extends an existing image segmentation repository, originally designed for garment part labeling. The fork improves data handling, training stability, and segmentation quality, with the goal of supporting 2D-to-3D reconstruction workflows in digital fashion and spatial modeling applications.

The segmentation task focuses on classifying pixels into four structural garment regions: **front**, **back**, **sleeves**, and **hood**. These regions serve as semantic primitives for 3D mesh generation, enabling structure-aware reconstruction from monocular images.

## Project Objective

The objective is to improve the segmentation pipeline by:
- Implementing a custom, flexible PyTorch dataloader
- Enhancing training with effective augmentation, normalization, and regularization
- Achieving better segmentation performance as measured by **mean IoU** and **validation loss**
- Providing a clean, extensible foundation for research into 2D-to-3D clothing reconstruction

## Dataset

The dataset consists of fashion product images and pixel-wise segmentation masks. While the label set includes more than four classes, only the following are used for this project:
- `Front`
- `Back`
- `Sleeves`
- `Hood`

All other labels are excluded or remapped to background.

## Methodology

### Dataloader
- Custom `torch.utils.data.Dataset` implementation
- Supports data augmentation: random crops, flips, scaling, normalization
- Dynamically remaps label masks to the target class set

### Training
- Loss function: cross-entropy with optional class balancing
- Optimizer: Adam with cosine annealing or step-based learning rate schedule
- Metrics tracked: training loss, validation loss, and mean IoU
- Logging per epoch including training time

### Evaluation
- Validation performed after each epoch
- Best models are selected based on validation performance (mean IoU)
- Designed to beat the following baseline:


## Installation

```bash
git clone https://github.com/yourusername/garment-segmentation-2d-to-3d.git
cd garment-segmentation-2d-to-3d
pip install -r requirements.txt
