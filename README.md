# ENT Medical Image Retrieval System

This project implements a deep learning-based medical image retrieval system for Ear, Nose, and Throat (ENT) images using vector field transformations and CLIP embeddings.

## Overview

The system uses a novel approach combining CLIP (Contrastive Language-Image Pre-training) embeddings with learned vector field transformations to improve medical image retrieval accuracy. The model is trained to enhance similarity matching for ENT anatomical structures.

## Dataset Classes

The system classifies and retrieves images across 7 ENT anatomical categories:

- **ear-left**: Left ear images
- **ear-right**: Right ear images  
- **nose-left**: Left nostril images
- **nose-right**: Right nostril images
- **throat**: Throat/pharynx images
- **vc-open**: Open vocal cords
- **vc-closed**: Closed vocal cords

## Project Structure

```
track_2/
├── train/              # Training notebooks
│   └── train.ipynb     # Model training pipeline
├── inference/          # Inference and prediction
│   └── predict.ipynb   # Image retrieval inference
├── evaluation/         # Model evaluation
│   ├── clip-score.ipynb
│   └── flow_matching_step10_step0_score.ipynb
├── model/              # Trained model weights
│   └── vf_model.pth    # Vector field model weights
└── sample/             # Sample dataset organized by class
    ├── ear-left/
    ├── ear-right/
    ├── nose-left/
    ├── nose-right/
    ├── throat/
    ├── vc-open/
    └── vc-closed/
```

## Key Components

### 1. Vector Field Model
- **GaussianFourierProjection**: Time encoding using sinusoidal embeddings
- **VectorField**: Multi-head neural network with residual connections
- **Euler Integration**: RK4 integration for transforming embeddings

### 2. Training Strategy
- **Triplet Learning**: Uses anchor-positive-negative triplets
- **MultiSimilarityLoss**: Advanced metric learning loss function
- **Hard Negative Mining**: Prioritizes anatomically incompatible classes

### 3. Anatomical Constraints
The system incorporates medical knowledge through negative class relationships:
- Left vs Right laterality (ear-left ↔ ear-right, nose-left ↔ nose-right)
- Vocal cord states (vc-open ↔ vc-closed, throat)

## Usage

### Training
```bash
# Open train/train.ipynb in Jupyter
# The notebook will:
# 1. Download and prepare the dataset
# 2. Generate CLIP embeddings
# 3. Create triplet pairs with hard negatives
# 4. Train the vector field model
# 5. Save model weights to vf_model.pth
```

### Inference
```bash
# Open inference/predict.ipynb in Jupyter
# The notebook will:
# 1. Load the trained model
# 2. Process query images
# 3. Retrieve top-k similar images
# 4. Display results with similarity scores
```

## Model Architecture

### Vector Field Network
```
Input: [B, 512] CLIP embeddings + time t
├── LayerNorm
├── GaussianFourierProjection(t) → [B, 32]
├── Concat → [B, 544]
├── 4x Multi-head branches:
│   ├── Linear(544 → 256)
│   ├── LayerNorm + SiLU + Dropout
│   └── Linear(256 → 512)
├── Average heads + Residual connection
└── Output: [B, 512] transformed embeddings
```

### Integration Process
- Uses 4th-order Runge-Kutta (RK4) integration
- Default: 10 integration steps
- Transforms embeddings along learned vector field

## Performance

The model achieves **95.4% Recall@1** on the test set, demonstrating strong performance in retrieving anatomically similar images.

## Requirements

```python
# Core dependencies
torch
torchvision
clip-by-openai
pytorch-metric-learning
scikit-learn
PIL
numpy
matplotlib
tqdm
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install torch torchvision
   pip install git+https://github.com/openai/CLIP.git
   pip install pytorch-metric-learning scikit-learn pillow matplotlib tqdm
   ```
3. Download sample data and model weights (handled automatically in notebooks)

## Medical Applications

This system can be used for:
- **Clinical Decision Support**: Finding similar cases for diagnosis
- **Medical Education**: Retrieving examples for teaching
- **Research**: Analyzing anatomical variations and patterns
- **Quality Assurance**: Identifying imaging artifacts or anomalies

## Technical Details

- **Base Model**: CLIP ViT-B/32 for initial embeddings
- **Embedding Dimension**: 512
- **Training**: 70/15/15 train/val/test split
- **Loss Function**: MultiSimilarityLoss with triplet mining
- **Optimization**: AdamW with learning rate scheduling
- **Early Stopping**: Patience-based with validation monitoring

## Evaluation Metrics

- **Recall@1**: Primary metric for retrieval accuracy
- **CLIP Score**: Semantic similarity evaluation
- **Flow Matching Score**: Vector field transformation quality

## License

This project is for research and educational purposes in medical imaging.