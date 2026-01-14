U2Net with Mamba Integration: Training Code
This repository contains the training code for a modified U2Net model integrated with Mamba layers for salient object detection tasks.

Overview
This project implements a hybrid architecture combining the U2Net (U-Net squared) model with Mamba (State Space Model) layers to enhance salient object detection performance. The Mamba layers are integrated to capture long-range dependencies more efficiently while maintaining the U-Net's strong local feature extraction capabilities.
Requirements<br>
Python 3.10<br>
PyTorch<br>
torchvision<br>
torchmetrics<br>
numpy<br>
mamba-ssm<br>

Install dependencies via pip:
bash
pip install torch torchvision torchmetrics numpy mamba-ssm

Dataset Preparation
The code is configured to work with the DUTS dataset by default. Follow these steps to prepare your dataset:
Download the DUTS dataset (DUTS-TR training set)
Organize the dataset in the following structure:

plaintext
train_data/<br>
└── DUTS-TR/<br>
    └── DUTS-TR/<br>
        ├── im_aug/    # Training images<br>
        └── gt_aug/    # Corresponding ground truth masks<br>
        
Model Architecture
The key modification from the original U2Net is the integration of Mamba layers through the MambaLayer class, which:
Uses LayerNorm for input normalization
Incorporates Mamba SSM (State Space Model) for sequence modeling
Handles 2D spatial data by flattening spatial dimensions into sequence tokens

Training Configuration
Model: U2Net with Mamba integration
Loss Function: Multi-scale BCE (Binary Cross-Entropy) loss
Optimizer: Adam with lr=0.0001
Metrics: Jaccard Index (IoU)
Input Size: 320x320 (rescaled) → 288x288 (random crop)
Batch Size: 12 (training)

Training Instructions
Clone this repository
Prepare your dataset as described above
Adjust configuration parameters in u2mamba_train.py if needed:
epoch_num: Total training epochs
batch_size_train: Training batch size
save_frq: Model save frequency (iterations)
data_dir: Path to training data
Run the training script:
bash
python u2mamba_train.py

Output
Trained models will be saved in the saved_models/u2net/ directory
Each saved model filename includes iteration number and training metrics
Training progress is printed periodically with loss values and IoU scores

Notes
The code uses CUDA by default (set via os.environ["CUDA_VISIBLE_DEVICES"] = "1"). Modify this to use different GPUs or CPU.
The training process includes multiple output scales (d0-d6) from the U2Net architecture, with loss computed at each scale.
Jaccard Index (IoU) is calculated on the primary output (d0) for monitoring training progress.

Acknowledgments
Based on the U2Net architecture (Qin et al., 2020)
Incorporates Mamba from the mamba-ssm library
Designed for use with the DUTS dataset (Li et al., 2017)
