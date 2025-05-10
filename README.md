# TMMTrack

# Interaction-aware Multi-object Tracking with Trajectory-based Motion Modeling

## Abstract

TMMTrack is an online multi-object tracking algorithm tailored for nonlinear motion scenarios. To address the limitations of the constant velocity assumption in existing methods, TMMTrack introduces a trajectory-based motion modeling strategy within the tracking-by-detection framework. It leverages a long-term memory buffer and learnable motion tokens to capture spatio-temporal interactions between trajectories, and employs a self-attention-based adaptive aggregation mechanism to model both individual and group motion patterns. Experiments on DanceTrack and SportsMOT benchmarks show that TMMTrack achieves approximately 4% improvement in key metrics such as HOTA and IDF1 over existing approaches.

## 📈 Performance Benchmark

| Dataset                                       | MOTA ↑ | IDF1 ↑ | HOTA ↑ | DetA ↑ | AssA ↑ |
| --------------------------------------------- | ------ | ------ | ------ | ------ | ------ |
| DanceTrack                                    | 91.4%  | 57.1%  | 55.8%  | 81.0%  | 38.6%  |
| SportsMOT                                     | 94.8%  | 72.5%  | 72.4%  | 87.0%  | 60.2%  |
| *(Results measured on a single RTX 3060 GPU)* |        |        |        |        |        |

## 🚀 Features

- **Temporal Interaction Module (TIM):** Captures long-term dependencies within a single object's trajectory using multi-layer Transformers.
- **Spatial-Temporal Interaction Module (STIM):** Models cross-object interactions over time to improve coherence in crowded or complex scenes.
- **Configurable via YAML:** Easily switch datasets, hyperparameters, and model architectures in `configs/`.
- **Benchmark-ready:** Out-of-the-box support for **DanceTrack** and **SportsMOT** datasets.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Junkman996/TMMTrack.git
cd TMMTrack

# Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📂 Project Structure（only TMMTrack）

```bash
TMMTrack
	├── configs/              # YAML config files
	├── data/                 # Dataset loaders & augmentations
	├── models/               # TIM, STIM, motion predictor, detector
	├── losses/               # Loss functions (SmoothL1 + DIoU)
	├── trainers/             # Training & validation loops
	├── inference/            # Online inference & visualization
	├── utils/                # Logging, distributed training, checkpoints
	├── scripts/              # Entry-point scripts (train, eval, infer)
	└── README.md             # This file
```

## 🎬 Quick Start

1. **Prepare Data**
   - Download the **DanceTrack** or **SportsMOT** dataset.
   - Convert dataset annotations to YAML format (see `configs/*.yaml` for examples).
   - Update paths in `configs/dance_track.yaml` or `configs/sports_mot.yaml`.

2. **Training**

```bash
bash scripts/train.sh configs/dance_track.yaml
```

3. **Evaluation**

```bash
bash scripts/eval.sh configs/dance_track.yaml work_dir/best.pth
```

4. **Inference**

```bash
bash scripts/infer.sh configs/dance_track.yaml work_dir/best.pth /path/to/video.mp4
```

