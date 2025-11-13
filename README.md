# Barlow Twins on CIFAR-10

This repository contains a from-scratch implementation of the [Barlow Twins](https://arxiv.org/abs/2103.03230) self-supervised learning method, applied to the CIFAR-10 dataset. The project is designed to showcase modern deep learning and software development best practices.

The project is divided into two main stages:
1.  **Self-Supervised Pre-training**: A ResNet-18 encoder is trained on CIFAR-10 without labels using the Barlow Twins objective. This teaches the model to learn meaningful visual representations.
2.  **Downstream Linear Evaluation**: The pre-trained encoder is frozen, and a linear classifier is trained on top of its features to evaluate the quality of the learned representations on the CIFAR-10 classification task.

## Project Structure

```
barlowtwins/
├── .gitignore
├── pyproject.toml      # Project definition and dependencies
├── README.md           # This file
├── scripts/
│   ├── pretrain.py     # Script for self-supervised pre-training
│   └── evaluate.py     # Script for downstream linear evaluation
└── src/
    ├── data.py         # Data loading and augmentation logic
    ├── loss.py         # The Barlow Twins loss function
    └── model.py        # Model architectures (BarlowTwins, LinearClassifier)
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/barlowtwins.git
    cd barlowtwins
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the project in editable mode:**
    This command reads the `pyproject.toml` file and installs all necessary dependencies.
    ```bash
    pip install -e .[dev]
    ```

## Usage

The training process is a two-step procedure.

### Step 1: Run Self-Supervised Pre-training

This will train the ResNet-18 encoder using the Barlow Twins method and save the trained backbone weights to `checkpoints/barlow_twins_backbone.pth`.

```bash
python scripts/pretrain.py --epochs 100 --batch_size 256
```
*You can adjust the number of epochs and batch size as needed.*

### Step 2: Run Linear Evaluation

This script loads the frozen backbone from the previous step, trains a linear classifier on top of it, and reports the final classification accuracy on the CIFAR-10 test set.

```bash
python scripts/evaluate.py --epochs 50
```

## Results

After running the full pipeline, the expected accuracy demonstrates the effectiveness of the self-supervised representations.

*   **Pre-training Epochs:** 100
*   **Evaluation Epochs:** 50
*   **Final Test Accuracy:** [Add your result here, e.g., 85.12%]

This project serves as a practical demonstration of implementing a modern self-supervised learning paper and structuring a deep learning project in a clean, reproducible, and professional manner.
