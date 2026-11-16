# Multi-Class Semantic Segmentation on CamVid

**Comparative Study of SegFormer-B2 and DeepLabV3+ with Loss Optimization and Agent-Based Experimentation**

## Overview

This project explores multi-class semantic segmentation for urban street scenes using the **CamVid (Cambridge-Driving Labeled Video Database)** dataset.

The objective is to train and evaluate modern deep-learning segmentation architectures, investigate the effect of different loss functions, and explore automated loss-weight optimization through an agent-based experimentation strategy.

Two architectures are studied:

- **SegFormer-B2** – A hierarchical vision transformer-based segmentation model.
- **DeepLabV3+** – A convolutional neural network architecture using atrous spatial pyramid pooling and a decoder for multi-scale feature extraction.

The project also investigates traditional baselines, loss-function comparisons, test-time augmentation, ensemble predictions, and per-class error analysis.

## Project Objectives

- Study modern semantic segmentation architectures and their underlying principles.
- Compare transformer-based and CNN-based segmentation models.
- Evaluate different loss functions for multi-class segmentation.
- Investigate class imbalance and segmentation boundary quality.
- Explore agent-assisted loss-weight optimization.
- Perform quantitative and qualitative evaluation.
- Analyze model performance across individual semantic classes.

## Dataset

**CamVid – Cambridge-Driving Labeled Video Database**

Dataset source: [CamVid Dataset on Kaggle](https://www.kaggle.com/datasets/carlolepelaars/camvid)

CamVid is an urban scene understanding dataset containing road-scene images with pixel-level semantic annotations. It includes categories such as roads, buildings, vehicles, pedestrians, sidewalks, trees, traffic lights, and other street-scene objects.

The dataset is organized into training, validation, and testing splits, with corresponding segmentation label directories.

### Dataset Structure

The expected local directory structure is:

```text
data/
└── CamVid/
    ├── train/
    ├── train_labels/
    ├── val/
    ├── val_labels/
    ├── test/
    ├── test_labels/
    └── class_dict.csv
```

**Note:** The dataset is not included in this GitHub repository because of its size. Download it separately from Kaggle and place it in the directory shown above.

## Models

### 1. SegFormer-B2

SegFormer is a transformer-based semantic segmentation architecture that combines hierarchical transformer encoders with a lightweight MLP decoder.

Key characteristics:

- Hierarchical multi-scale feature extraction.
- Efficient attention-based representation learning.
- Global contextual information.
- Lightweight segmentation decoder.

### 2. DeepLabV3+

DeepLabV3+ is a CNN-based semantic segmentation architecture that uses atrous convolutions and Atrous Spatial Pyramid Pooling (ASPP) to capture contextual information at multiple scales.

Key characteristics:

- Atrous convolution for enlarged receptive fields.
- ASPP for multi-scale context aggregation.
- Encoder-decoder architecture.
- Improved spatial detail through decoder refinement.

## Loss Functions

Multiple loss formulations are investigated to understand their impact on segmentation accuracy, class imbalance, and boundary quality.

The evaluated loss functions include:

- Cross-Entropy Loss
- Weighted Cross-Entropy Loss
- Focal Loss
- Dice Loss
- Tversky Loss
- Lovasz-Softmax Loss
- Composite Loss Functions

Different combinations and weight configurations are explored to study the trade-offs between dominant classes and less frequent semantic categories.

## Agent-Based Optimization

An automated experimentation strategy is used to investigate suitable loss-function weight configurations.

The optimization process evaluates candidate loss combinations using validation performance and compares their effectiveness against baseline loss configurations.

The objective is to identify configurations that improve segmentation performance without relying on test-set feedback for optimization.

## Experimental Evaluation

The models are evaluated using the following metrics:

| Metric | Description |
|---|---|
| Mean Intersection over Union (mIoU) | Average IoU across evaluated classes |
| Mean Dice Score | Average Dice coefficient across classes |
| Pixel Accuracy | Fraction of correctly classified pixels |
| Precision | Fraction of predicted class pixels that are correct |
| Recall | Fraction of ground-truth class pixels correctly identified |
| Boundary F1 | Boundary-level segmentation quality |

Additional experiments investigate:

- Multi-seed evaluation.
- Test-time horizontal flip augmentation.
- Model ensembling.
- Per-class performance analysis.
- Confusion matrix analysis.
- Qualitative prediction visualization.

## Results

The final ensemble evaluation on the CamVid test set produced the following results:

| Metric | Final Result |
|---|---:|
| mIoU | **0.5334** |
| mDice | **0.6587** |
| Mean Precision | **0.7657** |
| Mean Recall | **0.6340** |
| Pixel Accuracy | **0.9126** |
| Boundary F1 | **0.6763** |
| Evaluated Classes | **29** |

### Additional SegFormer-B2 Results

| Configuration | Test mIoU |
|---|---:|
| SegFormer-B2 (three-seed mean) | 0.5274 |
| SegFormer-B2 + Horizontal Flip TTA | 0.5365 |
| Final Ensemble + TTA + Per-Class Scaling | 0.5334 |

The results demonstrate the importance of evaluating segmentation models beyond overall pixel accuracy. Per-class metrics and confusion analysis help identify weaknesses involving rare classes, small objects, and visually similar categories.

## Qualitative Analysis

Qualitative evaluation is performed by comparing:

1. Original input images.
2. Ground-truth segmentation masks.
3. Model-predicted segmentation masks.

The visualizations help examine object boundaries, class confusion, small-object detection, and segmentation consistency across different urban scenes.

## Repository Structure

```text
ML-PROJECT/
│
├── data/
│   └── CamVid/
│       └── class_dict.csv
│
├── segproject/
│   ├── src/
│   ├── configs/
│   ├── results/
│   │   └── eval/
│   └── ...
│
├── Report-2401AI19.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

The exact internal structure of `segproject/` may vary depending on the experiment and implementation.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/CamVid-Semantic-Segmentation.git
cd CamVid-Semantic-Segmentation
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the CamVid dataset from Kaggle and place it inside `data/CamVid/` following the directory structure described above.

## Reproducing Experiments

The project includes training, evaluation, and experimentation code under the `segproject/` directory.

To reproduce the experiments:

1. Set up the Python environment and install dependencies.
2. Download and configure the CamVid dataset.
3. Review the experiment configuration files.
4. Run the corresponding training scripts for SegFormer-B2 and DeepLabV3+.
5. Execute the evaluation scripts to generate segmentation metrics.
6. Review the saved predictions, confusion matrices, and experiment results.

Refer to the source files and configuration files for the exact execution commands and experiment settings.

## Report

The detailed project report is available in:

**`Report-2401AI19.pdf`**

It covers the literature review, theoretical foundations, model architectures, loss functions, experimental methodology, optimization strategy, quantitative results, ablation studies, and qualitative analysis.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- SegFormer
- DeepLabV3+
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Optuna / automated optimization tools

## Author

**Nirmit Chaurasia**  
B.Tech – Computer Science and Engineering (AI & Data Sciences)  
Indian Institute of Technology Patna

---

*An academic project on multi-class semantic segmentation, model comparison, and automated loss-function optimization using the CamVid urban scene dataset.*