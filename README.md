# Wafer Defect Classifier

## Description
This project uses a custom PyTorch Convolutional Neural Network (CNN) to classify surface defects on semiconductor wafers. 

To make testing easy and avoid any issues, the entire evaluation pipeline is packaged into a Docker container.

## Tech Stack
ML Framework: PyTorch, Torchvision

Metrics & Plots: Scikit-learn, Matplotlib, Seaborn

Deployment: Docker

## What it detects
The model classifies electron microscope images into 6 types of anomalies:
`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, and `scratches`.

## Model Results
Tested on a validation set of 360 images (60 per class).
* **Overall Accuracy:** 84%

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Crazing | 1.00 | 0.85 | 0.92 |
| Inclusion | 0.91 | 0.70 | 0.79 |
| Patches | 0.72 | 1.00 | 0.84 |
| Pitted Surface | 0.81 | 0.92 | 0.86 |
| Rolled-in Scale | 0.93 | 0.70 | 0.80 |
| Scratches | 0.78 | 0.87 | 0.82 |

## How to run it

**1. Get the Dataset**

Download the NEU Surface Defect Database from [Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database).
Extract it and place it in the root of the project so the path looks like this: `data/validation/crazing/...`

**2. Run the Docker Container**

Open your terminal in the project root and run this command to pull the pre-built image and test the model on your local data:
```bash
docker run -v "$(pwd)/data:/app/data" nazariifilin/defect-ai:latest 
```
Or if you want to build the Docker image yourself:
```bash
docker build -t defect-ai .
```

```bash
docker run -v "$(pwd)/data:/app/data" defect-ai
```
