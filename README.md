# Power System Stability Prediction: A Comparative Analysis of Baseline, ANN Feature Extraction, and Hybrid Models
This repository presents a machine learning approach to predict smart grid stability. It explores the performance of baseline classification models, uses an Artificial Neural Network (ANN) for feature extraction, and then combines these extracted features with various classifiers to build hybrid ensemble models. The project is organized into modules, and all analysis and visualizations are done in a Jupyter Notebook.


## Table of Contents
- [Project Overview](#ProjectOverview)
- [Dataset](#stability_dataset.csv)
- [Modules](#modules)
  - [baseline_models.py](#baseline-modelspy)
  - [ann_feature_extraction.py](#ann-feature-extractionpy)
  - [ensemble_model.py](#ensemble-model.py)
- [Jupyter Notebook](#stability_pred.ipynb)
- [Installation](#Installation)
- [Usage](#Usage)
- [Results and Visualizations](#ResultsandVisualizations)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)


# Project Overview

The primary objective of this project is to develop and evaluate machine learning models for predicting the stability of power systems. We achieve this by:

- **Establishing Baselines**: Training and evaluating common classification algorithms to set performance benchmarks.
- **ANN as Feature Extractor**: Utilizing an Artificial Neural Network not for direct prediction, but as a powerful tool to learn and extract higher-level, more informative features from the raw input data.
- **Hybrid Ensemble Modeling**: Building models that incorporate these ANN-derived features, aiming for improved predictive accuracy and generalization compared to standalone baseline models.
- **Modular Codebase**: Organizing the modeling pipeline into distinct, reusable Python scripts for clarity and maintainability.
- **In-depth Analysis**: Performing all training, evaluation, and visualization steps within a Jupyter Notebook for an interactive and reproducible workflow.

## Dataset

The project uses the `stability_dataset.csv` file to predict power system stability. The dataset includes various electrical parameters and a **stabf** (stability flag) that indicates whether the system is **stable** or **unstable**.

### Key Transformations Applied:
- The categorical **stabf** column is mapped to numerical values (**0** for 'stable', **1** for 'unstable').
- Irrelevant columns (**stab, p1, p2, p3, p4**) are removed to streamline the data.
- The dataset is split into three sets:
  - **Training (70%)**
  - **Validation (10%)**
  - **Testing (20%)**  
  Stratification is applied to maintain class balance.
- Features are scaled using **StandardScaler** to normalize the input data, ensuring optimal model performance.



