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

The primary objective of this project is to develop and evaluate machine learning models for predicting the stability of a smart grid system. This is achieved by:

- **Establishing Baseline Models**: Training and evaluating common classification algorithms such as Logistic Regression, Support Vector Machines, Random Forest, K-Nearest Neighbours, and Decision Trees to set performance benchmarks.
- **ANN as Feature Extractor**: Utilizing an Artificial Neural Network for prediction, learning, and extracting informative features from hidden layers of the ANN.
- **Hybrid Ensemble Modeling**: Creating models that use features extracted by the ANN to improve accuracy and make better predictions than baseline models.
- **Modular Codebase**: Organizing the modeling pipeline into distinct, reusable Python scripts.
- **In-depth Analysis**: Performing all training, evaluation, and visualization steps within a Jupyter Notebook.

## Dataset

The project uses the `stability_dataset.csv` file from the UCI Machine Learning Repository to predict the stability of the smart grid. There are 10,000 Samples of 12 independent features. The features include Producers/Consumers(p1,p2,p3,p4), Reaction times(tau1, tau2, tau3, tau4) and Price elasticity(g1,g2,g3,g4) and a **stab** and **stabf** (stability flag) that indicates whether the system is **stable** or **unstable**.

### Preprocessing Techniques Applied:
- The categorical **stabf** column is mapped to numerical values (**0** for 'stable', **1** for 'unstable').
- Irrelevant columns (**stab, p1, p2, p3, p4**) are removed to streamline the data.
- The dataset is split into three sets:
  - **Training (70%)**
  - **Validation (10%)**
  - **Testing (20%)**  
- Stratification is applied to maintain class balance.
- Features are scaled using **StandardScaler** to normalize the input data.

## Modules

The project's functionality is built using three Python modules, ensuring a well-organized and reusable codebase.

### baseline_models.py
This module implements several traditional machine learning classifiers, providing a baseline for comparison against more advanced models. The models include:

- **Logistic Regression**
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Decision Tree Classifier**

These models are trained and evaluated on **training, validation, and test datasets** and provide insights on their predictive performance before feature extraction and ensemble modeling are applied.

### ann_feature_extractor.py
This script trains an  Artificial Neural Network (ANN) and extracts hidden features for hybrid modeling. It includes functions to:

- **Build a simple ANN**: A multi-layer perceptron with two hidden layers.
- **Train the ANN**: The network is trained for classification using validation accuracy, with Early Stopping to prevent overfitting.
- **Extract Features**: Once trained, the second hidden layer outputs are used as extracted features. These features capture complex, non-linear patterns in the data.

### ensemble_model.py
This module utilizes features extracted by the ANN to train other baseline classifiers. It creates **hybrid models**, combining the ANN's feature learning strengths with the baseline classifiers in **base_models.py**. The trained models include:

- **Logistic Regression (on ANN features)**
- **Random Forest (on ANN features)**
- **K-Nearest Neighbors (on ANN features)**
- **Support Vector Machine (on ANN features)**
- **Decision Tree (on ANN features)**

By comparing the performance of these hybrid models, this module evaluates how **ANN-based feature engineering** improves prediction accuracy.


