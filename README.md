# Power System Stability Prediction: A Comparative Analysis of Baseline models, ANN and Hybrid Models(ANN + Baseline models)
This repository presents a machine learning approach to predict smart grid stability. It explores the performance of baseline classification models, uses an Artificial Neural Network (ANN) for feature extraction, and then combines these extracted features with various classifiers to build hybrid ensemble models. The project is organized into modules, and all analysis and visualizations are done in a Jupyter Notebook.


## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Modules](#modules)
  - [baseline_models.py](#baseline_modelspy)
  - [ann_feature_extraction.py](#ann_feature_extractionpy)
  - [ensemble_models.py](#ensemble_modelspy)
- [Jupyter Notebook](#jupyter-notebook)
- [Results and Visualizations](#results-and-visualizations)
- [Contact](#contact)

# Project Overview

The primary objective of this project is to develop and evaluate machine learning models for predicting the stability of a smart grid system. This is achieved by:

- **Establishing Baseline Models**: Training and evaluating common classification algorithms such as Logistic Regression, Support Vector Machines, Random Forests, K-Nearest Neighbours, and Decision Trees to set performance benchmarks.
- **ANN as Feature Extractor**: Utilizing an Artificial Neural Network for prediction, learning, and extracting informative features from hidden layers of the ANN.
- **Hybrid Ensemble Modeling**: Creating models that use features extracted by the ANN to improve accuracy and make better predictions than baseline models.
- **Modular Codebase**: Organizing the modeling pipeline into distinct, reusable Python scripts.
- **In-depth Analysis**: Performing all training, evaluation, and visualization steps within a Jupyter Notebook.

## Dataset

The project uses the `stability_dataset.csv` file from the UCI Machine Learning Repository to predict the stability of the smart grid. There are 10,000 samples of 12 independent features. The features include Producers/Consumers(**p1, p2, p3, p4**), Reaction times(**tau1, tau2, tau3, tau4**) and Price elasticity(**g1, g2, g3, g4**) and a **stab** and **stabf** (stability flag) that indicates whether the system is **stable** or **unstable**.

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
This module trains an  Artificial Neural Network (ANN) and extracts hidden features for hybrid modeling. It includes functions to:

- **Build a simple ANN**: A multi-layer perceptron with two hidden layers.
- **Train the ANN**: The network is trained for classification using accuracy as the metric, with Early Stopping to prevent overfitting.
- **Extract Features**: Once trained, the second hidden layer outputs are used as extracted features. These features capture complex, non-linear patterns in the data.

### ensemble_model.py
This module utilizes features extracted by the ANN to train other baseline classifiers. It creates **hybrid models**, combining the ANN's feature learning strengths with the baseline classifiers in **base_models.py**. The trained models include:

- **Logistic Regression (on ANN features)**
- **Random Forest (on ANN features)**
- **K-Nearest Neighbors (on ANN features)**
- **Support Vector Machine (on ANN features)**
- **Decision Tree (on ANN features)**

By comparing the performance of these hybrid models, this module evaluates how **ANN-based feature engineering** improves prediction accuracy.

## Jupyter Notebook

The `stability_pred.ipynb` notebook orchestrates the entire workflow of the project. It handles:

- **Data Loading and Preprocessing**: Reads the dataset, processes categorical variables, and splits data into training, validation, and test sets.
- **Model Training and Evaluation**: imports and use functions `baseline_models.py`, `ann_feature_extractor.py`, and `ensemble_model.py` to train and evaluate all models.
- **Performance Metrics**: Computes accuracy scores across training, validation, and test sets to assess model performance.
- **Visualizations**: Generates comparative bar plots for model accuracies (validation and test) and learning curves for the ANN model for easy interpretation of results and identification of overfitting.

## Results and Visualizations

The Jupyter Notebook generates several insightful visualizations to evaluate model performance:

- **Base Model Validation Accuracy Comparison**: Bar plot comparing the validation accuracy of initial baseline models.
- **Base Model Test Accuracy Comparison**: Bar plot showing the test accuracy of baseline models.
- **Base Models: Train vs. Validation Accuracy**: Comparison plot to detect potential overfitting in baseline models.
- **ANN Model Learning Curves**: Graphs displaying training and validation accuracy and loss over epochs for the ANN feature extractor, for assessing its learning process and convergence.
- **Hybrid Models: Train vs. Validation Accuracy**: Comparison plot for the models trained on ANN-extracted features to for overfitting.
- **Hybrid Model Final Test Accuracy**: Bar plot highlighting the test performance of models utilizing ANN-derived features.

These plots provide a clear picture of how different modeling strategies perform in predicting power system stability.

## Contact

- **Name:** Ekow Quayson
- **LinkedIn:** [LinkedIn Profile ](https://www.linkedin.com/in/ekow-quayson/)
- **Email:** ekowquayson5@gmail.com

