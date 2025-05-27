# import libraries
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def train_ensemble_models(X_train_features, X_val_features, X_test_features, y_train, y_val, y_test):
    """
    Trains and evaluates multiple ensemble learning models using extracted ANN features.

    This function takes feature representations obtained from an ANN model and trains
    various machine learning models, including Logistic Regression, Random Forest, KNN,
    SVM, and Decision Trees. It evaluates their performance on both the training and
    validation datasets to monitor overfitting. Additionally, it performs a final
    evaluation on an unseen test dataset to assess generalization.

    Parameters:
    X_train_features (numpy.ndarray): Extracted ANN features for training.
    X_val_features (numpy.ndarray): Extracted ANN features for validation.
    X_test_features (numpy.ndarray): Extracted ANN features for testing.
    y_train (numpy.ndarray): Training labels.
    y_val (numpy.ndarray): Validation labels.
    y_test (numpy.ndarray): Test labels.

    Returns:
    pandas.DataFrame: A DataFrame summarizing the Train , Validation and Test set accuracy of each model.
    """
    models = {
        "ANN-Logistic Regression": LogisticRegression(random_state=42),
        "ANN-Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=5,max_features='sqrt', random_state=42),
        "ANN-KNN": KNeighborsClassifier(),
        "ANN-SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "ANN-DT": DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=10, min_samples_leaf=5,random_state=42)
    }

    val_train_results = []
    test_results = []

    print("\n Training and Evaluating Hybrid Models")
    for name, model in models.items():
        model.fit(X_train_features, y_train)

        # Evaluate on Training Features
        y_train_pred = model.predict(X_train_features)
        train_acc = accuracy_score(y_train, y_train_pred)

        # Evaluate on Validation Features
        y_val_pred = model.predict(X_val_features)
        val_acc = accuracy_score(y_val, y_val_pred)

        # Evaluate on Testing Features
        y_test_pred = model.predict(X_test_features)
        test_acc = accuracy_score(y_test, y_test_pred)

        val_train_results.append({
            "Model": name,
            "Train Accuracy": round(train_acc * 100, 2),
            "Validation Accuracy": round(val_acc * 100, 2)
        })

        test_results.append({
            "Model": name,
            "Test Accuracy": round(test_acc * 100, 2)
        })
        val_train_results_df = pd.DataFrame(val_train_results).sort_values(by="Validation Accuracy", ascending=False)
        test_results_df = pd.DataFrame(test_results).sort_values(by="Test Accuracy", ascending=False)

    return val_train_results_df, test_results_df