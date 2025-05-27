# import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def base_models_predictions(X_train_scaled, X_val_scaled, y_train, y_val, X_test_scaled, y_test):
    """
    Trains and evaluates several base classification models, compares their performance
    on training, validation, and test sets, and visualizes the results.

    The function instantiates Logistic Regression, Random Forest, K-Nearest Neighbors (KNN),
    Support Vector Machine (SVM), and Decision Tree models. It then trains each model
    on the scaled training data and evaluates their accuracy on the training, validation,
    and test sets. The results are compiled into a pandas DataFrame and displayed.

    Args:
        X_train_scaled (array-like): Scaled features of the training set.
        X_val_scaled (array-like): Scaled features of the validation set.
        y_train (array-like): Target labels of the training set.
        y_val (array-like): Target labels of the validation set.
        X_test_scaled (array-like): Scaled features of the test set.
        y_test (array-like): Target labels of the test set.

    Returns:
        pd.DataFrame: A DataFrame containing the 'Model', 'Train Accuracy',
                      'Validation Accuracy', and 'Test Accuracy' for each base model,
                      sorted by 'Validation Accuracy' in descending order.
    """

    # Instantiate the models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=5,max_features='sqrt', random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "DT": DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42)
    }

    results = []
    print("\n Training and Evaluating Base Models ")

    for name, model in models.items():
        # Train the models
        model.fit(X_train_scaled, y_train)

        # Evaluate on Training Set
        y_train_pred = model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Evaluate on Validation Set
        y_val_pred = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        # Evaluate on Test Set
        y_test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        results.append({
            "Model": name,
            "Train Accuracy": round(train_accuracy * 100, 2),
            "Validation Accuracy": round(val_accuracy * 100, 2),
            "Test Accuracy": round(test_accuracy * 100, 2)
        })
    results_df = pd.DataFrame(results).sort_values(by="Validation Accuracy", ascending=False)
    return results_df


