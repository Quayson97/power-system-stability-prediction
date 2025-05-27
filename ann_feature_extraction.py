# Import libraries
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

def build_ann_model(input_shape):
    """
    Builds and compiles an Artificial Neural Network (ANN) model for binary classification.

    This function constructs a feedforward neural network with:
    - An input layer matching the specified number of features.
    - Two hidden layers with ReLU activation (128 and 64 units, respectively).
    - A single-node output layer with sigmoid activation for probability-based classification.

    The model is compiled with:
    - Optimizer: Adam (adaptive learning rate optimization).
    - Loss function: Binary Crossentropy (for classification tasks).
    - Metric: Accuracy (to evaluate model performance).

    Parameters:
    input_shape (int): Number of features in the input data, defining the shape of the input layer.

    Returns:
    tensorflow.keras.models.Model: A compiled Keras model ready for training on classification tasks.
    """
    inputs = Input(shape=(input_shape,), name="input_layer")
    x = Dense(128, activation='relu', name='hidden_1')(inputs)
    x = Dense(64, activation='relu', name='hidden_2')(x)
    outputs = Dense(1, activation='sigmoid', name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def extract_ann_features(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val,y_test, epochs=100, batch_size=8):
    """
    Extracts features from a trained ANN model for further ensemble learning.

    The function trains an Artificial Neural Network (ANN).
    It incorporates early stopping to prevent overfitting based on validation accuracy.
    After training, it extracts features from the second hidden layer and returns them
    for integration into other machine learning models.

    Parameters:
    X_train_scaled (numpy.ndarray): Scaled feature matrix for training.
    X_val_scaled (numpy.ndarray): Scaled feature matrix for validation.
    X_test_scaled (numpy.ndarray): Scaled feature matrix for testing.
    y_train (numpy.ndarray): Training labels.
    y_val (numpy.ndarray): Validation labels.
    epochs (int, optional): Number of training epochs. Default is 100.
    batch_size (int, optional): Mini-batch size for training. Default is 8.

    Returns:
    tuple:
        - X_train_features (numpy.ndarray): Extracted features from the ANN for training data.
        - X_val_features (numpy.ndarray): Extracted features from the ANN for validation data.
        - X_test_features (numpy.ndarray): Extracted features from the ANN for test data.
        - model (tensorflow.keras.Model): Trained ANN model.
    """
    model = build_ann_model(X_train_scaled.shape[1])
    print("\n Training ANN model (Feature Extractor)")

    # Use EarlyStopping based on validation accuracy to prevent MLP overfitting
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_data=(X_val_scaled, y_val),
        callbacks=[early_stopping]
    )
    # Evaluate original ANN model on the dedicated test set
    loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"ANN Model Final Test Accuracy: {acc * 100:.2f}%")

    # Create feature extractor model from hidden layer_2
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer('hidden_2').output)
    X_train_features = feature_extractor.predict(X_train_scaled)
    X_val_features = feature_extractor.predict(X_val_scaled)
    X_test_features = feature_extractor.predict(X_test_scaled)

    return X_train_features, X_val_features, X_test_features, model, history