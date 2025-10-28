import os
import json
import joblib
import numpy as np
import tensorflow as tf

# Constants
ARTIFACTS_DIR = 'artifacts'
MODELS_DIR = 'models'

def predict_new_sample(X_new):
    """
    Predicts the class for a new data sample.

    Args:
        X_new (np.array): A 1D numpy array with 43 features.

    Returns:
        tuple: A tuple containing the predicted class name and a dictionary of class probabilities.
    """
    # Load artifacts
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'best_model.keras'))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
    with open(os.path.join(ARTIFACTS_DIR, 'classes.json'), 'r') as f:
        class_mapping = json.load(f)

    # Preprocess the input
    X_new_scaled = scaler.transform(X_new.reshape(1, -1))

    # Make prediction
    y_pred_proba = model.predict(X_new_scaled)
    y_pred_class_idx = np.argmax(y_pred_proba, axis=1)[0]

    # Decode prediction
    predicted_class_name = class_mapping[str(y_pred_class_idx)]

    # Get class probabilities
    probabilities = {class_mapping[str(i)]: float(p) for i, p in enumerate(y_pred_proba[0])}

    return predicted_class_name, probabilities

if __name__ == '__main__':
    # Create a dummy sample for prediction
    dummy_sample = np.random.rand(43)

    # Get prediction
    predicted_class, class_probabilities = predict_new_sample(dummy_sample)

    # Print results
    print(f"Predicted Class: {predicted_class}")
    print("Class Probabilities:")
    for class_name, prob in class_probabilities.items():
        print(f"  {class_name}: {prob:.4f}")
