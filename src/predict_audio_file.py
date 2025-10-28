import os
import json
import joblib
import numpy as np
import tensorflow as tf
import librosa
from scipy.stats import skew, kurtosis
import tkinter as tk
from tkinter import filedialog

# Constants
ARTIFACTS_DIR = 'artifacts'
MODELS_DIR = 'models'

def extract_features(file_path):
    """
    Extracts the 43 audio features from a file and returns them as a dictionary.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

    features_dict = {}

    # Basic features
    features_dict['mean_spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features_dict['std_spectral_centroid'] = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))
    features_dict['mean_spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features_dict['std_spectral_bandwidth'] = np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features_dict['mean_spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    features_dict['mean_spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=y))
    features_dict['mean_spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features_dict['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features_dict['rms_energy'] = np.mean(librosa.feature.rms(y=y))

    # Pitch features
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    features_dict['mean_pitch'] = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    features_dict['min_pitch'] = np.min(pitch_values) if len(pitch_values) > 0 else 0
    features_dict['max_pitch'] = np.max(pitch_values) if len(pitch_values) > 0 else 0
    features_dict['std_pitch'] = np.std(pitch_values) if len(pitch_values) > 0 else 0

    # Advanced features
    stft = np.abs(librosa.stft(y))
    features_dict['spectral_skew'] = np.mean(skew(stft, axis=0))
    features_dict['spectral_kurtosis'] = np.mean(kurtosis(stft, axis=0))

    # Simplified energy entropy
    frame_len = int(sr * 0.02)
    hop_len = int(sr * 0.01)
    energy = np.array([np.sum(np.square(y[i:i+frame_len])) for i in range(0, len(y) - frame_len, hop_len)])
    energy_prob = energy / np.sum(energy)
    features_dict['energy_entropy'] = -np.sum(energy_prob * np.log2(energy_prob + 1e-6))
    features_dict['log_energy'] = np.log(np.sum(np.square(y)) + 1e-6)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)
    for i in range(13):
        features_dict[f'mfcc_{i+1}_mean'] = mfcc_means[i]
        features_dict[f'mfcc_{i+1}_std'] = mfcc_stds[i]

    return features_dict

def predict_audio_file(file_path):
    """
    Main prediction workflow for a single audio file.
    """
    # Load necessary artifacts
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'best_model.keras'))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
    feature_list = joblib.load(os.path.join(ARTIFACTS_DIR, 'feature_list.pkl'))
    with open(os.path.join(ARTIFACTS_DIR, 'classes.json'), 'r') as f:
        class_mapping = json.load(f)

    # 1. Extract features from the selected audio file
    print(f"Extracting features from: {os.path.basename(file_path)}...")
    features_dict = extract_features(file_path)

    if features_dict is None:
        return

    # 2. Assemble features in the correct order
    ordered_features = np.array([features_dict[feature] for feature in feature_list])

    # 3. Scale the features
    features_scaled = scaler.transform(ordered_features.reshape(1, -1))

    # 4. Make prediction
    pred_proba = model.predict(features_scaled)
    pred_class_idx = np.argmax(pred_proba, axis=1)[0]

    # 5. Decode and display results
    predicted_class_name = class_mapping[str(pred_class_idx)]
    probabilities = {class_mapping[str(i)]: f"{p:.2%}" for i, p in enumerate(pred_proba[0])}

    print("\n--- Prediction Result ---")
    print(f"Predicted Class: {predicted_class_name}")
    print("\nProbabilities:")
    for class_name, prob in probabilities.items():
        print(f"  - {class_name}: {prob}")
    print("-------------------------\n")


def main():
    """
    Initializes Tkinter and opens the file selection dialog.
    """
    root = tk.Tk()
    root.withdraw()

    print("Please select an audio file to predict...")
    file_path = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac")]
    )

    if file_path:
        predict_audio_file(file_path)
    else:
        print("No file selected. Exiting.")


if __name__ == '__main__':
    main()
