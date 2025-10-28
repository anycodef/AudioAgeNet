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
    Extracts the 43 audio features from a file, matching the training data format.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

    # 1. Spectral Centroid (mean, std)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_spectral_centroid = np.mean(spectral_centroid)
    std_spectral_centroid = np.std(spectral_centroid)

    # 2. Spectral Bandwidth (mean, std)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    mean_spectral_bandwidth = np.mean(spectral_bandwidth)
    std_spectral_bandwidth = np.std(spectral_bandwidth)

    # 3. Spectral Contrast (mean)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mean_spectral_contrast = np.mean(spectral_contrast)

    # 4. Spectral Flatness (mean)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    mean_spectral_flatness = np.mean(spectral_flatness)

    # 5. Spectral Rolloff (mean)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    mean_spectral_rolloff = np.mean(spectral_rolloff)

    # 6. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate = np.mean(zcr)

    # 7. RMS Energy
    rms = librosa.feature.rms(y=y)
    rms_energy = np.mean(rms)

    # 8. Pitch (mean, min, max, std)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    # Get the pitch for each frame where there is energy
    pitch_values = pitches[pitches > 0]
    if len(pitch_values) > 0:
        mean_pitch = np.mean(pitch_values)
        min_pitch = np.min(pitch_values)
        max_pitch = np.max(pitch_values)
        std_pitch = np.std(pitch_values)
    else:
        mean_pitch, min_pitch, max_pitch, std_pitch = 0, 0, 0, 0

    # 9. Spectral Skew and Kurtosis
    stft = np.abs(librosa.stft(y))
    spectral_skew = np.mean(skew(stft, axis=0))
    spectral_kurtosis = np.mean(kurtosis(stft, axis=0))

    # 10. Energy Entropy
    # This is a simplified version; for a perfect match, the original implementation would be needed.
    frame_len = int(sr * 0.02) # 20ms frame
    hop_len = int(sr * 0.01) # 10ms hop
    energy = np.array([
        np.sum(np.square(y[i:i+frame_len]))
        for i in range(0, len(y) - frame_len, hop_len)
    ])
    energy_prob = energy / np.sum(energy)
    energy_entropy = -np.sum(energy_prob * np.log2(energy_prob + 1e-6))

    # 11. Log Energy
    log_energy = np.log(np.sum(np.square(y)) + 1e-6)

    # 12. MFCCs (13 coefficients, mean and std)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)

    # Combine all features into a single array in the correct order
    features = np.array([
        mean_spectral_centroid, std_spectral_centroid,
        mean_spectral_bandwidth, std_spectral_bandwidth,
        mean_spectral_contrast, mean_spectral_flatness,
        mean_spectral_rolloff, zero_crossing_rate, rms_energy,
        mean_pitch, min_pitch, max_pitch, std_pitch,
        spectral_skew, spectral_kurtosis, energy_entropy, log_energy,
        *mfcc_means, *mfcc_stds
    ])

    return features

def predict_audio_file(file_path):
    """
    Main prediction workflow for a single audio file.
    """
    # Load necessary artifacts
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'best_model.keras'))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
    with open(os.path.join(ARTIFACTS_DIR, 'classes.json'), 'r') as f:
        class_mapping = json.load(f)

    # 1. Extract features from the selected audio file
    print(f"Extracting features from: {os.path.basename(file_path)}...")
    features = extract_features(file_path)

    if features is None:
        return

    # 2. Scale the features
    features_scaled = scaler.transform(features.reshape(1, -1))

    # 3. Make prediction
    pred_proba = model.predict(features_scaled)
    pred_class_idx = np.argmax(pred_proba, axis=1)[0]

    # 4. Decode and display results
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
    root.withdraw()  # Hide the main window

    print("Please select an audio file to predict...")
    file_path = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac")]
    )

    if file_path:  # Proceed if the user selected a file
        predict_audio_file(file_path)
    else:
        print("No file selected. Exiting.")


if __name__ == '__main__':
    main()
