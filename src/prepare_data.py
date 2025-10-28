import os
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Constants
DATA_PATH = 'data/audio_features_with_gender_and_age.csv'
ARTIFACTS_DIR = 'artifacts'
REPORTS_DIR = 'reports'
SEED = 42

# Create directories if they don't exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def prepare_data():
    # Set seed for reproducibility
    np.random.seed(SEED)

    # Load data
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")

    # Sanity checks
    print(f"Initial shape: {df.shape}")
    df.drop_duplicates(inplace=True)
    print(f"Shape after dropping duplicates: {df.shape}")

    valid_gender = {'male', 'female'}
    valid_age = {'young', 'adult', 'senior'}
    assert set(df['gender'].unique()) <= valid_gender
    assert set(df['age'].unique()) <= valid_age
    print("Sanity checks passed.")

    # Create target variable
    df['target6'] = df['gender'] + '_' + df['age']

    # Encode target variable
    le = LabelEncoder()
    df['target6_encoded'] = le.fit_transform(df['target6'])

    # Save class mapping
    class_mapping = {int(i): str(c) for i, c in enumerate(le.classes_)}
    with open(os.path.join(ARTIFACTS_DIR, 'classes.json'), 'w') as f:
        json.dump(class_mapping, f, indent=4)
    print("Class mapping saved.")

    # Separate features and target
    excluded_cols = ['gender', 'age', 'target6', 'target6_encoded']
    features = [col for col in df.columns if col not in excluded_cols]
    X = df[features]
    y = df['target6_encoded']

    # Stratified split: 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )

    # Stratified split of train to get validation set (15% of original train)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
    )

    print("Data split into train, validation, and test sets.")
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
    print("Scaler saved.")

    # Save processed data
    np.save(os.path.join(ARTIFACTS_DIR, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(ARTIFACTS_DIR, 'X_val.npy'), X_val_scaled)
    np.save(os.path.join(ARTIFACTS_DIR, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(ARTIFACTS_DIR, 'y_train.npy'), y_train.values)
    np.save(os.path.join(ARTIFACTS_DIR, 'y_val.npy'), y_val.values)
    np.save(os.path.join(ARTIFACTS_DIR, 'y_test.npy'), y_test.values)
    print("Processed data saved.")

if __name__ == '__main__':
    prepare_data()
