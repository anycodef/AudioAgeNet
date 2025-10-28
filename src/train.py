import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Constants
ARTIFACTS_DIR = 'artifacts'
MODELS_DIR = 'models'
SEED = 42

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)

def train_model():
    # Set seed for reproducibility
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Load data
    X_train = np.load(os.path.join(ARTIFACTS_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(ARTIFACTS_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(ARTIFACTS_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(ARTIFACTS_DIR, 'y_val.npy'))
    print("Data loaded successfully.")

    # Get input shape and number of classes
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # Define the MLP architecture
    model = Sequential([
        Dense(256, kernel_initializer='he_normal', input_shape=(n_features,)),
        BatchNormalization(),
        ReLU(),
        Dropout(0.2),
        Dense(128, kernel_initializer='he_normal'),
        BatchNormalization(),
        ReLU(),
        Dropout(0.2),
        Dense(64, kernel_initializer='he_normal'),
        ReLU(),
        Dense(n_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    print("Model compiled.")

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, 'best_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-5,
        verbose=1
    )

    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weights_dict}")


    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[model_checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights_dict,
        verbose=1
    )

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(ARTIFACTS_DIR, 'history.csv'), index=False)
    print("Training history saved.")

if __name__ == '__main__':
    train_model()
