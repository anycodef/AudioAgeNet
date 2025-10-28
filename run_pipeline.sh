#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Run the data preparation script
echo "Running data preparation..."
python src/prepare_data.py

# Run the model training script
echo "Running model training..."
python src/train.py

# Run the model evaluation script
echo "Running model evaluation..."
python src/evaluate.py

echo "Pipeline finished successfully!"
