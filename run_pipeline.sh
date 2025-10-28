#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Get the absolute path of the directory where the script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Run the data preparation script
echo "Running data preparation..."
python "${SCRIPT_DIR}/src/prepare_data.py"

# Run the model training script
echo "Running model training..."
python "${SCRIPT_DIR}/src/train.py"

# Run the model evaluation script
echo "Running model evaluation..."
python "${SCRIPT_DIR}/src/evaluate.py"

echo "Pipeline finished successfully!"
