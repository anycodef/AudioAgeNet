import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
ARTIFACTS_DIR = 'artifacts'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'

def evaluate_model():
    # Load model, data, and class names
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'best_model.keras'))
    X_test = np.load(os.path.join(ARTIFACTS_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(ARTIFACTS_DIR, 'y_test.npy'))

    with open(os.path.join(ARTIFACTS_DIR, 'classes.json'), 'r') as f:
        class_mapping = json.load(f)
    class_names = [class_mapping[str(i)] for i in sorted(class_mapping, key=int)]

    print("Model and data loaded successfully.")

    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Generate classification report
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=class_names)

    # Calculate balanced accuracy
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report_dict['balanced_accuracy'] = bal_acc

    print("Classification Report:")
    print(report_str)
    print(f"Balanced Accuracy: {bal_acc}")

    # Save metrics to JSON
    with open(os.path.join(ARTIFACTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(report_dict, f, indent=4)
    print("Metrics saved to JSON.")

    # Save classification report to markdown
    with open(os.path.join(REPORTS_DIR, 'test_report.md'), 'w') as f:
        f.write("# Test Set Evaluation Report\n\n")
        f.write(f"**Balanced Accuracy:** {bal_acc:.4f}\n\n")
        f.write("```\n")
        f.write(report_str)
        f.write("\n```\n")
    print("Test report saved to markdown.")

    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'))
    print("Confusion matrix saved.")

if __name__ == '__main__':
    evaluate_model()
