"""Evaluation utilities for classification experiments."""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_classifier(y_true, y_pred):
    """Return common classification metrics and print detailed results."""
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=["model_a", "model_b"]
        ),
        "classification_report": classification_report(
            y_true, y_pred, labels=["model_a", "model_b"], zero_division=0
        ),
    }
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print("\nClassification report:")
    print(results["classification_report"])
    print("Confusion matrix (rows=true, columns=predicted; A then B):")
    print(results["confusion_matrix"])
    return results
