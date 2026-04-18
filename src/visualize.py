import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def plot_confusion_matrix(cm, output_path: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No Failure", "Failure"])
    plt.yticks(tick_marks, ["No Failure", "Failure"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance(model, feature_names, output_path: str) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_features, sorted_importances)
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(y_test, y_proba, auc_score: float, output_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess (AUC = 0.500)")
    plt.xlabel("False Positive Rate  (unnecessary maintenance triggers)")
    plt.ylabel("True Positive Rate  (failures correctly detected)")
    plt.title("ROC Curve — Machine Failure Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()