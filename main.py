from pathlib import Path
from src.load_data import load_data
from src.preprocess import preprocess_data, split_features_target
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_confusion_matrix, plot_feature_importance, plot_roc_curve


def main() -> None:
    input_path = "data/raw/data.csv"
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    df = load_data(input_path)
    df = preprocess_data(df)
    x, y = split_features_target(df, target_column="label")

    model, x_train, x_test, y_train, y_test = train_model(x, y)
    metrics, cm, report, predictions = evaluate_model(model, x_test, y_test)

    predictions.to_csv(output_dir / "predictions.csv", index=False)

    with open(output_dir / "metrics.txt", "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nInterpretation:\n")
        f.write("The model achieved high overall accuracy and strong precision for failure predictions.\n")
        f.write("Recall is moderate, which means that a relevant share of actual failures is still missed.\n")
        f.write("This reflects the challenge of imbalanced classification in predictive maintenance data.\n")
        f.write("AUC evaluates model separability across all thresholds — more reliable than accuracy for imbalanced data.\n")

    plot_confusion_matrix(cm, str(output_dir / "confusion_matrix.png"))
    plot_feature_importance(model, x.columns, str(output_dir / "feature_importance.png"))
    plot_roc_curve(
        y_test,
        predictions["predicted_probability"],
        metrics["roc_auc"],
        str(output_dir / "roc_curve.png"),
    )

    print("Training and evaluation finished successfully.")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()