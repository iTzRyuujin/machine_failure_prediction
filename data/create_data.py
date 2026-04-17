from pathlib import Path
from ucimlrepo import fetch_ucirepo


def main() -> None:
    dataset = fetch_ucirepo(id=601)

    x_data = dataset.data.features
    y_data = dataset.data.targets

    df = x_data.copy()
    df["label"] = y_data["Machine failure"].values

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "data.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Failures: {df['label'].sum()} ({df['label'].mean() * 100:.2f}%)")


if __name__ == "__main__":
    main()