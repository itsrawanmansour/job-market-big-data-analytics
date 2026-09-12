from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]

ML_DIR = BASE_DIR / "data" / "output" / "ml"
FIGURES_DIR = BASE_DIR / "reports" / "figures" / "ml"

SUMMARY_PATH = ML_DIR / "ml_summary.txt"
CONFUSION_PATH = ML_DIR / "confusion_matrix.csv"
TRAIN_DIST_PATH = ML_DIR / "train_salary_level_distribution.csv"
TEST_DIST_PATH = ML_DIR / "test_salary_level_distribution.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def extract_metric(text, metric_name):
    match = re.search(rf"{metric_name}:\s*([0-9.]+)", text)
    if match:
        return float(match.group(1))
    return None


def load_metrics():
    text = SUMMARY_PATH.read_text(encoding="utf-8")
    accuracy = extract_metric(text, "Accuracy")
    f1_score = extract_metric(text, "F1-score")
    return accuracy, f1_score


def plot_model_performance(accuracy, f1_score):
    labels = ["Accuracy", "F1-score"]
    values = [accuracy, f1_score]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)
    plt.ylim(0, 1)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center"
        )

    plt.title("Model Performance")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_performance.png", dpi=300)
    plt.close()


def plot_confusion_matrix():
    df = pd.read_csv(CONFUSION_PATH)

    actual_order = ["Low", "Medium", "High"]
    predicted_order = ["Low", "Medium", "High"]

    matrix = (
        df.pivot(
            index="actual_salary_level",
            columns="predicted_salary_level",
            values="count"
        )
        .reindex(index=actual_order, columns=predicted_order, fill_value=0)
    )

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix.values)

    plt.xticks(range(len(predicted_order)), predicted_order)
    plt.yticks(range(len(actual_order)), actual_order)

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300)
    plt.close()


def plot_distribution(csv_path, output_name, title):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="salary_level")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(df["salary_level"], df["count"])

    for bar, value in zip(bars, df["count"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(df["count"]) * 0.01,
            f"{int(value)}",
            ha="center"
        )

    plt.title(title)
    plt.xlabel("Salary Level")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300)
    plt.close()


def main():
    accuracy, f1_score = load_metrics()

    plot_model_performance(accuracy, f1_score)
    plot_confusion_matrix()
    plot_distribution(TRAIN_DIST_PATH, "train_distribution.png", "Training Set Salary Level Distribution")
    plot_distribution(TEST_DIST_PATH, "test_distribution.png", "Testing Set Salary Level Distribution")

    print("ML charts generated successfully.")
    print(f"Charts saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()