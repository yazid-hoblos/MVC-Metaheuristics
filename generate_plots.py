"""
Generate plots from results/results.csv and save them to report/figures.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "results.csv"
FIG_DIR = ROOT / "report" / "figures"


def _read_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    # Normalize boolean column that may be read as string
    df["is_valid"] = df["is_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def _save_fig(fig, filename: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    print(f"Saved: {path}")


def plot_algorithm_quality(df: pd.DataFrame):
    valid = df[df["is_valid"]].copy()
    stats = valid.groupby("algorithm")["cover_size"].agg(["mean", "std", "count"]).reindex(["GA", "SA", "TS"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(stats.index, stats["mean"], yerr=stats["std"], capsize=5, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("Average Cover Size (Valid Solutions Only)")
    ax.set_ylabel("Cover Size")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i, (algo, row) in enumerate(stats.iterrows()):
        ax.text(i, row["mean"] + (row["std"] if not np.isnan(row["std"]) else 0) + 0.5,
                f"n={int(row['count'])}", ha="center", fontsize=9)

    _save_fig(fig, "algorithm_quality.png")


def plot_algorithm_time(df: pd.DataFrame):
    stats = df.groupby("algorithm")["time_sec"].agg(["mean", "std"]).reindex(["GA", "SA", "TS"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(stats.index, stats["mean"], yerr=stats["std"], capsize=5, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("Average Runtime per Algorithm")
    ax.set_ylabel("Time (seconds)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    _save_fig(fig, "algorithm_time.png")


def plot_encoding_quality(df: pd.DataFrame):
    valid = df[df["is_valid"]].copy()
    stats = valid.groupby(["encoding", "algorithm"])["cover_size"].mean().unstack("algorithm").reindex(
        ["BinaryEncoding", "SetEncoding", "EdgeCentricEncoding"]
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(stats.index))
    width = 0.25
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for i, algo in enumerate(["GA", "SA", "TS"]):
        ax.bar(x + i * width - width, stats[algo], width, label=algo, color=colors[i])

    ax.set_title("Average Cover Size by Encoding (Valid Only)")
    ax.set_ylabel("Cover Size")
    ax.set_xticks(x)
    ax.set_xticklabels(["Binary", "Set", "Edge-Centric"])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    _save_fig(fig, "encoding_quality.png")


def plot_fitness_validity(df: pd.DataFrame):
    validity = df.groupby(["fitness_func", "algorithm"])["is_valid"].mean().unstack("algorithm").reindex(
        ["CoverSizeMinimization", "ConstraintPenalty", "EdgeCoverageOptimization"]
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(validity.index))
    width = 0.25
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for i, algo in enumerate(["GA", "SA", "TS"]):
        ax.bar(x + i * width - width, 100 * validity[algo], width, label=algo, color=colors[i])

    ax.set_title("Feasibility Rate by Fitness Function")
    ax.set_ylabel("Valid Solutions (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Size Min", "Penalty", "Edge Coverage"])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    _save_fig(fig, "fitness_validity.png")


def plot_instance_performance(df: pd.DataFrame):
    valid = df[df["is_valid"]].copy()
    stats = valid.groupby(["instance", "algorithm"])["cover_size"].mean().unstack("algorithm")
    ordered_instances = ["small_20nodes", "medium_50nodes", "large_100nodes", "scale_free_50nodes"]
    stats = stats.reindex(ordered_instances)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(stats.index))
    width = 0.25
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for i, algo in enumerate(["GA", "SA", "TS"]):
        ax.bar(x + i * width - width, stats[algo], width, label=algo, color=colors[i])

    ax.set_title("Instance-Level Average Cover Size (Valid Only)")
    ax.set_ylabel("Cover Size")
    ax.set_xticks(x)
    ax.set_xticklabels(["Small", "Medium", "Large", "Scale-Free"])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    _save_fig(fig, "instance_performance.png")


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")

    df = _read_results()
    plot_algorithm_quality(df)
    plot_algorithm_time(df)
    plot_encoding_quality(df)
    plot_fitness_validity(df)
    plot_instance_performance(df)


if __name__ == "__main__":
    main()
