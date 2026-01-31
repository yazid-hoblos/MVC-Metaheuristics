"""
Generate plots from results/results.csv and save them to report/figures.

Usage:
    python3 generate_plots.py                              # Default: results/
    python3 generate_plots.py --results ./my_results       # Custom results folder
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = ROOT / "results" / "results.csv"
DEFAULT_FIG_DIR = ROOT / "report" / "figures"

# These will be set by main()
RESULTS_PATH = DEFAULT_RESULTS_PATH
FIG_DIR = DEFAULT_FIG_DIR


def _read_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    # Normalize boolean column that may be read as string
    df["is_valid"] = df["is_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def _save_fig(fig, filename: str):
    """Save figure with high DPI."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def plot_algorithm_quality(df: pd.DataFrame):
    valid = df[df["is_valid"]].copy()
    stats = valid.groupby("algorithm")["cover_size"].agg(["mean", "std", "count"]).reindex(["GA", "SA", "TS"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(stats.index, stats["mean"], yerr=stats["std"], capsize=5, color=["#4C78A8", "#F58518", "#54A24B"], alpha=0.7)
    ax.set_title("Average Cover Size (Valid Solutions Only)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cover Size", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for i, (algo, row) in enumerate(stats.iterrows()):
        ax.text(i, row["mean"] + (row["std"] if not np.isnan(row["std"]) else 0) + 0.5,
                f"n={int(row['count'])}", ha="center", fontsize=9)

    _save_fig(fig, "algorithm_quality.png")


def plot_algorithm_time(df: pd.DataFrame):
    stats = df.groupby("algorithm")["time_sec"].agg(["mean", "std"]).reindex(["GA", "SA", "TS"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(stats.index, stats["mean"], yerr=stats["std"], capsize=5, color=["#4C78A8", "#F58518", "#54A24B"], alpha=0.7)
    ax.set_title("Average Runtime per Algorithm", fontsize=12, fontweight='bold')
    ax.set_ylabel("Time (seconds)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

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
        ax.bar(x + i * width - width, stats[algo], width, label=algo, color=colors[i], alpha=0.7)

    ax.set_title("Average Cover Size by Encoding (Valid Only)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cover Size", fontsize=11)
    ax.set_xlabel("Encoding", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(["Binary", "Set", "Edge-Centric"])
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

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
        ax.bar(x + i * width - width, 100 * validity[algo], width, label=algo, color=colors[i], alpha=0.7)

    ax.set_title("Feasibility Rate by Fitness Function", fontsize=12, fontweight='bold')
    ax.set_ylabel("Valid Solutions (%)", fontsize=11)
    ax.set_xlabel("Fitness Function", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(["Size Min", "Penalty", "Edge Coverage"])
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

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
        ax.bar(x + i * width - width, stats[algo], width, label=algo, color=colors[i], alpha=0.7)

    ax.set_title("Instance-Level Average Cover Size (Valid Only)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cover Size", fontsize=11)
    ax.set_xlabel("Instance Type", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(["Small", "Medium", "Large", "Scale-Free"])
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    _save_fig(fig, "instance_performance.png")


def main(results_dir: str = None, output_dir: str = None):
    global RESULTS_PATH, FIG_DIR
    
    if results_dir:
        results_path = Path(results_dir) / "results.csv"
    else:
        results_path = DEFAULT_RESULTS_PATH
    
    if output_dir:
        fig_dir = Path(output_dir)
    elif results_dir:
        fig_dir = Path(results_dir).parent / "report" / "figures"
    else:
        fig_dir = DEFAULT_FIG_DIR
    
    RESULTS_PATH = results_path
    FIG_DIR = fig_dir
    
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")

    print(f"\nUsing results from: {RESULTS_PATH}")
    df = _read_results()
    plot_algorithm_quality(df)
    plot_algorithm_time(df)
    plot_encoding_quality(df)
    plot_fitness_validity(df)
    plot_instance_performance(df)
    print(f"\nAll plots saved to: {FIG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate standard analysis plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 generate_plots.py                 # Use default results/results.csv
  python3 generate_plots.py --results ./custom_results   # Use custom folder
  python3 generate_plots.py --output ./my_plots  # Save plots to custom location
  python3 generate_plots.py --results ./exp1 --output ./plots/exp1
        """
    )
    parser.add_argument('--results', type=str, default=None,
                       help='Path to results folder (containing results.csv)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output directory for plots')
    
    args = parser.parse_args()
    main(results_dir=args.results, output_dir=args.output)
