"""
Advanced Visualization Suite for MVC Meta-Heuristic Experiments
Generates comprehensive plots for analysis and publication.

Usage:
    python3 generate_advanced_plots.py                              # Default: results/
    python3 generate_advanced_plots.py --results ./my_results       # Custom results folder
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = ROOT / "results" / "results.csv"
DEFAULT_FIG_DIR = ROOT / "report" / "figures"

# These will be set by main()
RESULTS_PATH = DEFAULT_RESULTS_PATH
FIG_DIR = DEFAULT_FIG_DIR


def _read_results() -> pd.DataFrame:
    """Read and clean results CSV."""
    df = pd.read_csv(RESULTS_PATH)
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


def plot_1_performance_profile(df: pd.DataFrame):
    """Plot 1: Performance profiles (cumulative distribution)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    valid = df[df["is_valid"]].copy()
    
    for algo in ["GA", "SA", "TS"]:
        algo_data = sorted(valid[valid["algorithm"] == algo]["cover_size"].values)
        if len(algo_data) == 0:
            continue
        
        # Cumulative distribution
        cumulative = np.arange(1, len(algo_data) + 1) / len(algo_data)
        ax.plot(algo_data, cumulative, marker='o', linewidth=2, markersize=4, 
               label=f"{algo} (n={len(algo_data)})", alpha=0.7)
    
    ax.set_xlabel("Cover Size (Valid Solutions Only)", fontsize=11)
    ax.set_ylabel("Cumulative Probability", fontsize=11)
    ax.set_title("Performance Profiles: Solution Quality Distribution", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    _save_fig(fig, "02_performance_profiles.png")


def plot_2_heatmap_algorithm_encoding(df: pd.DataFrame):
    """Plot 2: Heatmap of algorithm × encoding performance."""
    valid = df[df["is_valid"]].copy()
    
    # Create pivot table
    pivot = valid.pivot_table(
        values='cover_size',
        index='encoding',
        columns='algorithm',
        aggfunc=['mean', 'count']
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Mean cover size heatmap
    mean_data = pivot['mean'].reindex(
        ["BinaryEncoding", "SetEncoding", "EdgeCentricEncoding"],
        fill_value=np.nan
    )
    sns.heatmap(mean_data, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax1, 
               cbar_kws={'label': 'Avg Cover Size'}, vmin=10, vmax=45)
    ax1.set_title("Mean Cover Size (Valid Only)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Encoding")
    ax1.set_xlabel("Algorithm")
    
    # Sample count heatmap
    count_data = pivot['count'].reindex(
        ["BinaryEncoding", "SetEncoding", "EdgeCentricEncoding"],
        fill_value=0
    )
    sns.heatmap(count_data, annot=True, fmt='.0f', cmap='Blues', ax=ax2,
               cbar_kws={'label': 'Valid Solutions'})
    ax2.set_title("Number of Valid Solutions", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Encoding")
    ax2.set_xlabel("Algorithm")
    
    _save_fig(fig, "03_heatmap_algorithm_encoding.png")


def plot_3_convergence_curves(df: pd.DataFrame):
    """Plot 3: Convergence trends across instance sizes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    instances = ["small_20nodes", "medium_50nodes", "large_100nodes", "scale_free_50nodes"]
    colors = {'GA': '#4C78A8', 'SA': '#F58518', 'TS': '#54A24B'}
    
    for idx, instance in enumerate(instances):
        ax = axes[idx]
        inst_data = df[df["instance"] == instance]
        
        for algo in ["GA", "SA", "TS"]:
            algo_data = inst_data[inst_data["algorithm"] == algo]
            
            # Group by run and compute average
            valid_by_run = algo_data.groupby("run")["is_valid"].sum()
            ax.plot(valid_by_run.index, valid_by_run.values, marker='o', 
                   label=algo, color=colors[algo], linewidth=2, markersize=6)
        
        ax.set_xlabel("Run Number", fontsize=10)
        ax.set_ylabel("Valid Solutions Found", fontsize=10)
        ax.set_title(instance.replace("_", " ").title(), fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Convergence: Valid Solution Count per Instance", 
                fontsize=13, fontweight='bold', y=1.00)
    _save_fig(fig, "04_convergence_by_instance.png")


def plot_4_box_plot_distributions(df: pd.DataFrame):
    """Plot 4: Box plots of cover size distributions."""
    valid = df[df["is_valid"]].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # By algorithm
    ax = axes[0]
    data_by_algo = [valid[valid["algorithm"] == algo]["cover_size"].values 
                   for algo in ["GA", "SA", "TS"]]
    bp1 = ax.boxplot(data_by_algo, labels=["GA", "SA", "TS"], patch_artist=True)
    for patch, color in zip(bp1['boxes'], ['#4C78A8', '#F58518', '#54A24B']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Cover Size", fontsize=11)
    ax.set_title("By Algorithm", fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # By encoding
    ax = axes[1]
    encodings = ["BinaryEncoding", "SetEncoding", "EdgeCentricEncoding"]
    data_by_enc = [valid[valid["encoding"] == enc]["cover_size"].values for enc in encodings]
    bp2 = ax.boxplot(data_by_enc, labels=["Binary", "Set", "Edge-Centric"], patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('#E8C1A0')
        patch.set_alpha(0.6)
    ax.set_ylabel("Cover Size", fontsize=11)
    ax.set_title("By Encoding", fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # By fitness function
    ax = axes[2]
    fitness_funcs = ["CoverSizeMinimization", "ConstraintPenalty", "EdgeCoverageOptimization"]
    data_by_fitness = [valid[valid["fitness_func"] == ff]["cover_size"].values for ff in fitness_funcs]
    bp3 = ax.boxplot(data_by_fitness, labels=["Size Min", "Penalty", "Edge Cov"], patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('#B0B0B0')
        patch.set_alpha(0.6)
    ax.set_ylabel("Cover Size", fontsize=11)
    ax.set_title("By Fitness Function", fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    _save_fig(fig, "05_box_plot_distributions.png")


def plot_5_time_vs_quality(df: pd.DataFrame):
    """Plot 5: Runtime vs solution quality (Pareto front visualization)."""
    valid = df[df["is_valid"]].copy()
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    colors_map = {'GA': '#4C78A8', 'SA': '#F58518', 'TS': '#54A24B'}
    
    for algo in ["GA", "SA", "TS"]:
        algo_data = valid[valid["algorithm"] == algo]
        
        # Average by configuration
        grouped = algo_data.groupby(['encoding', 'fitness_func']).agg({
            'cover_size': 'mean',
            'time_sec': 'mean'
        }).reset_index()
        
        ax.scatter(grouped['time_sec'], grouped['cover_size'], 
                  s=150, alpha=0.6, label=algo, color=colors_map[algo], edgecolors='black', linewidth=1)
    
    ax.set_xlabel("Average Runtime (seconds, log scale)", fontsize=11)
    ax.set_ylabel("Average Cover Size", fontsize=11)
    ax.set_title("Runtime vs Solution Quality Trade-off", fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    _save_fig(fig, "06_time_vs_quality.png")


def plot_6_fitness_heatmap(df: pd.DataFrame):
    """Plot 6: Heatmap of fitness function impact."""
    valid = df[df["is_valid"]].copy()
    
    pivot = valid.pivot_table(
        values='cover_size',
        index='fitness_func',
        columns='algorithm',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='coolwarm', ax=ax,
               cbar_kws={'label': 'Avg Cover Size (Valid Only)'})
    
    ax.set_title("Fitness Function Impact on Solution Quality", fontsize=12, fontweight='bold')
    ax.set_ylabel("Fitness Function")
    ax.set_xlabel("Algorithm")
    
    _save_fig(fig, "07_fitness_function_heatmap.png")


def plot_7_feasibility_comparison(df: pd.DataFrame):
    """Plot 7: Feasibility rates comparison."""
    feasibility = df.groupby('algorithm')['is_valid'].agg(['sum', 'count'])
    feasibility['rate'] = feasibility['sum'] / feasibility['count']
    feasibility = feasibility.reindex(['GA', 'SA', 'TS'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Bar plot
    colors = ['#4C78A8', '#F58518', '#54A24B']
    bars = ax1.bar(feasibility.index, feasibility['rate'] * 100, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel("Feasibility Rate (%)", fontsize=11)
    ax1.set_title("Valid Solution Rate by Algorithm", fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.grid(axis='y', alpha=0.3)
    
    # Detailed breakdown by encoding
    feasibility_by_enc = df.pivot_table(
        values='is_valid',
        index='encoding',
        columns='algorithm',
        aggfunc=lambda x: (x.sum() / len(x) * 100)
    )
    
    feasibility_by_enc = feasibility_by_enc.reindex(
        ["BinaryEncoding", "SetEncoding", "EdgeCentricEncoding"]
    )
    
    x = np.arange(len(feasibility_by_enc.index))
    width = 0.25
    
    for i, algo in enumerate(['GA', 'SA', 'TS']):
        ax2.bar(x + i*width, feasibility_by_enc[algo], width, label=algo, 
               color=colors[i], alpha=0.7, edgecolor='black')
    
    ax2.set_ylabel("Feasibility Rate (%)", fontsize=11)
    ax2.set_xlabel("Encoding")
    ax2.set_title("Feasibility by Encoding and Algorithm", fontsize=11, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(["Binary", "Set", "Edge-Centric"])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    _save_fig(fig, "08_feasibility_comparison.png")


def plot_8_instance_difficulty(df: pd.DataFrame):
    """Plot 8: Instance difficulty analysis."""
    valid = df[df["is_valid"]].copy()
    
    instance_stats = valid.groupby('instance').agg({
        'cover_size': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    
    instances = ["small_20nodes", "medium_50nodes", "large_100nodes", "scale_free_50nodes"]
    instance_stats = instance_stats.reindex(instances)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean cover size with error bars
    means = instance_stats[('cover_size', 'mean')]
    stds = instance_stats[('cover_size', 'std')]
    
    ax1.errorbar(range(len(means)), means, yerr=stds, marker='o', capsize=5, 
                linewidth=2, markersize=8, color='#4C78A8', ecolor='#F58518')
    ax1.set_xticks(range(len(means)))
    ax1.set_xticklabels([inst.replace("_", " ").title() for inst in instances], rotation=15)
    ax1.set_ylabel("Average Cover Size", fontsize=11)
    ax1.set_title("Solution Quality by Instance", fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Sample count
    counts = instance_stats[('cover_size', 'count')]
    bars = ax2.bar(range(len(counts)), counts, color=['#54A24B', '#E8C1A0', '#F58518', '#4C78A8'], 
                  alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(counts)))
    ax2.set_xticklabels([inst.replace("_", " ").title() for inst in instances], rotation=15)
    ax2.set_ylabel("Number of Valid Solutions", fontsize=11)
    ax2.set_title("Instance Difficulty (fewer solutions = harder)", fontsize=11, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    ax2.grid(axis='y', alpha=0.3)
    
    _save_fig(fig, "09_instance_difficulty.png")


def plot_9_validity_breakdown(df: pd.DataFrame):
    """Plot 9: Valid solution rate breakdown by algorithm/encoding/fitness."""
    # Validity rate (%) by algorithm x encoding
    algo_enc = df.pivot_table(
        values='is_valid',
        index='encoding',
        columns='algorithm',
        aggfunc='mean'
    ) * 100

    algo_enc = algo_enc.reindex(
        ["BinaryEncoding", "SetEncoding", "EdgeCentricEncoding"]
    )
    algo_enc = algo_enc.reindex(columns=["GA", "SA", "TS"])

    # Validity rate (%) by algorithm x fitness
    algo_fit = df.pivot_table(
        values='is_valid',
        index='fitness_func',
        columns='algorithm',
        aggfunc='mean'
    ) * 100

    algo_fit = algo_fit.reindex(
        ["CoverSizeMinimization", "ConstraintPenalty", "EdgeCoverageOptimization"]
    )
    algo_fit = algo_fit.reindex(columns=["GA", "SA", "TS"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(algo_enc, annot=True, fmt='.1f', cmap='Blues', ax=ax1,
                cbar_kws={'label': 'Valid Rate (%)'})
    ax1.set_title("Validity Rate: Encoding × Algorithm", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Encoding")

    sns.heatmap(algo_fit, annot=True, fmt='.1f', cmap='Purples', ax=ax2,
                cbar_kws={'label': 'Valid Rate (%)'})
    ax2.set_title("Validity Rate: Fitness × Algorithm", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Algorithm")
    ax2.set_ylabel("Fitness Function")

    _save_fig(fig, "10_validity_breakdown.png")


def generate_summary_table(df: pd.DataFrame):
    """Generate summary statistics table."""
    valid = df[df["is_valid"]].copy()
    
    summary = []
    for algo in ["GA", "SA", "TS"]:
        algo_data = valid[valid["algorithm"] == algo]["cover_size"]
        if len(algo_data) > 0:
            summary.append({
                'Algorithm': algo,
                'Valid': len(algo_data),
                'Mean': f"{algo_data.mean():.2f}",
                'Std': f"{algo_data.std():.2f}",
                'Min': int(algo_data.min()),
                'Max': int(algo_data.max()),
            })
    
    summary_df = pd.DataFrame(summary)
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (Valid Solutions Only)")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70 + "\n")


def main(results_dir: str = None, output_dir: str = None):
    """Generate all advanced plots.
    
    Args:
        results_dir: Custom path to results folder (containing results.csv)
        output_dir: Custom path to output directory for plots
    """
    global RESULTS_PATH, FIG_DIR
    
    # Set results path
    if results_dir:
        RESULTS_PATH = Path(results_dir) / "results.csv"
    
    # Set figure directory
    if output_dir:
        fig_dir = Path(output_dir)
    elif results_dir:
        fig_dir = Path(results_dir).parent / "report" / "figures"
    else:
        fig_dir = DEFAULT_FIG_DIR
    
    FIG_DIR = fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    if not RESULTS_PATH.exists():
        print(f"Error: Results file not found: {RESULTS_PATH}")
        return
    
    print("Loading results...")
    df = _read_results()
    print(f"Loaded {len(df)} result entries")
    
    print("\nGenerating plots...")
    plot_1_performance_profile(df)
    plot_2_heatmap_algorithm_encoding(df)
    plot_3_convergence_curves(df)
    plot_4_box_plot_distributions(df)
    plot_5_time_vs_quality(df)
    plot_6_fitness_heatmap(df)
    plot_7_feasibility_comparison(df)
    plot_8_instance_difficulty(df)
    plot_9_validity_breakdown(df)
    
    generate_summary_table(df)
    
    print(f"\nAll plots saved to: {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate advanced analysis plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 generate_advanced_plots.py                 # Use default results/results.csv
  python3 generate_advanced_plots.py --results ./custom_results   # Use custom folder
  python3 generate_advanced_plots.py --output ./my_plots  # Save plots to custom location
  python3 generate_advanced_plots.py --results ./exp1 --output ./plots/exp1
        """
    )
    parser.add_argument('--results', type=str, default=None,
                       help='Path to results folder (containing results.csv)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output directory for plots')
    
    args = parser.parse_args()
    main(results_dir=args.results, output_dir=args.output)
