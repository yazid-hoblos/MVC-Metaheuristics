"""
Stratified and Normalized Analysis for MVC Meta-Heuristic Experiments

Addresses interpretability challenges by:
1. Stratifying results by instance type/size
2. Computing normalized metrics (approximation ratios, relative performance)
3. Generating separate plots for each instance class
4. Avoiding meaningless cross-instance averaging
5. Analyzing fitness function performance across algorithms and instances

Generates 11 plots:
- Plots 1-8: Instance-stratified analysis (quality, validity, encodings)
- Plots 9-11: Fitness function analysis (comparison, validity, heatmaps)

Usage:
    python3 generate_stratified_plots.py                              # Default: results/results.csv
    python3 generate_stratified_plots.py --results ./my_results.csv  # Custom CSV file
    python3 generate_stratified_plots.py --output ./my_plots          # Custom output location
    python3 generate_stratified_plots.py --results ./exp1.csv --output ./plots/exp1
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
DEFAULT_FIG_DIR = ROOT / "report" / "stratified_analysis"

# These will be set by main()
RESULTS_PATH = DEFAULT_RESULTS_PATH
FIG_DIR = DEFAULT_FIG_DIR


def _read_results() -> pd.DataFrame:
    """Read and clean results CSV."""
    df = pd.read_csv(RESULTS_PATH)
    df["is_valid"] = df["is_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def _map_fitness_name(fitness_value):
    """Convert fitness function identifier to readable name."""
    fitness_str = str(fitness_value)
    
    # Map known fitness function names
    if 'CoverSizeMinimization' in fitness_str:
        return 'Cover Size Min'
    elif 'ConstraintPenalty' in fitness_str:
        return 'Constraint Penalty'
    elif 'EdgeCoverageOptimization' in fitness_str:
        return 'Edge Coverage'
    elif 'standard' in fitness_str.lower():
        return 'Standard'
    elif 'penalty' in fitness_str.lower():
        return 'Penalty'
    elif 'adaptive' in fitness_str.lower():
        return 'Adaptive'
    else:
        # Return original value if no match
        return fitness_str


def _save_fig(fig, filename: str):
    """Save figure with high DPI."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def _extract_instance_info(df: pd.DataFrame) -> pd.DataFrame:
    """Extract instance size and type from instance column."""
    def parse_instance(instance_name):
        if "small" in instance_name or "20" in instance_name:
            return 20, "Random"
        elif "medium" in instance_name or "50" in instance_name:
            if "scale_free" in instance_name:
                return 50, "Scale-Free"
            else:
                return 50, "Random"
        elif "large" in instance_name or "100" in instance_name:
            return 100, "Random"
        else:
            return None, None
    
    df[['instance_size', 'instance_type']] = df['instance'].apply(
        lambda x: pd.Series(parse_instance(x))
    )
    return df


def _compute_approximation_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute approximation ratios.
    For each valid solution, compute ratio to best found solution on that instance.
    """
    result_list = []
    
    for instance in df['instance'].unique():
        inst_data = df[df['instance'] == instance].copy()
        valid_data = inst_data[inst_data['is_valid'] == True]
        
        if len(valid_data) == 0:
            continue
        
        # Best valid solution on this instance
        best_size = valid_data['cover_size'].min()
        
        # For each algorithm, compute its avg solution quality relative to best
        for algo in inst_data['algorithm'].unique():
            algo_valid = valid_data[valid_data['algorithm'] == algo]
            
            if len(algo_valid) > 0:
                approx_ratios = algo_valid['cover_size'] / best_size
                result_list.append({
                    'instance': instance,
                    'algorithm': algo,
                    'mean_approx_ratio': approx_ratios.mean(),
                    'std_approx_ratio': approx_ratios.std(),
                    'min_approx_ratio': approx_ratios.min(),
                    'num_valid': len(algo_valid),
                    'best_cover_size': best_size
                })
    
    return pd.DataFrame(result_list)


def plot_1_stratified_quality_by_instance(df: pd.DataFrame):
    """Plot 1: Cover size quality stratified by instance size."""
    df = _extract_instance_info(df)
    valid = df[df["is_valid"]].copy()
    
    instances = sorted(df['instance'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, instance in enumerate(instances[:4]):
        ax = axes[idx]
        inst_data = valid[valid['instance'] == instance]
        
        if len(inst_data) == 0:
            continue
        
        # Box plot by algorithm
        data_by_algo = [inst_data[inst_data['algorithm'] == algo]['cover_size'].values 
                       for algo in ['GA', 'SA', 'TS']]
        bp = ax.boxplot(data_by_algo, labels=['GA', 'SA', 'TS'], patch_artist=True)
        
        for patch, color in zip(bp['boxes'], ['#4C78A8', '#F58518', '#54A24B']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Add instance info
        inst_size = inst_data['instance_size'].iloc[0]
        inst_type = inst_data['instance_type'].iloc[0]
        ax.set_title(f"{instance.title()} ({inst_type})", fontsize=11, fontweight='bold')
        ax.set_ylabel("Cover Size", fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add sample counts
        for i, algo in enumerate(['GA', 'SA', 'TS']):
            count = len(data_by_algo[i])
            ax.text(i+1, ax.get_ylim()[1]*0.95, f"n={count}", ha='center', fontsize=8)
    
    fig.suptitle("Cover Size Distribution by Instance (Valid Solutions Only)", 
                fontsize=13, fontweight='bold', y=1.00)
    _save_fig(fig, "01_stratified_quality_by_instance.png")


def plot_2_approximation_ratios(df: pd.DataFrame):
    """Plot 2: Approximation ratios (normalized by best on each instance)."""
    df = _extract_instance_info(df)
    approx_df = _compute_approximation_ratios(df)
    
    if len(approx_df) == 0:
        return
    
    # Separate by instance
    instances = sorted(approx_df['instance'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, instance in enumerate(instances[:4]):
        ax = axes[idx]
        inst_data = approx_df[approx_df['instance'] == instance]
        
        if len(inst_data) == 0:
            continue
        
        # Bar plot with error bars
        algos = inst_data['algorithm'].values
        means = inst_data['mean_approx_ratio'].values
        stds = inst_data['std_approx_ratio'].values
        
        colors = ['#4C78A8', '#F58518', '#54A24B']
        ax.bar(range(len(algos)), means, yerr=stds, capsize=5, 
              color=colors[:len(algos)], alpha=0.7)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos)
        ax.set_ylabel("Approximation Ratio", fontsize=10)
        ax.set_title(f"{instance.title()}", fontsize=11, fontweight='bold')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Optimal')
        ax.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.02, f"{mean:.2f}", ha='center', fontsize=9)
    
    fig.suptitle("Approximation Ratios Relative to Best Found Solution (Lower is Better)", 
                fontsize=13, fontweight='bold', y=1.00)
    _save_fig(fig, "02_approximation_ratios.png")


def plot_3_validity_by_instance_and_encoding(df: pd.DataFrame):
    """Plot 3: Validity rates stratified by instance and encoding."""
    df = _extract_instance_info(df)
    instances = sorted(df['instance'].unique())[:4]
    
    # Encoding name mapping
    encoding_labels = {
        'BinaryEncoding': 'Binary',
        'SetEncoding': 'Set',
        'EdgeCentricEncoding': 'Edge-Centric'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, instance in enumerate(instances):
        ax = axes[idx]
        inst_data = df[df['instance'] == instance]
        
        # Pivot: encoding × algorithm, value = validity %
        validity = inst_data.groupby(['encoding', 'algorithm'])['is_valid'].mean().unstack('algorithm') * 100
        
        # Filter to available encodings
        available_encodings = [enc for enc in ['BinaryEncoding', 'SetEncoding', 'EdgeCentricEncoding'] 
                              if enc in validity.index]
        validity = validity.reindex(available_encodings).dropna(how='all')
        
        if len(validity) == 0:
            continue
        
        x = np.arange(len(validity.index))
        width = 0.25
        colors = ['#4C78A8', '#F58518', '#54A24B']
        
        for i, algo in enumerate(['GA', 'SA', 'TS']):
            if algo in validity.columns:
                ax.bar(x + i * width - width, validity[algo], width, label=algo, 
                      color=colors[i], alpha=0.7)
        
        ax.set_ylabel("Valid Solutions (%)", fontsize=10)
        ax.set_title(f"{instance.title()}", fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([encoding_labels.get(enc, enc) for enc in validity.index])
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])
    
    fig.suptitle("Validity Rates by Instance and Encoding", 
                fontsize=13, fontweight='bold', y=1.00)
    _save_fig(fig, "03_validity_by_instance_encoding.png")


def plot_4_algorithm_comparison_normalized(df: pd.DataFrame):
    """Plot 4: Algorithm performance normalized by instance."""
    df = _extract_instance_info(df)
    valid = df[df['is_valid']].copy()
    
    # For each instance, compute mean and std cover size per algorithm
    comparison_data = []
    
    for instance in df['instance'].unique():
        inst_data = valid[valid['instance'] == instance]
        
        for algo in ['GA', 'SA', 'TS']:
            algo_sizes = inst_data[inst_data['algorithm'] == algo]['cover_size'].values
            
            if len(algo_sizes) > 0:
                comparison_data.append({
                    'instance': instance,
                    'algorithm': algo,
                    'mean_size': algo_sizes.mean(),
                    'std_size': algo_sizes.std(),
                    'count': len(algo_sizes)
                })
    
    comp_df = pd.DataFrame(comparison_data)
    comp_df = _extract_instance_info(comp_df)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by algorithm
    instances = sorted(comp_df['instance'].unique())
    x = np.arange(len(instances))
    width = 0.25
    colors = ['#4C78A8', '#F58518', '#54A24B']
    
    for i, algo in enumerate(['GA', 'SA', 'TS']):
        algo_data = comp_df[comp_df['algorithm'] == algo]
        means = []
        stds = []
        
        for instance in instances:
            inst_algo = algo_data[algo_data['instance'] == instance]
            if len(inst_algo) > 0:
                means.append(inst_algo['mean_size'].values[0])
                stds.append(inst_algo['std_size'].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        ax.bar(x + i * width - width, means, width, yerr=stds, capsize=5,
              label=algo, color=colors[i], alpha=0.7)
    
    ax.set_ylabel("Mean Cover Size (Valid Only)", fontsize=11)
    ax.set_xlabel("Instance", fontsize=11)
    ax.set_title("Algorithm Comparison: Mean Cover Size per Instance", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([inst.title() for inst in instances], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    _save_fig(fig, "04_algorithm_comparison_by_instance.png")


def plot_5_encoding_quality_per_instance(df: pd.DataFrame):
    """Plot 5: Encoding quality normalized per instance."""
    df = _extract_instance_info(df)
    valid = df[df['is_valid']].copy()
    instances = sorted(df['instance'].unique())[:4]
    
    # Encoding name mapping
    encoding_labels = {
        'BinaryEncoding': 'Binary',
        'SetEncoding': 'Set',
        'EdgeCentricEncoding': 'Edge-Centric'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, instance in enumerate(instances):
        ax = axes[idx]
        inst_data = valid[valid['instance'] == instance]
        
        if len(inst_data) == 0:
            continue
        
        # Mean cover size by encoding
        encoding_stats = inst_data.groupby('encoding')['cover_size'].agg(['mean', 'std'])
        available_encodings = [enc for enc in ['BinaryEncoding', 'SetEncoding', 'EdgeCentricEncoding'] 
                              if enc in encoding_stats.index]
        encoding_stats = encoding_stats.reindex(available_encodings).dropna()
        
        if len(encoding_stats) == 0:
            continue
        
        ax.bar(range(len(encoding_stats)), encoding_stats['mean'], 
              yerr=encoding_stats['std'], capsize=5, 
              color=['#4C78A8', '#F58518', '#54A24B'][:len(encoding_stats)], alpha=0.7)
        ax.set_xticks(range(len(encoding_stats)))
        ax.set_xticklabels([encoding_labels.get(enc, enc) for enc in encoding_stats.index])
        ax.set_ylabel("Cover Size", fontsize=10)
        ax.set_title(f"{instance.title()}", fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle("Encoding Quality per Instance (Valid Solutions Only)", 
                fontsize=13, fontweight='bold', y=1.00)
    _save_fig(fig, "05_encoding_quality_per_instance.png")


def plot_6_validity_heatmap_per_instance(df: pd.DataFrame):
    """Plot 6: Separate heatmaps of algorithm × encoding validity per instance."""
    df = _extract_instance_info(df)
    instances = sorted(df['instance'].unique())[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, instance in enumerate(instances):
        ax = axes[idx]
        inst_data = df[df['instance'] == instance]
        
        # Create pivot: encoding × algorithm
        pivot = inst_data.pivot_table(
            values='is_valid',
            index='encoding',
            columns='algorithm',
            aggfunc='mean'
        ) * 100
        
        # Filter to available encodings
        available_encodings = [enc for enc in ['BinaryEncoding', 'SetEncoding', 'EdgeCentricEncoding'] 
                              if enc in pivot.index]
        pivot = pivot.reindex(available_encodings).dropna(how='all')
        
        if len(pivot) == 0:
            continue
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                   cbar_kws={'label': 'Validity %'}, vmin=0, vmax=100)
        ax.set_title(f"{instance.title()}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Encoding")
        ax.set_xlabel("Algorithm")
    
    fig.suptitle("Validity Rates: Algorithm × Encoding per Instance", 
                fontsize=13, fontweight='bold', y=1.00)
    _save_fig(fig, "06_validity_heatmap_per_instance.png")


def plot_7_cover_size_heatmap_per_instance(df: pd.DataFrame):
    """Plot 7: Separate heatmaps of mean cover size (algorithm × encoding) per instance."""
    df = _extract_instance_info(df)
    valid = df[df['is_valid']].copy()
    instances = sorted(df['instance'].unique())[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, instance in enumerate(instances):
        ax = axes[idx]
        inst_data = valid[valid['instance'] == instance]
        
        if len(inst_data) == 0:
            continue
        
        # Create pivot: encoding × algorithm, value = mean cover size
        pivot = inst_data.pivot_table(
            values='cover_size',
            index='encoding',
            columns='algorithm',
            aggfunc='mean'
        )
        
        # Filter to available encodings
        available_encodings = [enc for enc in ['BinaryEncoding', 'SetEncoding', 'EdgeCentricEncoding'] 
                              if enc in pivot.index]
        pivot = pivot.reindex(available_encodings).dropna(how='all')
        
        if len(pivot) == 0:
            continue
        
        # Determine min/max for consistent color scale within instance
        vmin = pivot.min().min()
        vmax = pivot.max().max()
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax,
                   cbar_kws={'label': 'Mean Cover Size'}, vmin=vmin, vmax=vmax)
        ax.set_title(f"{instance.title()}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Encoding")
        ax.set_xlabel("Algorithm")
    
    fig.suptitle("Mean Cover Size (Valid Solutions Only): Algorithm × Encoding per Instance", 
                fontsize=13, fontweight='bold', y=1.00)
    _save_fig(fig, "07_cover_size_heatmap_per_instance.png")


def plot_8_cover_size_std_heatmap_per_instance(df: pd.DataFrame):
    """Plot 8: Separate heatmaps of cover size std dev (algorithm × encoding) per instance."""
    df = _extract_instance_info(df)
    valid = df[df['is_valid']].copy()
    instances = sorted(df['instance'].unique())[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, instance in enumerate(instances):
        ax = axes[idx]
        inst_data = valid[valid['instance'] == instance]
        
        if len(inst_data) == 0:
            continue
        
        # Create pivot: encoding × algorithm, value = std dev cover size
        pivot = inst_data.pivot_table(
            values='cover_size',
            index='encoding',
            columns='algorithm',
            aggfunc='std'
        )
        
        # Filter to available encodings
        available_encodings = [enc for enc in ['BinaryEncoding', 'SetEncoding', 'EdgeCentricEncoding'] 
                              if enc in pivot.index]
        pivot = pivot.reindex(available_encodings).dropna(how='all')
        
        if len(pivot) == 0:
            continue
        
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Std Dev Cover Size'})
        ax.set_title(f"{instance.title()}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Encoding")
        ax.set_xlabel("Algorithm")
    
    fig.suptitle("Cover Size Variability (Std Dev): Algorithm × Encoding per Instance", 
                fontsize=13, fontweight='bold', y=1.00)
    _save_fig(fig, "08_cover_size_std_heatmap_per_instance.png")


def plot_9_fitness_function_comparison(df: pd.DataFrame):
    """Plot 9: Compare fitness functions across algorithms."""
    valid = df[df['is_valid'] == True].copy()
    
    if len(valid) == 0:
        print("Warning: No valid solutions for fitness function comparison")
        return
    
    # Get unique fitness functions
    if 'fitness_func' not in valid.columns:
        print("Warning: 'fitness_func' column not found in data")
        return
    
    # Map fitness function names to readable labels
    valid['fitness_label'] = valid['fitness_func'].apply(_map_fitness_name)
    fitness_funcs = sorted(valid['fitness_label'].unique())
    algorithms = sorted(valid['algorithm'].unique())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, algo in enumerate(algorithms):
        ax = axes[idx]
        algo_data = valid[valid['algorithm'] == algo]
        
        # Compute mean cover size for each fitness function
        fitness_stats = []
        for fit in fitness_funcs:
            fit_data = algo_data[algo_data['fitness_label'] == fit]['cover_size']
            if len(fit_data) > 0:
                mean_val = fit_data.mean()
                std_val = fit_data.std() if len(fit_data) > 1 else 0.0
                # Replace NaN with 0
                if pd.isna(mean_val):
                    mean_val = 0.0
                if pd.isna(std_val):
                    std_val = 0.0
                fitness_stats.append({
                    'fitness': fit,
                    'mean': mean_val,
                    'std': std_val,
                    'count': len(fit_data)
                })
        
        if not fitness_stats:
            continue
        
        fit_df = pd.DataFrame(fitness_stats)
        
        # Skip if no valid data
        if len(fit_df) == 0 or fit_df['mean'].isna().all():
            continue
        
        # Bar plot
        colors = ['#E15759', '#76B7B2', '#EDC948']
        x_pos = range(len(fit_df))
        ax.bar(x_pos, fit_df['mean'], yerr=fit_df['std'], 
              capsize=5, color=colors[:len(fit_df)], alpha=0.7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(fit_df['fitness'], rotation=45, ha='right')
        ax.set_ylabel("Mean Cover Size", fontsize=10)
        ax.set_title(f"{algo}", fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add count annotations
        for i, row in fit_df.iterrows():
            ax.text(i, row['mean'] + row['std'] + 0.5, 
                   f"n={row['count']}", ha='center', fontsize=8)
    
    fig.suptitle("Fitness Function Performance by Algorithm (Valid Solutions Only)", 
                fontsize=13, fontweight='bold', y=0.98)
    _save_fig(fig, "09_fitness_function_comparison.png")


def plot_10_fitness_validity_rates(df: pd.DataFrame):
    """Plot 10: Validity rates by fitness function."""
    if 'fitness_func' not in df.columns:
        print("Warning: 'fitness_func' column not found in data")
        return
    
    # Map fitness function names to readable labels
    df = df.copy()
    df['fitness_label'] = df['fitness_func'].apply(_map_fitness_name)
    fitness_funcs = sorted(df['fitness_label'].unique())
    algorithms = sorted(df['algorithm'].unique())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute validity rates
    validity_data = []
    for algo in algorithms:
        for fit in fitness_funcs:
            subset = df[(df['algorithm'] == algo) & (df['fitness_label'] == fit)]
            if len(subset) > 0:
                validity_rate = subset['is_valid'].sum() / len(subset) * 100
                validity_data.append({
                    'algorithm': algo,
                    'fitness': fit,
                    'validity_rate': validity_rate,
                    'total': len(subset),
                    'valid': subset['is_valid'].sum()
                })
    
    validity_df = pd.DataFrame(validity_data)
    
    if len(validity_df) == 0:
        print("Warning: No data for fitness validity rates")
        return
    
    # Create grouped bar chart
    x = np.arange(len(fitness_funcs))
    width = 0.25
    colors = ['#4C78A8', '#F58518', '#54A24B']
    
    for idx, algo in enumerate(algorithms):
        algo_data = validity_df[validity_df['algorithm'] == algo]
        if len(algo_data) > 0:
            algo_rates = []
            for f in fitness_funcs:
                match = algo_data[algo_data['fitness'] == f]['validity_rate']
                if len(match) > 0 and not pd.isna(match.values[0]):
                    algo_rates.append(match.values[0])
                else:
                    algo_rates.append(0)
            ax.bar(x + idx*width, algo_rates, width, label=algo, 
                  color=colors[idx], alpha=0.7)
    
    ax.set_ylabel('Validity Rate (%)', fontsize=11)
    ax.set_xlabel('Fitness Function', fontsize=11)
    ax.set_title('Solution Validity Rates by Fitness Function', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(fitness_funcs, rotation=45, ha='right')
    ax.legend(title='Algorithm', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    _save_fig(fig, "10_fitness_validity_rates.png")


def plot_11_fitness_vs_instance_heatmap(df: pd.DataFrame):
    """Plot 11: Heatmap of fitness function performance across instances."""
    df = _extract_instance_info(df)
    valid = df[df['is_valid'] == True].copy()
    
    if len(valid) == 0 or 'fitness_func' not in valid.columns:
        print("Warning: Insufficient data for fitness vs instance heatmap")
        return
    
    # Map fitness function names to readable labels
    valid['fitness_label'] = valid['fitness_func'].apply(_map_fitness_name)
    instances = sorted(valid['instance'].unique())
    fitness_funcs = sorted(valid['fitness_label'].unique())
    
    # Create separate heatmap for each algorithm
    algorithms = sorted(valid['algorithm'].unique())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, algo in enumerate(algorithms):
        ax = axes[idx]
        algo_data = valid[valid['algorithm'] == algo]
        
        # Create matrix: instances x fitness functions
        matrix = np.zeros((len(instances), len(fitness_funcs)))
        
        for i, inst in enumerate(instances):
            for j, fit in enumerate(fitness_funcs):
                subset = algo_data[(algo_data['instance'] == inst) & 
                                  (algo_data['fitness_label'] == fit)]
                if len(subset) > 0:
                    mean_val = subset['cover_size'].mean()
                    matrix[i, j] = mean_val if not pd.isna(mean_val) else np.nan
                else:
                    matrix[i, j] = np.nan
        
        # Skip if all NaN
        if np.isnan(matrix).all():
            continue
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
        
        ax.set_xticks(range(len(fitness_funcs)))
        ax.set_xticklabels(fitness_funcs, rotation=45, ha='right')
        ax.set_yticks(range(len(instances)))
        ax.set_yticklabels([inst.replace('_', ' ').title() for inst in instances])
        ax.set_title(f"{algo}", fontsize=11, fontweight='bold')
        
        # Add text annotations
        for i in range(len(instances)):
            for j in range(len(fitness_funcs)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=9)
        
        # Colorbar for each subplot
        plt.colorbar(im, ax=ax, label='Mean Cover Size')
    
    fig.suptitle("Fitness Function Performance Across Instances (Lower is Better)", 
                fontsize=13, fontweight='bold', y=0.98)
    _save_fig(fig, "11_fitness_vs_instance_heatmap.png")


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics stratified by instance."""
    df = _extract_instance_info(df)
    
    print("\n" + "="*80)
    print("STRATIFIED ANALYSIS SUMMARY")
    print("="*80)
    
    for instance in sorted(df['instance'].unique()):
        inst_data = df[df['instance'] == instance]
        inst_size = inst_data['instance_size'].iloc[0]
        inst_type = inst_data['instance_type'].iloc[0]
        
        print(f"\n{instance.upper()} ({inst_size} nodes, {inst_type})")
        print("-" * 80)
        
        # Overall validity
        total_valid = inst_data['is_valid'].sum()
        total_runs = len(inst_data)
        print(f"Overall validity: {total_valid}/{total_runs} ({100*total_valid/total_runs:.1f}%)")
        
        # By algorithm
        print("\nBy Algorithm:")
        for algo in ['GA', 'SA', 'TS']:
            algo_data = inst_data[inst_data['algorithm'] == algo]
            if len(algo_data) == 0:
                continue
            
            valid_count = algo_data['is_valid'].sum()
            valid_pct = 100 * valid_count / len(algo_data)
            
            valid_only = algo_data[algo_data['is_valid']]
            if len(valid_only) > 0:
                mean_size = valid_only['cover_size'].mean()
                print(f"  {algo:3s}: {valid_count:3d}/{len(algo_data):3d} valid ({valid_pct:5.1f}%) | Avg size: {mean_size:6.2f}")
            else:
                print(f"  {algo:3s}: {valid_count:3d}/{len(algo_data):3d} valid ({valid_pct:5.1f}%)")
        
        # By encoding
        print("\nBy Encoding:")
        for encoding in ['BinaryEncoding', 'SetEncoding', 'EdgeCentricEncoding']:
            enc_data = inst_data[inst_data['encoding'] == encoding]
            if len(enc_data) == 0:
                continue
            
            valid_count = enc_data['is_valid'].sum()
            valid_pct = 100 * valid_count / len(enc_data)
            print(f"  {encoding:20s}: {valid_count:3d}/{len(enc_data):3d} valid ({valid_pct:5.1f}%)")


def main(results_dir: str = None, output_dir: str = None):
    global RESULTS_PATH, FIG_DIR
    
    if results_dir:
        results_path = Path(results_dir)
        # If it ends with .csv, use it directly; otherwise treat as folder
        if str(results_path).endswith('.csv'):
            results_path = results_path
        else:
            results_path = results_path / "results.csv"
    else:
        results_path = DEFAULT_RESULTS_PATH
    
    if output_dir:
        fig_dir = Path(output_dir)
    elif results_dir:
        fig_dir = Path(results_dir).parent / "report" / "stratified_analysis"
    else:
        fig_dir = DEFAULT_FIG_DIR
    
    RESULTS_PATH = results_path
    FIG_DIR = fig_dir
    
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")
    
    print(f"\nUsing results from: {RESULTS_PATH}")
    print("Reading results...")
    df = _read_results()
    
    print("Generating plots...")
    plot_1_stratified_quality_by_instance(df)
    plot_2_approximation_ratios(df)
    plot_3_validity_by_instance_and_encoding(df)
    plot_4_algorithm_comparison_normalized(df)
    plot_5_encoding_quality_per_instance(df)
    plot_6_validity_heatmap_per_instance(df)
    plot_7_cover_size_heatmap_per_instance(df)
    plot_8_cover_size_std_heatmap_per_instance(df)
    plot_9_fitness_function_comparison(df)
    plot_10_fitness_validity_rates(df)
    plot_11_fitness_vs_instance_heatmap(df)
    
    print_summary_statistics(df)
    
    print(f"\n{'='*80}")
    print(f"All plots saved to: {FIG_DIR}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate stratified analysis plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 generate_stratified_plots.py                 # Use default results/results.csv
  python3 generate_stratified_plots.py --results ./custom_results   # Use custom folder
  python3 generate_stratified_plots.py --output ./my_plots  # Save plots to custom location
  python3 generate_stratified_plots.py --results ./exp1 --output ./plots/exp1
        """
    )
    parser.add_argument('--results', type=str, default=None,
                       help='Path to results CSV file (or folder containing results.csv)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output directory for plots')
    
    args = parser.parse_args()
    main(results_dir=args.results, output_dir=args.output)
