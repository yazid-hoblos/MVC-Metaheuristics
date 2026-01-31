"""
Simulated Annealing Parameter Analysis Script

Tests SA with Binary Encoding across:
- Different fitness functions (3)
- Different SA parameters (temperature, cooling rate, iterations)
- Generates comprehensive plots and analysis
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from problem import MinimumVertexCoverProblem
from sa import SimulatedAnnealing, SAParams
from mvc_fitness import FitnessFunctionFactory
from mvc_encodings import BinaryEncoding
import networkx as nx


# Output directory for results
OUTPUT_DIR = Path(__file__).parent / "sa_analysis_results"
OUTPUT_DIR.mkdir(exist_ok=True)


class SAParameterAnalyzer:
    """Comprehensive SA parameter analysis."""
    
    def __init__(self, num_runs: int = 5):
        self.num_runs = num_runs
        self.results = []
        
        # Test instances
        self.instances = [
            ("20_nodes", 20, 0.3),
            ("50_nodes", 50, 0.25),
        ]
    
    def generate_instance(self, num_nodes: int, edge_prob: float) -> MinimumVertexCoverProblem:
        """Generate random graph instance."""
        graph = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=None)
        return MinimumVertexCoverProblem(graph)
    
    def test_fitness_functions(self, problem: MinimumVertexCoverProblem, instance_name: str):
        """Test different fitness functions with standard SA parameters."""
        print(f"\n{'='*80}")
        print(f"Testing Fitness Functions on {problem.num_nodes} nodes")
        print(f"{'='*80}")
        
        fitness_functions = {
            'CoverSizeMin': FitnessFunctionFactory.create_cover_size_minimization(problem),
            'ConstraintPenalty': FitnessFunctionFactory.create_constraint_penalty(problem),
            'EdgeCoverage': FitnessFunctionFactory.create_edge_coverage_optimization(problem),
        }
        
        # Standard SA parameters
        params = SAParams(
            initial_temperature=100.0,
            cooling_rate=0.95,
            iterations_per_temperature=50,
            min_temperature=0.01
        )
        
        print(f"\n{'Fitness Function':<25} {'Valid %':<12} {'Avg Size':<12} {'Time (s)':<12}")
        print("-" * 80)
        
        for func_name, fitness_func in fitness_functions.items():
            valid_count = 0
            valid_sizes = []
            times = []
            
            for run in range(self.num_runs):
                encoding = BinaryEncoding()
                sa = SimulatedAnnealing(problem, encoding, fitness_func, params)
                
                start_time = time.time()
                result = sa.run()
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                # Extract cover from result
                cover = result['best_cover'] if isinstance(result, dict) else result
                if problem.is_valid_cover(cover):
                    valid_count += 1
                    valid_sizes.append(len(cover))
                
                # Store result
                self.results.append({
                    'instance': instance_name,
                    'fitness_func': func_name,
                    'run': run,
                    'valid': problem.is_valid_cover(cover),
                    'cover_size': len(cover) if cover else 0,
                    'time': elapsed,
                    'initial_temp': params.initial_temperature,
                    'cooling_rate': params.cooling_rate,
                    'iters_per_temp': params.iterations_per_temperature
                })
            
            validity = (valid_count / self.num_runs) * 100
            avg_size = np.mean(valid_sizes) if valid_sizes else 0
            avg_time = np.mean(times)
            
            print(f"{func_name:<25} {validity:>10.1f}% {avg_size:>10.2f} {avg_time:>10.3f}s")
    
    def test_temperature_variations(self, problem: MinimumVertexCoverProblem, instance_name: str):
        """Test different initial temperatures."""
        print(f"\n{'='*80}")
        print(f"Testing Initial Temperature Variations on {problem.num_nodes} nodes")
        print(f"{'='*80}")
        
        fitness_func = FitnessFunctionFactory.create_cover_size_minimization(problem)
        temperatures = [50.0, 100.0, 200.0, 500.0]
        
        print(f"\n{'Temperature':<15} {'Valid %':<12} {'Avg Size':<12} {'Time (s)':<12}")
        print("-" * 80)
        
        for temp in temperatures:
            params = SAParams(
                initial_temperature=temp,
                cooling_rate=0.95,
                iterations_per_temperature=50,
                min_temperature=0.01
            )
            
            valid_count = 0
            valid_sizes = []
            times = []
            
            for run in range(self.num_runs):
                encoding = BinaryEncoding()
                sa = SimulatedAnnealing(problem, encoding, fitness_func, params)
                
                start_time = time.time()
                result = sa.run()
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                cover = result['best_cover'] if isinstance(result, dict) else result
                if problem.is_valid_cover(cover):
                    valid_count += 1
                    valid_sizes.append(len(cover))
                
                self.results.append({
                    'instance': instance_name,
                    'fitness_func': 'CoverSizeMin',
                    'run': run,
                    'valid': problem.is_valid_cover(cover),
                    'cover_size': len(cover) if cover else 0,
                    'time': elapsed,
                    'initial_temp': temp,
                    'cooling_rate': params.cooling_rate,
                    'iters_per_temp': params.iterations_per_temperature
                })
            
            validity = (valid_count / self.num_runs) * 100
            avg_size = np.mean(valid_sizes) if valid_sizes else 0
            avg_time = np.mean(times)
            
            print(f"{temp:<15.1f} {validity:>10.1f}% {avg_size:>10.2f} {avg_time:>10.3f}s")
    
    def test_cooling_rate_variations(self, problem: MinimumVertexCoverProblem, instance_name: str):
        """Test different cooling rates."""
        print(f"\n{'='*80}")
        print(f"Testing Cooling Rate Variations on {problem.num_nodes} nodes")
        print(f"{'='*80}")
        
        fitness_func = FitnessFunctionFactory.create_cover_size_minimization(problem)
        cooling_rates = [0.85, 0.90, 0.95, 0.98]
        
        print(f"\n{'Cooling Rate':<15} {'Valid %':<12} {'Avg Size':<12} {'Time (s)':<12}")
        print("-" * 80)
        
        for rate in cooling_rates:
            params = SAParams(
                initial_temperature=100.0,
                cooling_rate=rate,
                iterations_per_temperature=50,
                min_temperature=0.01
            )
            
            valid_count = 0
            valid_sizes = []
            times = []
            
            for run in range(self.num_runs):
                encoding = BinaryEncoding()
                sa = SimulatedAnnealing(problem, encoding, fitness_func, params)
                
                start_time = time.time()
                result = sa.run()
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                cover = result['best_cover'] if isinstance(result, dict) else result
                if problem.is_valid_cover(cover):
                    valid_count += 1
                    valid_sizes.append(len(cover))
                
                self.results.append({
                    'instance': instance_name,
                    'fitness_func': 'CoverSizeMin',
                    'run': run,
                    'valid': problem.is_valid_cover(cover),
                    'cover_size': len(cover) if cover else 0,
                    'time': elapsed,
                    'initial_temp': params.initial_temperature,
                    'cooling_rate': rate,
                    'iters_per_temp': params.iterations_per_temperature
                })
            
            validity = (valid_count / self.num_runs) * 100
            avg_size = np.mean(valid_sizes) if valid_sizes else 0
            avg_time = np.mean(times)
            
            print(f"{rate:<15.2f} {validity:>10.1f}% {avg_size:>10.2f} {avg_time:>10.3f}s")
    
    def test_iterations_variations(self, problem: MinimumVertexCoverProblem, instance_name: str):
        """Test different iterations per temperature."""
        print(f"\n{'='*80}")
        print(f"Testing Iterations per Temperature on {problem.num_nodes} nodes")
        print(f"{'='*80}")
        
        fitness_func = FitnessFunctionFactory.create_cover_size_minimization(problem)
        iterations = [25, 50, 100, 200]
        
        print(f"\n{'Iters/Temp':<15} {'Valid %':<12} {'Avg Size':<12} {'Time (s)':<12}")
        print("-" * 80)
        
        for iters in iterations:
            params = SAParams(
                initial_temperature=100.0,
                cooling_rate=0.95,
                iterations_per_temperature=iters,
                min_temperature=0.01
            )
            
            valid_count = 0
            valid_sizes = []
            times = []
            
            for run in range(self.num_runs):
                encoding = BinaryEncoding()
                sa = SimulatedAnnealing(problem, encoding, fitness_func, params)
                
                start_time = time.time()
                result = sa.run()
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                cover = result['best_cover'] if isinstance(result, dict) else result
                if problem.is_valid_cover(cover):
                    valid_count += 1
                    valid_sizes.append(len(cover))
                
                self.results.append({
                    'instance': instance_name,
                    'fitness_func': 'CoverSizeMin',
                    'run': run,
                    'valid': problem.is_valid_cover(cover),
                    'cover_size': len(cover) if cover else 0,
                    'time': elapsed,
                    'initial_temp': params.initial_temperature,
                    'cooling_rate': params.cooling_rate,
                    'iters_per_temp': iters
                })
            
            validity = (valid_count / self.num_runs) * 100
            avg_size = np.mean(valid_sizes) if valid_sizes else 0
            avg_time = np.mean(times)
            
            print(f"{iters:<15} {validity:>10.1f}% {avg_size:>10.2f} {avg_time:>10.3f}s")
    
    def save_results(self):
        """Save results to CSV."""
        df = pd.DataFrame(self.results)
        csv_path = OUTPUT_DIR / "sa_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        return df
    
    def generate_plots(self, df: pd.DataFrame):
        """Generate comprehensive analysis plots."""
        print(f"\nGenerating plots...")
        
        # Plot 1: Fitness Function Comparison
        self._plot_fitness_comparison(df)
        
        # Plot 2: Temperature Impact
        self._plot_temperature_impact(df)
        
        # Plot 3: Cooling Rate Impact
        self._plot_cooling_rate_impact(df)
        
        # Plot 4: Iterations Impact
        self._plot_iterations_impact(df)
        
        # Plot 5: Combined Parameter Heatmap
        self._plot_parameter_heatmap(df)
        
        print(f"All plots saved to: {OUTPUT_DIR}")
    
    def _plot_fitness_comparison(self, df: pd.DataFrame):
        """Plot fitness function comparison."""
        # Filter standard params only
        standard_df = df[(df['initial_temp'] == 100.0) & 
                        (df['cooling_rate'] == 0.95) &
                        (df['iters_per_temp'] == 50)]
        
        if len(standard_df) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Validity rate
        valid_stats = standard_df.groupby('fitness_func')['valid'].mean() * 100
        colors = ['#4C78A8', '#F58518', '#54A24B']
        ax1.bar(valid_stats.index, valid_stats.values, color=colors, alpha=0.7)
        ax1.set_title('Validity Rate by Fitness Function', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Valid Solutions (%)', fontsize=11)
        ax1.set_xlabel('Fitness Function', fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        
        # Average cover size (valid only)
        valid_only = standard_df[standard_df['valid'] == True]
        if len(valid_only) > 0:
            size_stats = valid_only.groupby('fitness_func')['cover_size'].agg(['mean', 'std'])
            ax2.bar(size_stats.index, size_stats['mean'], yerr=size_stats['std'], 
                   capsize=5, color=colors, alpha=0.7)
            ax2.set_title('Average Cover Size (Valid Only)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Cover Size', fontsize=11)
            ax2.set_xlabel('Fitness Function', fontsize=11)
            ax2.grid(axis='y', alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "01_fitness_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_temperature_impact(self, df: pd.DataFrame):
        """Plot temperature parameter impact."""
        temp_df = df[df['cooling_rate'] == 0.95]
        if len(temp_df) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Validity by temperature
        valid_stats = temp_df.groupby('initial_temp')['valid'].mean() * 100
        ax1.plot(valid_stats.index, valid_stats.values, marker='o', linewidth=2, 
                markersize=8, color='#4C78A8')
        ax1.set_title('Validity Rate vs Initial Temperature', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Valid Solutions (%)', fontsize=11)
        ax1.set_xlabel('Initial Temperature', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Average size by temperature (valid only)
        valid_only = temp_df[temp_df['valid'] == True]
        if len(valid_only) > 0:
            size_stats = valid_only.groupby('initial_temp')['cover_size'].agg(['mean', 'std'])
            ax2.errorbar(size_stats.index, size_stats['mean'], yerr=size_stats['std'],
                        marker='o', linewidth=2, markersize=8, capsize=5, color='#F58518')
            ax2.set_title('Cover Size vs Initial Temperature', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Cover Size', fontsize=11)
            ax2.set_xlabel('Initial Temperature', fontsize=11)
            ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "02_temperature_impact.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_cooling_rate_impact(self, df: pd.DataFrame):
        """Plot cooling rate parameter impact."""
        cool_df = df[df['initial_temp'] == 100.0]
        if len(cool_df) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Validity by cooling rate
        valid_stats = cool_df.groupby('cooling_rate')['valid'].mean() * 100
        ax1.plot(valid_stats.index, valid_stats.values, marker='o', linewidth=2,
                markersize=8, color='#54A24B')
        ax1.set_title('Validity Rate vs Cooling Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Valid Solutions (%)', fontsize=11)
        ax1.set_xlabel('Cooling Rate', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Average time by cooling rate
        time_stats = cool_df.groupby('cooling_rate')['time'].agg(['mean', 'std'])
        ax2.errorbar(time_stats.index, time_stats['mean'], yerr=time_stats['std'],
                    marker='o', linewidth=2, markersize=8, capsize=5, color='#E45756')
        ax2.set_title('Runtime vs Cooling Rate', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Time (seconds)', fontsize=11)
        ax2.set_xlabel('Cooling Rate', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "03_cooling_rate_impact.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_iterations_impact(self, df: pd.DataFrame):
        """Plot iterations per temperature impact."""
        iter_df = df[(df['initial_temp'] == 100.0) & (df['cooling_rate'] == 0.95)]
        if len(iter_df) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Validity by iterations
        valid_stats = iter_df.groupby('iters_per_temp')['valid'].mean() * 100
        ax1.plot(valid_stats.index, valid_stats.values, marker='o', linewidth=2,
                markersize=8, color='#B07AA1')
        ax1.set_title('Validity Rate vs Iterations per Temperature', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Valid Solutions (%)', fontsize=11)
        ax1.set_xlabel('Iterations per Temperature', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Time vs quality trade-off
        valid_only = iter_df[iter_df['valid'] == True]
        if len(valid_only) > 0:
            scatter_data = valid_only.groupby('iters_per_temp').agg({
                'time': 'mean',
                'cover_size': 'mean'
            })
            ax2.scatter(scatter_data['time'], scatter_data['cover_size'], 
                       s=200, alpha=0.6, color='#FF9DA7')
            for idx, row in scatter_data.iterrows():
                ax2.annotate(f'{int(idx)}', (row['time'], row['cover_size']),
                           fontsize=9, ha='center')
            ax2.set_title('Time vs Quality Trade-off', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Time (seconds)', fontsize=11)
            ax2.set_ylabel('Cover Size', fontsize=11)
            ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "04_iterations_impact.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_parameter_heatmap(self, df: pd.DataFrame):
        """Plot parameter combination heatmap."""
        # Filter for 20 node instance for clarity
        df_20 = df[df['instance'] == '20_nodes']
        if len(df_20) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap 1: Temperature vs Cooling Rate (validity)
        pivot1 = df_20.pivot_table(
            values='valid',
            index='initial_temp',
            columns='cooling_rate',
            aggfunc='mean'
        ) * 100
        
        sns.heatmap(pivot1, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[0],
                   cbar_kws={'label': 'Valid %'}, vmin=0, vmax=100)
        axes[0].set_title('Validity: Temperature × Cooling Rate', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Cooling Rate', fontsize=11)
        axes[0].set_ylabel('Initial Temperature', fontsize=11)
        
        # Heatmap 2: Cooling Rate vs Iterations (cover size for valid)
        valid_df = df_20[df_20['valid'] == True]
        if len(valid_df) > 0:
            pivot2 = valid_df.pivot_table(
                values='cover_size',
                index='cooling_rate',
                columns='iters_per_temp',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=axes[1],
                       cbar_kws={'label': 'Avg Cover Size'})
            axes[1].set_title('Cover Size: Cooling Rate × Iterations', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Iterations per Temperature', fontsize=11)
            axes[1].set_ylabel('Cooling Rate', fontsize=11)
        
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "05_parameter_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def run_analysis(self):
        """Run full SA parameter analysis."""
        print("\n" + "="*80)
        print("SIMULATED ANNEALING PARAMETER ANALYSIS")
        print("="*80)
        
        for instance_name, num_nodes, edge_prob in self.instances:
            print(f"\n\nAnalyzing instance: {instance_name}")
            problem = self.generate_instance(num_nodes, edge_prob)
            print(f"Nodes: {problem.num_nodes}, Edges: {problem.num_edges}")
            
            self.test_fitness_functions(problem, instance_name)
            self.test_temperature_variations(problem, instance_name)
            self.test_cooling_rate_variations(problem, instance_name)
            self.test_iterations_variations(problem, instance_name)
        
        df = self.save_results()
        self.generate_plots(df)
        
        print(f"\n{'='*80}")
        print("Analysis complete!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    analyzer = SAParameterAnalyzer(num_runs=5)
    analyzer.run_analysis()
