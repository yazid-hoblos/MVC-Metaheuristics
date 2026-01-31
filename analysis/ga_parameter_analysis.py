"""
Genetic Algorithm Parameter Analysis Script

Tests GA with Binary Encoding across:
- Different fitness functions (3)
- Different GA parameters (population size, mutation rate, generations)
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
from ga import GeneticAlgorithm, GAParams
from mvc_fitness import FitnessFunctionFactory
from mvc_encodings import BinaryEncoding
import networkx as nx


# Output directory for results
OUTPUT_DIR = Path(__file__).parent / "ga_analysis_results"
OUTPUT_DIR.mkdir(exist_ok=True)


class GAParameterAnalyzer:
    """Comprehensive GA parameter analysis."""
    
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
        """Test different fitness functions with standard GA parameters."""
        print(f"\n{'='*80}")
        print(f"Testing Fitness Functions on {problem.num_nodes} nodes")
        print(f"{'='*80}")
        
        fitness_functions = {
            'CoverSizeMin': FitnessFunctionFactory.create_cover_size_minimization(problem),
            'ConstraintPenalty': FitnessFunctionFactory.create_constraint_penalty(problem),
            'EdgeCoverage': FitnessFunctionFactory.create_edge_coverage_optimization(problem),
        }
        
        # Standard GA parameters
        params = GAParams(
            population_size=100,
            generations=300,
            mutation_rate=0.1
        )
        
        print(f"\n{'Fitness Function':<25} {'Valid %':<12} {'Avg Size':<12} {'Time (s)':<12}")
        print("-" * 80)
        
        for func_name, fitness_func in fitness_functions.items():
            valid_count = 0
            valid_sizes = []
            times = []
            
            for run in range(self.num_runs):
                encoding = BinaryEncoding()
                ga = GeneticAlgorithm(problem, encoding, fitness_func, params)
                
                start_time = time.time()
                result = ga.run()
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
                    'population_size': params.population_size,
                    'generations': params.generations,
                    'mutation_rate': params.mutation_rate
                })
            
            validity = (valid_count / self.num_runs) * 100
            avg_size = np.mean(valid_sizes) if valid_sizes else 0
            avg_time = np.mean(times)
            
            print(f"{func_name:<25} {validity:>10.1f}% {avg_size:>10.2f} {avg_time:>10.3f}s")
    
    def test_population_size_variations(self, problem: MinimumVertexCoverProblem, instance_name: str):
        """Test different population sizes."""
        print(f"\n{'='*80}")
        print(f"Testing Population Size Variations on {problem.num_nodes} nodes")
        print(f"{'='*80}")
        
        fitness_func = FitnessFunctionFactory.create_cover_size_minimization(problem)
        pop_sizes = [50, 100, 200, 300]
        
        print(f"\n{'Pop Size':<15} {'Valid %':<12} {'Avg Size':<12} {'Time (s)':<12}")
        print("-" * 80)
        
        for pop_size in pop_sizes:
            params = GAParams(
                population_size=pop_size,
                generations=300,
                mutation_rate=0.1
            )
            
            valid_count = 0
            valid_sizes = []
            times = []
            
            for run in range(self.num_runs):
                encoding = BinaryEncoding()
                ga = GeneticAlgorithm(problem, encoding, fitness_func, params)
                
                start_time = time.time()
                result = ga.run()
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
                    'population_size': pop_size,
                    'generations': params.generations,
                    'mutation_rate': params.mutation_rate
                })
            
            validity = (valid_count / self.num_runs) * 100
            avg_size = np.mean(valid_sizes) if valid_sizes else 0
            avg_time = np.mean(times)
            
            print(f"{pop_size:<15} {validity:>10.1f}% {avg_size:>10.2f} {avg_time:>10.3f}s")
    
    def test_generations_variations(self, problem: MinimumVertexCoverProblem, instance_name: str):
        """Test different generation counts."""
        print(f"\n{'='*80}")
        print(f"Testing Generation Count Variations on {problem.num_nodes} nodes")
        print(f"{'='*80}")
        
        fitness_func = FitnessFunctionFactory.create_cover_size_minimization(problem)
        generations_list = [100, 200, 300, 500]
        
        print(f"\n{'Generations':<15} {'Valid %':<12} {'Avg Size':<12} {'Time (s)':<12}")
        print("-" * 80)
        
        for gens in generations_list:
            params = GAParams(
                population_size=100,
                generations=gens,
                mutation_rate=0.1
            )
            
            valid_count = 0
            valid_sizes = []
            times = []
            
            for run in range(self.num_runs):
                encoding = BinaryEncoding()
                ga = GeneticAlgorithm(problem, encoding, fitness_func, params)
                
                start_time = time.time()
                result = ga.run()
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
                    'population_size': params.population_size,
                    'generations': gens,
                    'mutation_rate': params.mutation_rate
                })
            
            validity = (valid_count / self.num_runs) * 100
            avg_size = np.mean(valid_sizes) if valid_sizes else 0
            avg_time = np.mean(times)
            
            print(f"{gens:<15} {validity:>10.1f}% {avg_size:>10.2f} {avg_time:>10.3f}s")
    
    def test_mutation_rate_variations(self, problem: MinimumVertexCoverProblem, instance_name: str):
        """Test different mutation rates."""
        print(f"\n{'='*80}")
        print(f"Testing Mutation Rate Variations on {problem.num_nodes} nodes")
        print(f"{'='*80}")
        
        fitness_func = FitnessFunctionFactory.create_cover_size_minimization(problem)
        mut_rates = [0.05, 0.1, 0.15, 0.2]
        
        print(f"\n{'Mutation Rate':<15} {'Valid %':<12} {'Avg Size':<12} {'Time (s)':<12}")
        print("-" * 80)
        
        for mut_rate in mut_rates:
            params = GAParams(
                population_size=100,
                generations=300,
                mutation_rate=mut_rate
            )
            
            valid_count = 0
            valid_sizes = []
            times = []
            
            for run in range(self.num_runs):
                encoding = BinaryEncoding()
                ga = GeneticAlgorithm(problem, encoding, fitness_func, params)
                
                start_time = time.time()
                result = ga.run()
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
                    'population_size': params.population_size,
                    'generations': params.generations,
                    'mutation_rate': mut_rate
                })
            
            validity = (valid_count / self.num_runs) * 100
            avg_size = np.mean(valid_sizes) if valid_sizes else 0
            avg_time = np.mean(times)
            
            print(f"{mut_rate:<15.2f} {validity:>10.1f}% {avg_size:>10.2f} {avg_time:>10.3f}s")
    
    def save_results(self):
        """Save results to CSV."""
        df = pd.DataFrame(self.results)
        csv_path = OUTPUT_DIR / "ga_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        return df
    
    def generate_plots(self, df: pd.DataFrame):
        """Generate comprehensive analysis plots."""
        print(f"\nGenerating plots...")
        
        # Plot 1: Fitness Function Comparison
        self._plot_fitness_comparison(df)
        
        # Plot 2: Population Size Impact
        self._plot_population_impact(df)
        
        # Plot 3: Generations Impact
        self._plot_generations_impact(df)
        
        # Plot 4: Mutation Rate Impact
        self._plot_mutation_rate_impact(df)
        
        # Plot 5: Combined Parameter Heatmap
        self._plot_parameter_heatmap(df)
        
        print(f"All plots saved to: {OUTPUT_DIR}")
    
    def _plot_fitness_comparison(self, df: pd.DataFrame):
        """Plot fitness function comparison."""
        # Filter standard params only
        standard_df = df[(df['population_size'] == 100) & 
                        (df['generations'] == 300) &
                        (df['mutation_rate'] == 0.1)]
        
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
    
    def _plot_population_impact(self, df: pd.DataFrame):
        """Plot population size parameter impact."""
        pop_df = df[df['generations'] == 300]
        if len(pop_df) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Validity by population size
        valid_stats = pop_df.groupby('population_size')['valid'].mean() * 100
        ax1.plot(valid_stats.index, valid_stats.values, marker='o', linewidth=2, 
                markersize=8, color='#4C78A8')
        ax1.set_title('Validity Rate vs Population Size', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Valid Solutions (%)', fontsize=11)
        ax1.set_xlabel('Population Size', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Average size by population (valid only)
        valid_only = pop_df[pop_df['valid'] == True]
        if len(valid_only) > 0:
            size_stats = valid_only.groupby('population_size')['cover_size'].agg(['mean', 'std'])
            ax2.errorbar(size_stats.index, size_stats['mean'], yerr=size_stats['std'],
                        marker='o', linewidth=2, markersize=8, capsize=5, color='#F58518')
            ax2.set_title('Cover Size vs Population Size', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Cover Size', fontsize=11)
            ax2.set_xlabel('Population Size', fontsize=11)
            ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "02_population_impact.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_generations_impact(self, df: pd.DataFrame):
        """Plot generations parameter impact."""
        gen_df = df[df['population_size'] == 100]
        if len(gen_df) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Validity by generations
        valid_stats = gen_df.groupby('generations')['valid'].mean() * 100
        ax1.plot(valid_stats.index, valid_stats.values, marker='o', linewidth=2,
                markersize=8, color='#54A24B')
        ax1.set_title('Validity Rate vs Number of Generations', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Valid Solutions (%)', fontsize=11)
        ax1.set_xlabel('Generations', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Average time by generations
        time_stats = gen_df.groupby('generations')['time'].agg(['mean', 'std'])
        ax2.errorbar(time_stats.index, time_stats['mean'], yerr=time_stats['std'],
                    marker='o', linewidth=2, markersize=8, capsize=5, color='#E45756')
        ax2.set_title('Runtime vs Number of Generations', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Time (seconds)', fontsize=11)
        ax2.set_xlabel('Generations', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "03_generations_impact.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_mutation_rate_impact(self, df: pd.DataFrame):
        """Plot mutation rate parameter impact."""
        mut_df = df[(df['population_size'] == 100) & (df['generations'] == 300)]
        if len(mut_df) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Validity by mutation rate
        valid_stats = mut_df.groupby('mutation_rate')['valid'].mean() * 100
        ax1.plot(valid_stats.index, valid_stats.values, marker='o', linewidth=2,
                markersize=8, color='#B07AA1')
        ax1.set_title('Validity Rate vs Mutation Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Valid Solutions (%)', fontsize=11)
        ax1.set_xlabel('Mutation Rate', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Time vs quality trade-off
        valid_only = mut_df[mut_df['valid'] == True]
        if len(valid_only) > 0:
            scatter_data = valid_only.groupby('mutation_rate').agg({
                'time': 'mean',
                'cover_size': 'mean'
            })
            ax2.scatter(scatter_data['time'], scatter_data['cover_size'], 
                       s=200, alpha=0.6, color='#FF9DA7')
            for idx, row in scatter_data.iterrows():
                ax2.annotate(f'{idx:.2f}', (row['time'], row['cover_size']),
                           fontsize=9, ha='center')
            ax2.set_title('Time vs Quality Trade-off', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Time (seconds)', fontsize=11)
            ax2.set_ylabel('Cover Size', fontsize=11)
            ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "04_mutation_rate_impact.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_parameter_heatmap(self, df: pd.DataFrame):
        """Plot parameter combination heatmap."""
        # Filter for 20 node instance for clarity
        df_20 = df[df['instance'] == '20_nodes']
        if len(df_20) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap 1: Population Size vs Generations (validity)
        pivot1 = df_20.pivot_table(
            values='valid',
            index='population_size',
            columns='generations',
            aggfunc='mean'
        ) * 100
        
        sns.heatmap(pivot1, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[0],
                   cbar_kws={'label': 'Valid %'}, vmin=0, vmax=100)
        axes[0].set_title('Validity: Population Size × Generations', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Generations', fontsize=11)
        axes[0].set_ylabel('Population Size', fontsize=11)
        
        # Heatmap 2: Generations vs Mutation Rate (cover size for valid)
        valid_df = df_20[df_20['valid'] == True]
        if len(valid_df) > 0:
            pivot2 = valid_df.pivot_table(
                values='cover_size',
                index='generations',
                columns='mutation_rate',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=axes[1],
                       cbar_kws={'label': 'Avg Cover Size'})
            axes[1].set_title('Cover Size: Generations × Mutation Rate', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Mutation Rate', fontsize=11)
            axes[1].set_ylabel('Generations', fontsize=11)
        
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "05_parameter_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def run_analysis(self):
        """Run full GA parameter analysis."""
        print("\n" + "="*80)
        print("GENETIC ALGORITHM PARAMETER ANALYSIS")
        print("="*80)
        
        for instance_name, num_nodes, edge_prob in self.instances:
            print(f"\n\nAnalyzing instance: {instance_name}")
            problem = self.generate_instance(num_nodes, edge_prob)
            print(f"Nodes: {problem.num_nodes}, Edges: {problem.num_edges}")
            
            self.test_fitness_functions(problem, instance_name)
            self.test_population_size_variations(problem, instance_name)
            self.test_generations_variations(problem, instance_name)
            self.test_mutation_rate_variations(problem, instance_name)
        
        df = self.save_results()
        self.generate_plots(df)
        
        print(f"\n{'='*80}")
        print("Analysis complete!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    analyzer = GAParameterAnalyzer(num_runs=5)
    analyzer.run_analysis()
