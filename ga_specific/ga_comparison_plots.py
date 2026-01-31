"""
GA Comparison Plots Generator

Creates comprehensive visualizations comparing:
1. Fitness functions effectiveness
2. Encoding approaches
3. Parameter sensitivity
4. Parameter space heatmaps
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from problem import MinimumVertexCoverProblem
from mvc_fitness import FitnessFunctionFactory
from mvc_encodings import BinaryEncoding
from ga_advanced_fitness import (
    FitnessSharingGA, AdaptivePenaltyGA, MultiObjectiveGA, 
    PhenotypePreservingGA
)
from ga_advanced_encodings import (
    DegreeBiasedBinary, GreedyHybridEncoding, AdaptiveThresholdEncoding,
    PermutationEncoding
)


class GAComparisonPlotter:
    """Generate comprehensive comparison plots."""
    
    def __init__(self):
        self.results_data = {}
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
    
    def load_results(self, filename: str = "ga_variations_results.json"):
        """Load results from JSON."""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.results_data = json.load(f)
            print(f"Loaded results from {filename}")
        else:
            print(f"Warning: {filename} not found")
    
    def plot_fitness_function_comparison(self):
        """Compare different fitness functions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        fitness_funcs = ['CoverSizeMin', 'FitnessSharingGA', 'AdaptivePenaltyGA', 
                        'MultiObjectiveGA', 'PhenotypePreservingGA']
        
        validity = []
        avg_sizes = []
        
        for func in fitness_funcs:
            key = f'fitness_{func}'
            if key in self.results_data:
                validity.append(self.results_data[key].get('validity', 0))
                avg_sizes.append(self.results_data[key].get('avg_size', 0))
        
        x_pos = np.arange(len(validity))
        
        # Validity comparison
        axes[0].bar(x_pos, validity, color='steelblue', alpha=0.8, edgecolor='black')
        axes[0].set_ylabel('Validity (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Fitness Functions: Validity', fontsize=13, fontweight='bold')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(fitness_funcs, rotation=45, ha='right')
        axes[0].set_ylim([0, 105])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(validity):
            axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)
        
        # Average cover size comparison
        axes[1].bar(x_pos, avg_sizes, color='coral', alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('Average Cover Size', fontsize=12, fontweight='bold')
        axes[1].set_title('Fitness Functions: Solution Quality', fontsize=13, fontweight='bold')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(fitness_funcs, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(avg_sizes):
            axes[1].text(i, v + 1, f'{v:.1f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('ga_fitness_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: ga_fitness_comparison.png")
        plt.close()
    
    def plot_encoding_comparison(self):
        """Compare different encodings."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        encodings = ['Binary', 'DegreeBiased', 'GreedyHybrid', 'AdaptiveThreshold', 'Permutation']
        
        validity = []
        avg_sizes = []
        times = []
        
        for enc in encodings:
            key = f'encoding_{enc}'
            if key in self.results_data:
                validity.append(self.results_data[key].get('validity', 0))
                avg_sizes.append(self.results_data[key].get('avg_size', 0))
                times.append(self.results_data[key].get('time', 0))
        
        x_pos = np.arange(len(validity))
        
        # Validity and quality trade-off
        ax = axes[0]
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x_pos - 0.2, validity, width=0.4, label='Validity %', 
                      color='steelblue', alpha=0.8, edgecolor='black')
        bars2 = ax2.bar(x_pos + 0.2, avg_sizes, width=0.4, label='Avg Size', 
                       color='coral', alpha=0.8, edgecolor='black')
        
        ax.set_ylabel('Validity (%)', fontsize=12, fontweight='bold', color='steelblue')
        ax2.set_ylabel('Average Cover Size', fontsize=12, fontweight='bold', color='coral')
        ax.set_title('Encodings: Validity & Quality Trade-off', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(encodings, rotation=45, ha='right')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax.grid(axis='y', alpha=0.3)
        
        # Runtime comparison
        axes[1].bar(x_pos, times, color='lightgreen', alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        axes[1].set_title('Encodings: Computational Cost', fontsize=13, fontweight='bold')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(encodings, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(times):
            axes[1].text(i, v + 0.01, f'{v:.3f}s', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('ga_encoding_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: ga_encoding_comparison.png")
        plt.close()
    
    def plot_parameter_sensitivity(self):
        """Parameter sensitivity heatmaps."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Extract parameter data
        pop_sizes = [50, 100, 200]
        mut_rates = [0.05, 0.1, 0.2, 0.3]
        
        validity_matrix = np.zeros((len(mut_rates), len(pop_sizes)))
        quality_matrix = np.zeros((len(mut_rates), len(pop_sizes)))
        
        for i, mut_rate in enumerate(mut_rates):
            for j, pop_size in enumerate(pop_sizes):
                key = f'params_pop{pop_size}_mut{mut_rate}'
                if key in self.results_data:
                    validity_matrix[i, j] = self.results_data[key].get('validity', 0)
                    quality_matrix[i, j] = self.results_data[key].get('avg_size', 0)
        
        # Validity heatmap
        sns.heatmap(validity_matrix, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                   cbar_kws={'label': 'Validity (%)'}, ax=axes[0],
                   xticklabels=[f'{p}' for p in pop_sizes],
                   yticklabels=[f'{m:.2f}' for m in mut_rates])
        axes[0].set_xlabel('Population Size', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Mutation Rate', fontsize=12, fontweight='bold')
        axes[0].set_title('Parameter Sensitivity: Validity', fontsize=13, fontweight='bold')
        
        # Quality heatmap
        sns.heatmap(quality_matrix, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                   cbar_kws={'label': 'Avg Cover Size'}, ax=axes[1],
                   xticklabels=[f'{p}' for p in pop_sizes],
                   yticklabels=[f'{m:.2f}' for m in mut_rates])
        axes[1].set_xlabel('Population Size', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Mutation Rate', fontsize=12, fontweight='bold')
        axes[1].set_title('Parameter Sensitivity: Solution Quality', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('ga_parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        print("Saved: ga_parameter_sensitivity.png")
        plt.close()
    
    def plot_best_combination(self):
        """Visualize best combination found."""
        fig = plt.figure(figsize=(12, 8))
        
        if 'best_combination' in self.results_data:
            best = self.results_data['best_combination']
            
            # Create text summary
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            summary_text = f"""
BEST COMBINATION FOUND

Fitness Function: {best.get('fitness', 'N/A')}
Encoding: {best.get('encoding', 'N/A')}
Population Size: {best.get('pop_size', 'N/A')}
Mutation Rate: {best.get('mutation_rate', 'N/A')}

Performance Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Validity: {best.get('validity', 0):.1f}%
  Average Cover Size: {best.get('avg_size', 0):.2f}
  Best Valid Size: {best.get('best_valid', 0)}
  Runtime: {best.get('time', 0):.3f}s

Recommendations:
  • This combination achieved highest validity
  • Consider ensemble with other methods
  • Fine-tune local search on top of GA
"""
            
            ax.text(0.1, 0.5, summary_text, fontsize=14, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig('ga_best_combination.png', dpi=300, bbox_inches='tight')
            print("Saved: ga_best_combination.png")
            plt.close()
    
    def generate_all_plots(self):
        """Generate all comparison plots."""
        print("\n" + "="*70)
        print("GENERATING GA COMPARISON PLOTS")
        print("="*70)
        
        if not self.results_data:
            self.load_results()
        
        if self.results_data:
            self.plot_fitness_function_comparison()
            self.plot_encoding_comparison()
            self.plot_parameter_sensitivity()
            self.plot_best_combination()
            print("\nAll plots generated successfully!")
        else:
            print("No results data available. Run ga_variations_analysis.py first.")


if __name__ == "__main__":
    plotter = GAComparisonPlotter()
    plotter.generate_all_plots()
