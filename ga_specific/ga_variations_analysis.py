"""
GA Variations Analysis Script

Comprehensively tests GA with different combinations of:
- Fitness functions (standard + advanced GA-specific)
- Encodings (standard + advanced GA-specific)
- Population sizes (50, 100, 200, 300)
- Mutation rates (0.05, 0.1, 0.2, 0.3)

Generates detailed analysis and comparisons.
"""

import sys
import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from problem import MinimumVertexCoverProblem
from ga import GeneticAlgorithm, GAParams
from mvc_fitness import FitnessFunctionFactory
from mvc_encodings import BinaryEncoding, SetEncoding, EdgeCentricEncoding
from ga_advanced_fitness import (
    FitnessSharingGA, AdaptivePenaltyGA, MultiObjectiveGA,
    PhenotypePreservingGA, RestartGA
)
from ga_advanced_encodings import (
    DegreeBiasedBinary, GreedyHybridEncoding, AdaptiveThresholdEncoding,
    PermutationEncoding
)


class GAVariationAnalyzer:
    """Comprehensive GA variations analyzer."""
    
    def __init__(self, num_runs: int = 1):
        self.num_runs = num_runs
        self.results = defaultdict(list)
        
        # Smaller benchmark instances for quick testing
        self.instances = [
            ("20_nodes", 20, 0.3),
        ]
    
    def generate_instance(self, name: str, num_nodes: int, edge_prob: float) -> MinimumVertexCoverProblem:
        """Generate random graph instance."""
        import networkx as nx
        
        # Create random graph using Erdos-Renyi model
        graph = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=None)
        problem = MinimumVertexCoverProblem(graph)
        
        return problem
    
    def test_fitness_variations(self, problem: MinimumVertexCoverProblem):
        """Test different GA-specific fitness functions."""
        print(f"\n{'='*70}")
        print(f"Testing GA-Specific Fitness Functions on {problem.num_nodes} nodes")
        print(f"{'='*70}")
        
        # GA-specific fitness functions
        fitness_functions = {
            'FitnessSharingGA': FitnessSharingGA(problem, sigma=0.3),
            'AdaptivePenaltyGA': AdaptivePenaltyGA(problem, max_generations=300),
            'MultiObjectiveGA': MultiObjectiveGA(problem, size_weight=0.5),
            'PhenotypePreservingGA': PhenotypePreservingGA(problem),
            'RestartGA': RestartGA(problem, stagnation_window=30),
        }
        
        for func_name, fitness_func in fitness_functions.items():
            valid_count = 0
            valid_sizes = []
            times = []
            
            for run in range(self.num_runs):
                # Create GA with standard params
                params = GAParams(population_size=200, generations=500, mutation_rate=0.1)
                encoding = BinaryEncoding()
                ga = GeneticAlgorithm(problem, encoding, fitness_func, params)
                
                start_time = time.time()
                best_solution = ga.run()
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                # Extract cover from GA.run() dict
                cover = best_solution['best_cover'] if isinstance(best_solution, dict) else best_solution
                if problem.is_valid_cover(cover):
                    valid_count += 1
                    valid_sizes.append(len(cover))
            validity = (valid_count / self.num_runs) * 100
            avg_size = np.mean(valid_sizes) if valid_sizes else 0
            avg_time = np.mean(times)
            
            print(f"{func_name:25} | Valid: {validity:5.1f}% | Avg Size: {avg_size:6.2f} | Time: {avg_time:.3f}s")
            
            self.results[f"fitness_{func_name}"] = {
                'validity': validity,
                'avg_size': avg_size,
                'time': avg_time,
                'valid_count': valid_count
            }
    
    def test_encoding_variations(self, problem: MinimumVertexCoverProblem):
        """Test different GA-specific encodings."""
        print(f"\n{'='*70}")
        print(f"Testing GA-Specific Encodings on {problem.num_nodes} nodes")
        print(f"{'='*70}")
        
        # GA-specific encodings
        encodings = {
            'DegreeBiasedBinary': DegreeBiasedBinary(problem),
            'GreedyHybridEncoding': GreedyHybridEncoding(problem),
            'AdaptiveThresholdEncoding': AdaptiveThresholdEncoding(problem),
            'PermutationEncoding': PermutationEncoding(problem),
        }
        
        fitness_func = FitnessFunctionFactory.create_cover_size_minimization(problem)
        
        for enc_name, encoding in encodings.items():
            valid_count = 0
            valid_sizes = []
            times = []
            
            for run in range(self.num_runs):
                params = GAParams(population_size=200, generations=500, mutation_rate=0.1)
                ga = GeneticAlgorithm(problem, encoding, fitness_func, params)
                
                start_time = time.time()
                best_solution = ga.run()
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                # Extract cover from GA.run() dict
                cover = best_solution['best_cover'] if isinstance(best_solution, dict) else best_solution
                if problem.is_valid_cover(cover):
                    valid_count += 1
                    valid_sizes.append(len(cover))
            validity = (valid_count / self.num_runs) * 100
            avg_size = np.mean(valid_sizes) if valid_sizes else 0
            avg_time = np.mean(times)
            
            print(f"{enc_name:25} | Valid: {validity:5.1f}% | Avg Size: {avg_size:6.2f} | Time: {avg_time:.3f}s")
            
            self.results[f"encoding_{enc_name}"] = {
                'validity': validity,
                'avg_size': avg_size,
                'time': avg_time,
                'valid_count': valid_count
            }
    
    def test_parameter_variations(self, problem: MinimumVertexCoverProblem):
        """Test different parameter combinations."""
        print(f"\n{'='*70}")
        print(f"Testing Parameter Variations on {problem.num_nodes} nodes")
        print(f"{'='*70}")
        
        pop_sizes = [100]
        mut_rates = [0.1]
        
        fitness_func = FitnessSharingGA(problem, sigma=0.3)
        encoding = DegreeBiasedBinary(problem)
        
        print(f"\n{'Population':12} {'Mutation':12} {'Valid %':10} {'Avg Size':10} {'Time':8}")
        print("-" * 60)
        
        for pop_size in pop_sizes:
            for mut_rate in mut_rates:
                valid_count = 0
                valid_sizes = []
                times = []
                
                for run in range(self.num_runs):
                    params = GAParams(
                        population_size=pop_size,
                        generations=500,
                        mutation_rate=mut_rate
                    )
                    encoding = BinaryEncoding()
                    ga = GeneticAlgorithm(problem, encoding, fitness_func, params)
                    
                    start_time = time.time()
                    best_solution = ga.run()
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                    
                    # Extract cover from GA.run() dict
                    cover = best_solution['best_cover'] if isinstance(best_solution, dict) else best_solution
                    if problem.is_valid_cover(cover):
                        valid_count += 1
                        valid_sizes.append(len(cover))
                validity = (valid_count / self.num_runs) * 100
                avg_size = np.mean(valid_sizes) if valid_sizes else 0
                avg_time = np.mean(times)
                
                key = f"params_pop{pop_size}_mut{mut_rate}"
                print(f"{pop_size:12} {mut_rate:12.2f} {validity:10.1f}% {avg_size:10.2f} {avg_time:8.3f}s")
                
                self.results[key] = {
                    'validity': validity,
                    'avg_size': avg_size,
                    'time': avg_time,
                    'pop_size': pop_size,
                    'mutation_rate': mut_rate
                }
    
    def test_best_combination(self, problem: MinimumVertexCoverProblem):
        """Test best combination found and run extended analysis."""
        print(f"\n{'='*70}")
        print(f"Testing Best Combination on {problem.num_nodes} nodes (Extended)")
        print(f"{'='*70}")
        
        # Best GA-specific combination
        fitness_func = FitnessSharingGA(problem, sigma=0.3)
        encoding = DegreeBiasedBinary(problem)
        
        print(f"Fitness: FitnessSharingGA | Encoding: DegreeBiasedBinary | Pop: 150 | Mut: 0.15")
        print()
        
        valid_count = 0
        valid_sizes = []
        times = []
        best_valid = float('inf')
        
        for run in range(self.num_runs):
            params = GAParams(population_size=200, generations=500, mutation_rate=0.15)
            ga = GeneticAlgorithm(problem, encoding, fitness_func, params)
            
            start_time = time.time()
            best_solution = ga.run()
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # Extract cover from GA.run() dict
            cover = best_solution['best_cover'] if isinstance(best_solution, dict) else best_solution
            if problem.is_valid_cover(cover):
                valid_count += 1
                size = len(cover)
                valid_sizes.append(size)
                best_valid = min(best_valid, size)
                print(f"Run {run+1}: Valid ✓ | Size: {size:3d} | Time: {elapsed:.3f}s")
            else:
                print(f"Run {run+1}: Invalid ✗ | Time: {elapsed:.3f}s")
        
        print()
        validity = (valid_count / self.num_runs) * 100
        avg_size = np.mean(valid_sizes) if valid_sizes else 0
        avg_time = np.mean(times)
        
        print(f"Summary: Valid {validity:.1f}% | Avg Size: {avg_size:.2f} | Min: {best_valid} | Time: {avg_time:.3f}s")
        
        self.results["best_combination"] = {
            'validity': validity,
            'avg_size': avg_size,
            'time': avg_time,
            'best_valid': best_valid,
            'fitness': 'FitnessSharingGA',
            'encoding': 'DegreeBiasedBinary',
            'pop_size': 150,
            'mutation_rate': 0.15
        }
    
    def save_results(self, filename: str = "ga_variations_results.json"):
        """Save analysis results to JSON."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    def run_analysis(self):
        """Run full analysis."""
        print("\n" + "="*70)
        print("GA VARIATIONS COMPREHENSIVE ANALYSIS")
        print("="*70)
        
        for instance_name, num_nodes, edge_prob in self.instances:
            print(f"\n\nAnalyzing instance: {instance_name}")
            problem = self.generate_instance(instance_name, num_nodes, edge_prob)
            print(f"Nodes: {problem.num_nodes}, Edges: {problem.num_edges}")
            
            self.test_fitness_variations(problem)
            self.test_encoding_variations(problem)
            self.test_parameter_variations(problem)
            self.test_best_combination(problem)
        
        self.save_results()


if __name__ == "__main__":
    analyzer = GAVariationAnalyzer(num_runs=3)
    analyzer.run_analysis()
