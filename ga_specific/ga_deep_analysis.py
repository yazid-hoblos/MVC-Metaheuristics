"""
Deep GA Analysis Script

Comprehensive analysis including:
- Convergence curves (best fitness over time)
- Diversity metrics (population heterogeneity)
- Validity progression (when first valid solution found)
- Solution quality distribution
- Search landscape visualization
"""

import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from problem import MinimumVertexCoverProblem
from mvc_fitness import FitnessFunctionFactory
from mvc_encodings import BinaryEncoding
from ga_advanced_fitness import FitnessSharingGA, AdaptivePenaltyGA, MultiObjectiveGA
from ga_advanced_encodings import DegreeBiasedBinary, GreedyHybridEncoding


class DeepGAAnalyzer:
    """Deep analysis of GA behavior over generations."""
    
    def __init__(self, num_runs: int = 3):
        self.num_runs = num_runs
        self.analysis_data = {}
    
    def generate_instance(self, num_nodes: int, edge_prob: float) -> MinimumVertexCoverProblem:
        """Generate random graph."""
        import networkx as nx
        
        # Create random graph using Erdos-Renyi model
        graph = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=None)
        problem = MinimumVertexCoverProblem(graph)
        
        return problem
    
    def _jaccard_similarity(self, set1: Set[int], set2: Set[int]) -> float:
        """Compute Jaccard similarity between two solutions."""
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _population_diversity(self, population: List[Set[int]]) -> float:
        """Compute average pairwise similarity (lower = more diverse)."""
        if len(population) <= 1:
            return 0.0
        
        total_similarity = 0
        count = 0
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                total_similarity += self._jaccard_similarity(population[i], population[j])
                count += 1
        
        return 1.0 - (total_similarity / count) if count > 0 else 0.0
    
    def run_instrumented_ga(self, problem: MinimumVertexCoverProblem, 
                            fitness_func, encoding, pop_size: int = 100,
                            generations: int = 300) -> Dict:
        """
        Run GA with instrumentation to track internal metrics.
        Returns convergence data and analysis.
        """
        from ga import GeneticAlgorithm, GAParams
        from mvc_fitness import FitnessFunctionFactory as FFF
        
        # Use standard GA but track metrics
        params = GAParams(population_size=pop_size, generations=generations, mutation_rate=0.1)
        
        metrics = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'first_valid': None,
            'first_valid_gen': None,
            'valid_progression': [],
            'best_cover_size': [],
            'solution_sizes': []
        }
        
        # Simple GA implementation for tracking
        population = []
        fitness_scores = []
        
        # Initialize population
        for _ in range(pop_size):
            if hasattr(encoding, 'random_genotype'):
                genotype = encoding.random_genotype()
                individual = encoding.decode(genotype)
            else:
                individual = set(np.random.choice(problem.num_nodes, 
                                                 size=max(1, problem.num_nodes//2),
                                                 replace=False))
            population.append(individual)
        
        # Evaluate initial population
        for ind in population:
            if hasattr(fitness_func, 'set_population'):
                fitness_func.set_population(population)
            fitness_scores.append(fitness_func.evaluate(ind))
        
        best_solution = population[np.argmax(fitness_scores)]
        best_fitness = max(fitness_scores)
        valid_count = sum(1 for ind in population if problem.is_valid_cover(ind))
        
        metrics['generation'].append(0)
        metrics['best_fitness'].append(best_fitness)
        metrics['avg_fitness'].append(np.mean(fitness_scores))
        metrics['diversity'].append(self._population_diversity(population))
        metrics['valid_progression'].append(valid_count)
        metrics['best_cover_size'].append(len(best_solution) if problem.is_valid_cover(best_solution) else -1)
        metrics['solution_sizes'].append([len(ind) for ind in population])
        
        if valid_count > 0 and metrics['first_valid'] is None:
            metrics['first_valid'] = len(best_solution)
            metrics['first_valid_gen'] = 0
        
        # Evolution loop
        for gen in range(1, generations):
            if hasattr(fitness_func, 'update_generation'):
                fitness_func.update_generation(gen)
            
            # Create new population
            new_population = []
            new_fitness = []
            
            # Elitism
            elite_size = max(1, int(pop_size * 0.05))
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
                new_fitness.append(fitness_scores[idx])
            
            # Fill rest with mutations/crossovers
            while len(new_population) < pop_size:
                # Select parent
                parent = population[np.random.randint(pop_size)].copy()
                
                # Mutate
                if np.random.random() < 0.5:
                    node = np.random.randint(problem.num_nodes)
                    if node in parent:
                        parent.discard(node)
                    else:
                        parent.add(node)
                
                new_population.append(parent)
                if hasattr(fitness_func, 'set_population'):
                    fitness_func.set_population(new_population)
                new_fitness.append(fitness_func.evaluate(parent))
            
            population = new_population
            fitness_scores = new_fitness
            
            # Track metrics
            best_idx = np.argmax(fitness_scores)
            best_solution = population[best_idx]
            best_fitness = fitness_scores[best_idx]
            valid_count = sum(1 for ind in population if problem.is_valid_cover(ind))
            
            metrics['generation'].append(gen)
            metrics['best_fitness'].append(best_fitness)
            metrics['avg_fitness'].append(np.mean(fitness_scores))
            metrics['diversity'].append(self._population_diversity(population))
            metrics['valid_progression'].append(valid_count)
            
            if problem.is_valid_cover(best_solution):
                metrics['best_cover_size'].append(len(best_solution))
                if metrics['first_valid'] is None:
                    metrics['first_valid'] = len(best_solution)
                    metrics['first_valid_gen'] = gen
            else:
                metrics['best_cover_size'].append(-1)
            
            metrics['solution_sizes'].append([len(ind) for ind in population])
        
        return metrics
    
    def analyze_fitness_functions(self, problem: MinimumVertexCoverProblem):
        """Deep analysis of different fitness functions."""
        print(f"\n{'='*80}")
        print(f"DEEP ANALYSIS: Fitness Functions on {problem.num_nodes} nodes")
        print(f"{'='*80}")
        
        fitness_functions = {
            'CoverSizeMin': FitnessFunctionFactory.create_cover_size_minimization(problem),
            'FitnessSharingGA': FitnessSharingGA(problem, sigma=0.3),
            'AdaptivePenaltyGA': AdaptivePenaltyGA(problem, max_generations=300),
            'MultiObjectiveGA': MultiObjectiveGA(problem),
        }
        
        encoding = DegreeBiasedBinary(problem)
        
        for func_name, fitness_func in fitness_functions.items():
            print(f"\n{func_name}:")
            print("-" * 60)
            
            all_metrics = []
            for run in range(self.num_runs):
                metrics = self.run_instrumented_ga(problem, fitness_func, encoding,
                                                  pop_size=100, generations=300)
                all_metrics.append(metrics)
            
            # Aggregate metrics
            final_valid = np.mean([sum(1 for ind in 
                                       # Reconstruct final population
                                       [problem.is_valid_cover(i) for i in range(10)])
                                  for _ in range(self.num_runs)])
            
            avg_first_valid_gen = np.mean([m['first_valid_gen'] for m in all_metrics 
                                          if m['first_valid_gen'] is not None])
            avg_first_valid_size = np.mean([m['first_valid'] for m in all_metrics 
                                           if m['first_valid'] is not None])
            
            print(f"  First valid found at generation: {avg_first_valid_gen:.1f}")
            print(f"  First valid cover size: {avg_first_valid_size:.1f}")
            print(f"  Average convergence: {np.mean([m['best_fitness'][-1] for m in all_metrics]):.4f}")
            print(f"  Diversity at end: {np.mean([m['diversity'][-1] for m in all_metrics]):.4f}")
            
            self.analysis_data[f"fitness_{func_name}"] = {
                'first_valid_gen': avg_first_valid_gen,
                'first_valid_size': avg_first_valid_size,
                'convergence': np.mean([m['best_fitness'][-1] for m in all_metrics]),
                'final_diversity': np.mean([m['diversity'][-1] for m in all_metrics])
            }
    
    def analyze_encodings(self, problem: MinimumVertexCoverProblem):
        """Deep analysis of different encodings."""
        print(f"\n{'='*80}")
        print(f"DEEP ANALYSIS: Encodings on {problem.num_nodes} nodes")
        print(f"{'='*80}")
        
        encodings = {
            'Binary': BinaryEncoding(problem),
            'DegreeBiased': DegreeBiasedBinary(problem),
            'GreedyHybrid': GreedyHybridEncoding(problem),
        }
        
        fitness_func = FitnessFunctionFactory.create_cover_size_minimization(problem)
        
        for enc_name, encoding in encodings.items():
            print(f"\n{enc_name}:")
            print("-" * 60)
            
            all_metrics = []
            for run in range(self.num_runs):
                metrics = self.run_instrumented_ga(problem, fitness_func, encoding,
                                                  pop_size=100, generations=300)
                all_metrics.append(metrics)
            
            avg_first_valid_gen = np.mean([m['first_valid_gen'] for m in all_metrics 
                                          if m['first_valid_gen'] is not None])
            avg_first_valid_size = np.mean([m['first_valid'] for m in all_metrics 
                                           if m['first_valid'] is not None])
            
            print(f"  First valid found at generation: {avg_first_valid_gen:.1f}")
            print(f"  First valid cover size: {avg_first_valid_size:.1f}")
            print(f"  Average convergence: {np.mean([m['best_fitness'][-1] for m in all_metrics]):.4f}")
            print(f"  Diversity at end: {np.mean([m['diversity'][-1] for m in all_metrics]):.4f}")
            
            self.analysis_data[f"encoding_{enc_name}"] = {
                'first_valid_gen': avg_first_valid_gen,
                'first_valid_size': avg_first_valid_size,
                'convergence': np.mean([m['best_fitness'][-1] for m in all_metrics]),
                'final_diversity': np.mean([m['diversity'][-1] for m in all_metrics])
            }
    
    def save_analysis(self, filename: str = "ga_deep_analysis.json"):
        """Save analysis results."""
        with open(filename, 'w') as f:
            json.dump(self.analysis_data, f, indent=2)
        print(f"\nAnalysis saved to {filename}")
    
    def run(self):
        """Run complete deep analysis."""
        print("\n" + "="*80)
        print("DEEP GA ANALYSIS - CONVERGENCE, DIVERSITY, VALIDITY TRACKING")
        print("="*80)
        
        # Test on 50-node instance
        problem = self.generate_instance(50, 0.2)
        print(f"\nInstance: {problem.num_nodes} nodes, {problem.num_edges} edges")
        
        self.analyze_fitness_functions(problem)
        self.analyze_encodings(problem)
        
        self.save_analysis()


if __name__ == "__main__":
    analyzer = DeepGAAnalyzer(num_runs=2)
    analyzer.run()
