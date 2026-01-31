"""
Fast Experiment Runner - Smaller scale for testing
Supports filtering by encoding and fitness function

Usage:
    python3 run_experiments.py                              # Run all, save to results/results.csv
    python3 run_experiments.py --encoding SetEncoding       # Only SetEncoding
    python3 run_experiments.py --fitness EdgeCoverage       # Only EdgeCoverage fitness
    python3 run_experiments.py --encoding BinaryEncoding --fitness CoverSizeMinimization
    python3 run_experiments.py --output ./my_results.csv    # Save to custom CSV file
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import json
import csv
import time
import argparse
import numpy as np
from typing import List, Dict, Optional
from problem import InstanceGenerator, MinimumVertexCoverProblem
from mvc_encodings import EncodingFactory
from mvc_fitness import FitnessFunctionFactory
from ga import GeneticAlgorithm, GAParams
from sa import SimulatedAnnealing, SAParams
from ts import TabuSearch, TSParams


class FastExperimentRunner:
    """Run experiments for report."""
    
    def __init__(self, output_file: str = "./results/results.csv", 
                 encoding_filter: Optional[str] = None,
                 fitness_filter: Optional[str] = None):
        output_path = Path(output_file)
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = output_path
        self.results = []
        self.encoding_filter = encoding_filter
        self.fitness_filter = fitness_filter
    
    def run_all_algorithms(self, problem, instance_name, run_id):
        """Run GA, SA, TS on problem instance."""
        all_encodings = EncodingFactory.get_all_encodings(problem.edges)
        all_fitness_functions = FitnessFunctionFactory.get_all_functions(problem)
        
        # Filter encodings
        if self.encoding_filter:
            encodings = [e for e in all_encodings if e.get_name() == self.encoding_filter]
            if not encodings:
                print(f"  Warning: Encoding '{self.encoding_filter}' not found. Available: {[e.get_name() for e in all_encodings]}")
                return
        else:
            encodings = all_encodings
        
        # Filter fitness functions
        if self.fitness_filter:
            fitness_functions = [f for f in all_fitness_functions if f.get_name() == self.fitness_filter]
            if not fitness_functions:
                print(f"  Warning: Fitness function '{self.fitness_filter}' not found. Available: {[f.get_name() for f in all_fitness_functions]}")
                return
        else:
            fitness_functions = all_fitness_functions
        
        for encoding in encodings:
            for fitness_func in fitness_functions:
                enc_name = encoding.get_name()
                ff_name = fitness_func.get_name()
                
                print(f"  {enc_name:20s} + {ff_name:30s}: ", end='', flush=True)
                
                # GA
                try:
                    params_ga = GAParams(population_size=100, generations=300)
                    ga = GeneticAlgorithm(problem, encoding, fitness_func, params_ga)
                    start = time.time()
                    result_ga = ga.run()
                    time_ga = time.time() - start
                    
                    self.results.append({
                        'instance': instance_name,
                        'algorithm': 'GA',
                        'encoding': enc_name,
                        'fitness_func': ff_name,
                        'run': run_id,
                        'cover_size': result_ga['best_cover_size'],
                        'is_valid': result_ga['is_valid'],
                        'fitness': result_ga['best_fitness'],
                        'time_sec': time_ga
                    })
                    print(f"GA[{result_ga['best_cover_size']}] [{result_ga['is_valid']}] ", end='', flush=True)
                except Exception as e:
                    print(f"GA[ERR] ", end='', flush=True)
                
                # SA
                try:
                    params_sa = SAParams(initial_temperature=100.0, max_iterations=5000)
                    sa = SimulatedAnnealing(problem, encoding, fitness_func, params_sa)
                    start = time.time()
                    result_sa = sa.run()
                    time_sa = time.time() - start
                    
                    self.results.append({
                        'instance': instance_name,
                        'algorithm': 'SA',
                        'encoding': enc_name,
                        'fitness_func': ff_name,
                        'run': run_id,
                        'cover_size': result_sa['best_cover_size'],
                        'is_valid': result_sa['is_valid'],
                        'fitness': result_sa['best_fitness'],
                        'time_sec': time_sa
                    })
                    print(f"SA[{result_sa['best_cover_size']}] ", end='', flush=True)
                except Exception as e:
                    print(f"SA[ERR] ", end='', flush=True)
                
                # TS
                try:
                    params_ts = TSParams(tabu_list_size=50, max_iterations=5000)
                    ts = TabuSearch(problem, encoding, fitness_func, params_ts)
                    start = time.time()
                    result_ts = ts.run()
                    time_ts = time.time() - start
                    
                    self.results.append({
                        'instance': instance_name,
                        'algorithm': 'TS',
                        'encoding': enc_name,
                        'fitness_func': ff_name,
                        'run': run_id,
                        'cover_size': result_ts['best_cover_size'],
                        'is_valid': result_ts['is_valid'],
                        'fitness': result_ts['best_fitness'],
                        'time_sec': time_ts
                    })
                    print(f"TS[{result_ts['best_cover_size']}]", end='', flush=True)
                except Exception as e:
                    print(f"TS[ERR]", end='', flush=True)
                
                print()
    
    def run_experiments(self, num_runs: int = 5):
        """Run full experiment suite."""
        print("\nGenerating benchmark instances...")
        instances = InstanceGenerator.generate_benchmark_instances()
        
        # Display filter settings
        if self.encoding_filter or self.fitness_filter:
            print(f"\nFilters applied:")
            if self.encoding_filter:
                print(f"  Encoding: {self.encoding_filter}")
            if self.fitness_filter:
                print(f"  Fitness Function: {self.fitness_filter}")
            print()
        
        for problem, instance_name in instances:
            print(f"\n{'='*70}")
            print(f"Instance: {instance_name} (Nodes={problem.num_nodes}, Edges={problem.num_edges})")
            print(f"{'='*70}")
            
            for run_id in range(num_runs):
                print(f"Run {run_id + 1}/{num_runs}:")
                self.run_all_algorithms(problem, instance_name, run_id + 1)
        
        return self.results
    
    def save_results(self):
        """Save results to CSV."""
        if self.results:
            keys = self.results[0].keys()
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)
            print(f"\nResults saved to: {self.output_file}")
    
    def print_summary(self):
        """Print summary statistics."""
        if not self.results:
            return
        
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        # Group by algorithm
        for algo in ['GA', 'SA', 'TS']:
            algo_results = [r for r in self.results if r['algorithm'] == algo]
            if not algo_results:
                continue
            
            valid_results = [r for r in algo_results if r['is_valid']]
            if valid_results:
                sizes = [r['cover_size'] for r in valid_results]
                times = [r['time_sec'] for r in valid_results]
                print(f"\n{algo}:")
                print(f"  Valid solutions: {len(valid_results)}/{len(algo_results)}")
                print(f"  Avg cover size: {np.mean(sizes):.2f} ± {np.std(sizes):.2f}")
                print(f"  Min/Max cover: {min(sizes)}/{max(sizes)}")
                print(f"  Avg time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
            else:
                print(f"\n{algo}: NO VALID SOLUTIONS FOUND")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MVC metaheuristic experiments with optional filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 run_experiments.py                                    # Run all
  python3 run_experiments.py --encoding SetEncoding             # Only SetEncoding
  python3 run_experiments.py --fitness EdgeCoverage             # Only EdgeCoverage
  python3 run_experiments.py --encoding BinaryEncoding --fitness CoverSizeMinimization
        """
    )
    parser.add_argument('--encoding', type=str, default=None,
                       help='Filter by encoding (e.g., BinaryEncoding, SetEncoding, EdgeCentricEncoding)')
    parser.add_argument('--fitness', type=str, default=None,
                       help='Filter by fitness function (e.g., CoverSizeMinimization, ConstraintPenalty, EdgeCoverage)')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per configuration (default: 3)')
    parser.add_argument('--output', type=str, default='./results/results.csv',
                       help='Output CSV file path (default: ./results/results.csv)')
    
    args = parser.parse_args()
    
    runner = FastExperimentRunner(
        output_file=args.output,
        encoding_filter=args.encoding,
        fitness_filter=args.fitness
    )
    
    print("Starting experiments...")
    runner.run_experiments(num_runs=args.runs)
    
    runner.save_results()
    runner.print_summary()
    
    print("\nExperiments completed!")
