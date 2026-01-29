"""
Fast Experiment Runner - Smaller scale for testing
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import json
import csv
import time
import numpy as np
from typing import List, Dict
from problem import InstanceGenerator, MinimumVertexCoverProblem
from mvc_encodings import EncodingFactory
from mvc_fitness import FitnessFunctionFactory
from ga import GeneticAlgorithm, GAParams
from sa import SimulatedAnnealing, SAParams
from ts import TabuSearch, TSParams


class FastExperimentRunner:
    """Run experiments for report."""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def run_all_algorithms(self, problem, instance_name, run_id):
        """Run GA, SA, TS on problem instance."""
        encodings = EncodingFactory.get_all_encodings(problem.edges)
        fitness_functions = FitnessFunctionFactory.get_all_functions(problem)
        
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
                    print(f"GA[{result_ga['best_cover_size']}] ", end='', flush=True)
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
        csv_path = self.output_dir / "results.csv"
        if self.results:
            keys = self.results[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)
            print(f"\nResults saved to: {csv_path}")
    
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
    runner = FastExperimentRunner(output_dir="./results")
    
    print("Starting experiments...")
    runner.run_experiments(num_runs=3)  # 3 runs for quick testing
    
    runner.save_results()
    runner.print_summary()
    
    print("\nExperiments completed!")
