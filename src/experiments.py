"""
Comprehensive Experiment Runner
Tests all combinations of meta-heuristics, encodings, and fitness functions
"""

import json
import csv
import time
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import sys

# Adjust path for imports
sys.path.insert(0, str(Path(__file__).parent))

from problem import InstanceGenerator, MinimumVertexCoverProblem
from mvc_encodings import EncodingFactory
from mvc_fitness import FitnessFunctionFactory
from ga import GeneticAlgorithm, GAParams
from sa import SimulatedAnnealing, SAParams
from ts import TabuSearch, TSParams


class ExperimentRunner:
    """Run comprehensive experiments on MVC problem."""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def run_ga_experiment(self,
                         problem: MinimumVertexCoverProblem,
                         encoding,
                         fitness_func,
                         run_id: int,
                         instance_name: str) -> Dict:
        """Run single GA experiment."""
        params = GAParams(
            population_size=100,
            generations=300,
            mutation_rate=0.1,
            crossover_rate=0.8,
            selection_type="tournament",
            tournament_size=3,
            elitism_rate=0.05
        )
        
        ga = GeneticAlgorithm(problem, encoding, fitness_func, params)
        
        start_time = time.time()
        result = ga.run()
        elapsed_time = time.time() - start_time
        
        return {
            'instance': instance_name,
            'algorithm': 'GA',
            'encoding': encoding.get_name(),
            'fitness_func': fitness_func.get_name(),
            'run_id': run_id,
            'cover_size': result['best_cover_size'],
            'is_valid': result['is_valid'],
            'fitness': result['best_fitness'],
            'time_seconds': elapsed_time,
            'problem_nodes': problem.num_nodes,
            'problem_edges': problem.num_edges
        }
    
    def run_sa_experiment(self,
                         problem: MinimumVertexCoverProblem,
                         encoding,
                         fitness_func,
                         run_id: int,
                         instance_name: str) -> Dict:
        """Run single SA experiment."""
        params = SAParams(
            initial_temperature=100.0,
            cooling_rate=0.95,
            iterations_per_temperature=50,
            min_temperature=0.01,
            max_iterations=5000
        )
        
        sa = SimulatedAnnealing(problem, encoding, fitness_func, params)
        
        start_time = time.time()
        result = sa.run()
        elapsed_time = time.time() - start_time
        
        return {
            'instance': instance_name,
            'algorithm': 'SA',
            'encoding': encoding.get_name(),
            'fitness_func': fitness_func.get_name(),
            'run_id': run_id,
            'cover_size': result['best_cover_size'],
            'is_valid': result['is_valid'],
            'fitness': result['best_fitness'],
            'time_seconds': elapsed_time,
            'problem_nodes': problem.num_nodes,
            'problem_edges': problem.num_edges
        }
    
    def run_ts_experiment(self,
                         problem: MinimumVertexCoverProblem,
                         encoding,
                         fitness_func,
                         run_id: int,
                         instance_name: str) -> Dict:
        """Run single Tabu Search experiment."""
        params = TSParams(
            tabu_list_size=50,
            max_iterations=5000,
            aspiration_criteria=True
        )
        
        ts = TabuSearch(problem, encoding, fitness_func, params)
        
        start_time = time.time()
        result = ts.run()
        elapsed_time = time.time() - start_time
        
        return {
            'instance': instance_name,
            'algorithm': 'TS',
            'encoding': encoding.get_name(),
            'fitness_func': fitness_func.get_name(),
            'run_id': run_id,
            'cover_size': result['best_cover_size'],
            'is_valid': result['is_valid'],
            'fitness': result['best_fitness'],
            'time_seconds': elapsed_time,
            'problem_nodes': problem.num_nodes,
            'problem_edges': problem.num_edges
        }
    
    def run_full_experiments(self, num_runs: int = 30):
        """
        Run full experimental suite.
        
        Tests:
        - 4 problem instances (small, medium, large, scale-free)
        - 3 meta-heuristics (GA, SA, TS)
        - 3 encodings (Binary, Set, EdgeCentric)
        - 3 fitness functions
        - num_runs independent runs per configuration
        """
        print("Generating benchmark instances...")
        instances = InstanceGenerator.generate_benchmark_instances()
        
        total_experiments = (len(instances) * 3 * 3 * 3 * num_runs)  # 3 algos, 3 encodings, 3 fitness
        completed = 0
        
        for problem, instance_name in instances:
            print(f"\n{'='*60}")
            print(f"Instance: {instance_name}")
            print(f"Nodes: {problem.num_nodes}, Edges: {problem.num_edges}")
            print(f"{'='*60}")
            
            encodings = EncodingFactory.get_all_encodings(problem.edges)
            fitness_functions = FitnessFunctionFactory.get_all_functions(problem)
            
            for run_id in range(num_runs):
                print(f"\nRun {run_id + 1}/{num_runs}")
                
                for encoding in encodings:
                    for fitness_func in fitness_functions:
                        # GA
                        try:
                            result = self.run_ga_experiment(
                                problem, encoding, fitness_func, run_id, instance_name
                            )
                            self.results.append(result)
                            completed += 1
                            print(f"  GA + {encoding.get_name()[:3]} + {fitness_func.get_name()[:3]}: "
                                  f"cover={result['cover_size']}, valid={result['is_valid']}")
                        except Exception as e:
                            print(f"  GA failed: {e}")
                            completed += 1
                        
                        # SA
                        try:
                            result = self.run_sa_experiment(
                                problem, encoding, fitness_func, run_id, instance_name
                            )
                            self.results.append(result)
                            completed += 1
                        except Exception as e:
                            print(f"  SA failed: {e}")
                            completed += 1
                        
                        # TS
                        try:
                            result = self.run_ts_experiment(
                                problem, encoding, fitness_func, run_id, instance_name
                            )
                            self.results.append(result)
                            completed += 1
                        except Exception as e:
                            print(f"  TS failed: {e}")
                            completed += 1
                        
                        print(f"  Progress: {completed}/{total_experiments} ({100*completed/total_experiments:.1f}%)")
        
        return self.results
    
    def save_results(self):
        """Save results to CSV and JSON."""
        # CSV format
        csv_path = self.output_dir / "results.csv"
        if self.results:
            keys = self.results[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)
        
        # JSON format
        json_path = self.output_dir / "results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  {csv_path}")
        print(f"  {json_path}")
    
    def analyze_results(self):
        """Generate summary statistics."""
        if not self.results:
            return
        
        summary_path = self.output_dir / "summary.txt"
        
        with open(summary_path, 'w') as f:
            # Group by algorithm
            for algo in ['GA', 'SA', 'TS']:
                algo_results = [r for r in self.results if r['algorithm'] == algo]
                
                if not algo_results:
                    continue
                
                f.write(f"\n{'='*60}\n")
                f.write(f"{algo} RESULTS\n")
                f.write(f"{'='*60}\n")
                
                # Group by instance
                instances = set(r['instance'] for r in algo_results)
                for instance in sorted(instances):
                    inst_results = [r for r in algo_results if r['instance'] == instance]
                    
                    f.write(f"\n{instance}:\n")
                    
                    # By encoding
                    encodings = set(r['encoding'] for r in inst_results)
                    for encoding in sorted(encodings):
                        enc_results = [r for r in inst_results if r['encoding'] == encoding]
                        
                        valid_results = [r for r in enc_results if r['is_valid']]
                        
                        if valid_results:
                            sizes = [r['cover_size'] for r in valid_results]
                            f.write(f"  {encoding}: "
                                   f"avg_size={np.mean(sizes):.2f} ± {np.std(sizes):.2f}, "
                                   f"min={min(sizes)}, max={max(sizes)}, "
                                   f"valid={len(valid_results)}/{len(enc_results)}\n")
                        else:
                            f.write(f"  {encoding}: NO VALID SOLUTIONS\n")
        
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    import sys
    
    # Run experiments
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    runner = ExperimentRunner(output_dir="./results")
    
    print(f"Starting experiments with {num_runs} runs per configuration...")
    runner.run_full_experiments(num_runs=num_runs)
    
    print("\nSaving results...")
    runner.save_results()
    
    print("\nAnalyzing results...")
    runner.analyze_results()
    
    print("\nExperiments completed!")
