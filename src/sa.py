"""
Simulated Annealing for Minimum Vertex Cover Problem
"""

import random
import math
import numpy as np
from typing import List, Set, Callable
from dataclasses import dataclass
from copy import deepcopy
from mvc_encodings import Encoding
from mvc_fitness import FitnessFunction


@dataclass
class SAParams:
    """Simulated Annealing parameters."""
    initial_temperature: float = 100.0
    cooling_rate: float = 0.95
    iterations_per_temperature: int = 50
    min_temperature: float = 0.01
    max_iterations: int = 5000


class NeighborhoodOperator:
    """Generate neighbor solutions."""
    
    def __init__(self, encoding: Encoding):
        self.encoding = encoding
    
    def get_neighbor(self, solution, num_nodes: int):
        """
        Generate a neighbor solution by small modification.
        
        For binary encoding: flip 1-2 random bits
        For set encoding: add/remove a node
        """
        encoding_name = self.encoding.get_name()
        neighbor = deepcopy(solution)
        
        if encoding_name == "BinaryEncoding":
            # Flip 1-2 random bits
            num_flips = random.randint(1, 2)
            for _ in range(num_flips):
                idx = random.randint(0, len(neighbor) - 1)
                neighbor[idx] = 1 - neighbor[idx]
        
        elif encoding_name == "SetEncoding":
            if random.random() < 0.5 and len(neighbor) > 0:
                # Remove random node
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
            else:
                # Add random node - ensure it's not already in the set
                candidate_nodes = [n for n in range(num_nodes) if n not in neighbor]
                if candidate_nodes:
                    node = random.choice(candidate_nodes)
                    neighbor.append(node)
                    neighbor.sort()
                elif len(neighbor) > 0:
                    # If all nodes are in the set, remove one instead
                    idx = random.randint(0, len(neighbor) - 1)
                    neighbor.pop(idx)
        
        else:  # EdgeCentricEncoding
            # Flip 1-2 random bits
            num_flips = random.randint(1, 2)
            for _ in range(num_flips):
                idx = random.randint(0, len(neighbor) - 1)
                neighbor[idx] = 1 - neighbor[idx]
        
        return neighbor


class SimulatedAnnealing:
    """
    Simulated Annealing for Minimum Vertex Cover
    
    Algorithm:
    1. Start with random solution
    2. Generate neighbor solution
    3. Accept if better, or accept with probability based on temperature
    4. Cool temperature gradually
    5. Stop when temperature is very low
    
    Key parameters:
    - Initial temperature: controls initial acceptance probability
    - Cooling rate: how fast temperature decreases (0.9-0.99)
    - Iterations per temperature: how many iterations at each temperature level
    """
    
    def __init__(self,
                 problem,
                 encoding: Encoding,
                 fitness_func: FitnessFunction,
                 params: SAParams = None):
        """
        Initialize Simulated Annealing.
        
        Args:
            problem: MinimumVertexCoverProblem instance
            encoding: Encoding strategy
            fitness_func: Fitness function
            params: SA parameters
        """
        self.problem = problem
        self.encoding = encoding
        self.fitness_func = fitness_func
        self.params = params or SAParams()
        self.neighborhood = NeighborhoodOperator(encoding)
        
        # Statistics
        self.iteration_stats = []
    
    def acceptance_probability(self, current_fitness: float, 
                              neighbor_fitness: float, 
                              temperature: float) -> float:
        """
        Calculate probability of accepting worse solution.
        
        Uses Metropolis criterion:
        P = exp((neighbor_fitness - current_fitness) / temperature)
        """
        if neighbor_fitness >= current_fitness:
            return 1.0  # Always accept better solution
        
        delta = neighbor_fitness - current_fitness
        return math.exp(delta / temperature) if temperature > 0 else 0.0
    
    def run(self) -> dict:
        """
        Run Simulated Annealing.
        
        Returns:
            Dictionary with results
        """
        # Initialize with random solution
        current_solution = self.encoding.random_solution(self.problem.num_nodes)
        current_cover = self.encoding.solution_to_cover(current_solution)
        current_fitness = self.fitness_func.evaluate(current_cover)
        
        # Track best solution found
        best_solution = deepcopy(current_solution)
        best_cover = deepcopy(current_cover)
        best_fitness = current_fitness
        
        temperature = self.params.initial_temperature
        iteration = 0
        
        while temperature > self.params.min_temperature and iteration < self.params.max_iterations:
            
            for _ in range(self.params.iterations_per_temperature):
                # Generate neighbor
                neighbor_solution = self.neighborhood.get_neighbor(current_solution, 
                                                                   self.problem.num_nodes)
                neighbor_cover = self.encoding.solution_to_cover(neighbor_solution)
                neighbor_fitness = self.fitness_func.evaluate(neighbor_cover)
                
                # Acceptance decision
                accept_prob = self.acceptance_probability(current_fitness, 
                                                         neighbor_fitness, 
                                                         temperature)
                
                if random.random() < accept_prob:
                    current_solution = neighbor_solution
                    current_cover = neighbor_cover
                    current_fitness = neighbor_fitness
                    
                    # Update best if needed
                    if current_fitness > best_fitness:
                        best_fitness = current_fitness
                        best_solution = deepcopy(current_solution)
                        best_cover = deepcopy(current_cover)
                
                iteration += 1
                
                # Store statistics
                self.iteration_stats.append({
                    'iteration': iteration,
                    'temperature': temperature,
                    'current_fitness': current_fitness,
                    'best_fitness': best_fitness,
                    'best_cover_size': len(best_cover)
                })
                
                if iteration >= self.params.max_iterations:
                    break
            
            # Cool temperature
            temperature *= self.params.cooling_rate
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'best_cover': best_cover,
            'best_cover_size': len(best_cover),
            'is_valid': self.problem.is_valid_cover(best_cover),
            'iteration_stats': self.iteration_stats,
            'total_iterations': iteration
        }


if __name__ == "__main__":
    from problem import InstanceGenerator
    from encodings import BinaryEncoding
    from fitness import CoverSizeMinimization
    
    # Test SA
    problem, _ = InstanceGenerator.generate_benchmark_instances()[0]
    encoding = BinaryEncoding()
    fitness = CoverSizeMinimization(problem)
    
    params = SAParams(initial_temperature=50.0, max_iterations=2000)
    sa = SimulatedAnnealing(problem, encoding, fitness, params)
    
    result = sa.run()
    
    print(f"Best cover size: {result['best_cover_size']}")
    print(f"Is valid: {result['is_valid']}")
    print(f"Best fitness: {result['best_fitness']:.6f}")
    print(f"Total iterations: {result['total_iterations']}")
