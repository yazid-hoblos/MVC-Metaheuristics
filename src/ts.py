"""
Tabu Search for Minimum Vertex Cover Problem
"""

import random
import numpy as np
from typing import List, Set, Tuple
from dataclasses import dataclass, field
from copy import deepcopy
from collections import deque
from mvc_encodings import Encoding
from mvc_fitness import FitnessFunction


@dataclass
class TSParams:
    """Tabu Search parameters."""
    tabu_list_size: int = 50
    max_iterations: int = 5000
    aspiration_criteria: bool = True


class TabuList:
    """
    Tabu list to prevent recently visited solutions.
    Uses a deque of fixed size.
    """
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.list = deque(maxlen=max_size)
    
    def is_tabu(self, solution_hash: str) -> bool:
        """Check if solution is tabu."""
        return solution_hash in self.list
    
    def add(self, solution_hash: str) -> None:
        """Add solution to tabu list."""
        self.list.append(solution_hash)
    
    def clear(self) -> None:
        """Clear tabu list."""
        self.list.clear()
    
    @staticmethod
    def hash_solution(solution) -> str:
        """Create hash of solution for tabu tracking."""
        if isinstance(solution, list):
            return str(tuple(solution))
        return str(solution)


class NeighborhoodExplorer:
    """
    Explore neighborhood of a solution.
    For MVC: add or remove a single node at a time.
    """
    
    def __init__(self, encoding: Encoding, problem):
        self.encoding = encoding
        self.problem = problem
    
    def get_neighbors(self, solution) -> List[Tuple]:
        """
        Generate all neighbors by single node addition/removal.
        Returns list of (neighbor, move_type, affected_node) tuples.
        """
        neighbors = []
        encoding_name = self.encoding.get_name()
        
        if encoding_name == "BinaryEncoding":
            # For binary: try flipping each bit
            for i in range(len(solution)):
                neighbor = deepcopy(solution)
                neighbor[i] = 1 - neighbor[i]
                move_type = "flip"
                neighbors.append((neighbor, move_type, i))
        
        elif encoding_name == "SetEncoding":
            cover = set(solution)
            
            # Try removing each node in cover
            for node in cover:
                neighbor = [n for n in solution if n != node]
                neighbors.append((neighbor, "remove", node))
            
            # Try adding each node not in cover
            all_nodes = set(range(self.problem.num_nodes))
            for node in all_nodes - cover:
                neighbor = sorted(list(cover | {node}))
                neighbors.append((neighbor, "add", node))
        
        else:  # EdgeCentricEncoding
            # For edge-centric: flip each bit
            for i in range(len(solution)):
                neighbor = deepcopy(solution)
                neighbor[i] = 1 - neighbor[i]
                move_type = "flip"
                neighbors.append((neighbor, move_type, i))
        
        return neighbors


class TabuSearch:
    """
    Tabu Search for Minimum Vertex Cover
    
    Algorithm:
    1. Start with initial solution
    2. Explore neighborhood (all neighbor solutions)
    3. Choose best non-tabu neighbor (or tabu if aspiration criteria met)
    4. Add current to tabu list
    5. Update best solution found so far
    6. Repeat until termination
    
    Key features:
    - Tabu list prevents cycling back to recent solutions
    - Aspiration criteria allows tabu moves if they improve best
    - Explores full neighborhood at each step
    """
    
    def __init__(self,
                 problem,
                 encoding: Encoding,
                 fitness_func: FitnessFunction,
                 params: TSParams = None):
        """
        Initialize Tabu Search.
        
        Args:
            problem: MinimumVertexCoverProblem instance
            encoding: Encoding strategy
            fitness_func: Fitness function
            params: TS parameters
        """
        self.problem = problem
        self.encoding = encoding
        self.fitness_func = fitness_func
        self.params = params or TSParams()
        self.tabu_list = TabuList(self.params.tabu_list_size)
        self.neighborhood = NeighborhoodExplorer(encoding, problem)
        
        # Statistics
        self.iteration_stats = []
    
    def run(self) -> dict:
        """
        Run Tabu Search.
        
        Returns:
            Dictionary with results
        """
        # Initialize with random solution
        current_solution = self.encoding.random_solution(self.problem.num_nodes)
        current_cover = self.encoding.solution_to_cover(current_solution)
        current_fitness = self.fitness_func.evaluate(current_cover)
        
        # Track best solution
        best_solution = deepcopy(current_solution)
        best_cover = deepcopy(current_cover)
        best_fitness = current_fitness
        
        # Add initial solution to tabu list
        self.tabu_list.add(TabuList.hash_solution(current_solution))
        
        iteration = 0
        no_improvement_count = 0
        max_no_improvement = 100  # Stop if no improvement for 100 iterations
        
        while iteration < self.params.max_iterations and no_improvement_count < max_no_improvement:
            
            # Explore neighborhood
            neighbors = self.neighborhood.get_neighbors(current_solution)
            
            if not neighbors:
                break
            
            # Evaluate all neighbors
            best_neighbor = None
            best_neighbor_fitness = float('-inf')
            best_neighbor_tabu = False
            
            for neighbor, move_type, affected in neighbors:
                neighbor_hash = TabuList.hash_solution(neighbor)
                is_tabu = self.tabu_list.is_tabu(neighbor_hash)
                
                neighbor_cover = self.encoding.solution_to_cover(neighbor)
                neighbor_fitness = self.fitness_func.evaluate(neighbor_cover)
                
                # Aspiration criteria: accept tabu move if it improves best
                aspiration_met = (self.params.aspiration_criteria and 
                                neighbor_fitness > best_fitness)
                
                # Select neighbor if:
                # 1. Not tabu and better than current best neighbor
                # 2. Or tabu but aspiration criteria met
                if (not is_tabu and neighbor_fitness > best_neighbor_fitness) or \
                   (aspiration_met and neighbor_fitness > best_neighbor_fitness):
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
                    best_neighbor_tabu = is_tabu
            
            # No valid neighbor found
            if best_neighbor is None:
                break
            
            # Move to best neighbor
            current_solution = best_neighbor
            current_cover = self.encoding.solution_to_cover(current_solution)
            current_fitness = best_neighbor_fitness
            
            # Add to tabu list
            self.tabu_list.add(TabuList.hash_solution(current_solution))
            
            # Update best if improved
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = deepcopy(current_solution)
                best_cover = deepcopy(current_cover)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Store statistics
            self.iteration_stats.append({
                'iteration': iteration,
                'current_fitness': current_fitness,
                'best_fitness': best_fitness,
                'best_cover_size': len(best_cover),
                'tabu_size': len(self.tabu_list.list)
            })
            
            iteration += 1
        
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
    from encodings import SetEncoding
    from fitness import ConstraintPenalty
    
    # Test TS
    problem, _ = InstanceGenerator.generate_benchmark_instances()[1]
    encoding = SetEncoding()
    fitness = ConstraintPenalty(problem, penalty_weight=1.0)
    
    params = TSParams(tabu_list_size=30, max_iterations=1000)
    ts = TabuSearch(problem, encoding, fitness, params)
    
    result = ts.run()
    
    print(f"Best cover size: {result['best_cover_size']}")
    print(f"Is valid: {result['is_valid']}")
    print(f"Best fitness: {result['best_fitness']:.6f}")
    print(f"Total iterations: {result['total_iterations']}")
