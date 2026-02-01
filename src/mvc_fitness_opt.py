"""
Corrected Fitness Functions for Minimum Vertex Cover (MVC).
Strategies: Soft Constraints, Adaptive Weighting, and Repair Heuristics.
"""

from abc import ABC, abstractmethod
from typing import Set, List, Dict, Tuple
from problem import MinimumVertexCoverProblem
import random

class FitnessFunction(ABC):
    """Abstract base class for fitness functions."""
    
    def __init__(self, problem: MinimumVertexCoverProblem):
        self.problem = problem
    
    @abstractmethod
    def evaluate(self, cover: Set[int]) -> float:
        """
        Evaluate fitness of a cover.
        Returns a float where HIGHER is BETTER.
        Note: Internally these minimize cost, so we return -cost.
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return name of this fitness function."""
        pass


class LinearSoftPenalty(FitnessFunction):
    """
    Fitness Function 1: Linear Soft Penalty (Best for Simulated Annealing)
    
    Instead of a cliff (valid vs invalid), this creates a smooth slope.
    
    Formula: Fitness = -( |C| + w * |E_unc| )
    
    where w > 1.0. This ensures that covering an edge is always 
    mathematically 'cheaper' than removing a node, guiding the 
    algorithm naturally toward valid solutions.
    """
    
    def __init__(self, problem: MinimumVertexCoverProblem, penalty_multiplier: float = 1.2):
        super().__init__(problem)
        # Multiplier must be > 1.0 to guarantee the optimal solution is a valid cover.
        self.penalty_multiplier = penalty_multiplier

    def evaluate(self, cover: Set[int]) -> float:
        uncovered_count = 0
        
        # Efficient single-pass counting
        for u, v in self.problem.edges:
            if u not in cover and v not in cover:
                uncovered_count += 1
        
        size = len(cover)
        
        # Total Cost = Size + Penalty for breaking constraints
        cost = size + (self.penalty_multiplier * uncovered_count)
        
        # Return negative cost (Maximization: -10 is better than -15)
        return -cost

    def get_name(self) -> str:
        return "LinearSoftPenalty"


class AdaptiveEdgeWeighting(FitnessFunction):
    """
    Fitness Function 2: Adaptive Edge Weighting (Best for Tabu Search)
    
    Based on the 'Breakout Local Search' strategy.
    
    Formula: Fitness = -( |C| + sum(weight(e) for e in E_unc) )
    
    Includes a mechanism to increase weights of 'hard' edges that
    remain uncovered for too long, helping escape local optima.
    """
    
    def __init__(self, problem: MinimumVertexCoverProblem):
        super().__init__(problem)
        # Initialize all edge weights to 1.0
        # Map edge tuples to weights
        self.edge_weights: Dict[Tuple[int, int], float] = {
            edge: 1.0 for edge in self.problem.edges
        }
    
    def update_weights(self, current_cover: Set[int], increment: float = 1.0):
        """
        CRITICAL: Call this method when the solver is stuck (stagnation).
        Increases penalty for currently uncovered edges.
        """
        for edge in self.problem.edges:
            u, v = edge
            if u not in current_cover and v not in current_cover:
                self.edge_weights[edge] += increment

    def evaluate(self, cover: Set[int]) -> float:
        penalty_score = 0.0
        
        for edge in self.problem.edges:
            u, v = edge
            if u not in cover and v not in cover:
                # Add the specific weight of this uncovered edge
                penalty_score += self.edge_weights[edge]
        
        size = len(cover)
        cost = size + penalty_score
        return -cost

    def get_name(self) -> str:
        return "AdaptiveEdgeWeighting"


class RepairBasedFitness(FitnessFunction):
    """
    Fitness Function 3: Repair-Based / Lamarckian (Best for Genetic Algorithms)
    
    GAs often produce invalid offspring. Instead of assigning a penalty,
    this function 'repairs' the solution greedily to making it valid, 
    then evaluates the size of that valid solution.
    
    Formula: Fitness = -( |Repair(C)| )
    """
    
    def __init__(self, problem: MinimumVertexCoverProblem):
        super().__init__(problem)
        # Pre-calculate adjacency for fast greedy decisions
        self.adj = {i: [] for i in range(problem.num_nodes)}
        for u, v in problem.edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

    def _repair(self, cover: Set[int]) -> int:
        """
        Simulate a repair of the cover and return the NEW size.
        Does not modify the input set permanently (unless desired).
        """
        # Create a working copy
        temp_cover = cover.copy()
        
        # Find all uncovered edges
        uncovered_edges = []
        for u, v in self.problem.edges:
            if u not in temp_cover and v not in temp_cover:
                uncovered_edges.append((u, v))
        
        if not uncovered_edges:
            return len(temp_cover)
            
        # Greedy Heuristic: Cover edges using the node with higher degree
        for u, v in uncovered_edges:
            if u in temp_cover or v in temp_cover:
                continue # Already covered by a previous step
            
            # Pick the node with higher degree (greedy choice)
            if len(self.adj[u]) > len(self.adj[v]):
                temp_cover.add(u)
            else:
                temp_cover.add(v)
                
        return len(temp_cover)

    def evaluate(self, cover: Set[int]) -> float:
        # The fitness is based on the POTENTIAL valid solution
        repaired_size = self._repair(cover)
        return -float(repaired_size)

    def get_name(self) -> str:
        return "RepairBasedFitness"


class FitnessFunctionFactory:
    """Factory for creating fitness functions."""
    
    @staticmethod
    def get_linear_penalty(problem) -> LinearSoftPenalty:
        return LinearSoftPenalty(problem)
    
    @staticmethod
    def get_adaptive(problem) -> AdaptiveEdgeWeighting:
        return AdaptiveEdgeWeighting(problem)
        
    @staticmethod
    def get_repair_based(problem) -> RepairBasedFitness:
        return RepairBasedFitness(problem)

    @staticmethod
    def get_all_functions(problem):
        return [
            LinearSoftPenalty(problem),
            AdaptiveEdgeWeighting(problem),
            RepairBasedFitness(problem)
        ]