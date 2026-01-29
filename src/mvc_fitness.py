"""
Fitness Functions for Minimum Vertex Cover Problem
"""

from abc import ABC, abstractmethod
from typing import Set, List
from problem import MinimumVertexCoverProblem


class FitnessFunction(ABC):
    """Abstract base class for fitness functions."""
    
    def __init__(self, problem: MinimumVertexCoverProblem):
        self.problem = problem
    
    @abstractmethod
    def evaluate(self, cover: Set[int]) -> float:
        """
        Evaluate fitness of a cover.
        Higher is better (maximization problem).
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return name of this fitness function."""
        pass


class CoverSizeMinimization(FitnessFunction):
    """
    Fitness Function 1: Direct Cover Size Minimization with Validity Penalty
    
    For valid covers: Fitness = 1 / (1 + cover_size)
    For invalid covers: Fitness = -1000 (strong penalty)
    
    Advantages:
    - Direct optimization of cover size for valid solutions
    - Strong penalty discourages invalid covers
    - Guides algorithms toward feasible region first
    
    Disadvantages:
    - Penalty is discontinuous (may be hard for some algorithms)
    - Once feasible region is reached, optimization begins
    """
    
    def evaluate(self, cover: Set[int]) -> float:
        """
        Evaluate fitness as inverse of cover size for valid covers.
        Invalid covers receive harsh penalty.
        """
        is_valid = self.problem.is_valid_cover(cover)
        
        if not is_valid:
            # Strong penalty for invalid covers
            return -1000.0
        
        size = len(cover)
        # Normalize by problem size
        normalized_size = size / self.problem.num_nodes
        return 1.0 / (1.0 + normalized_size)
    
    def get_name(self) -> str:
        return "CoverSizeMinimization"


class ConstraintPenalty(FitnessFunction):
    """
    Fitness Function 2: Constraint-Based with Penalties
    
    Fitness = coverage_satisfaction - λ * (cover_size + uncovered_edges)
    
    Where:
    - coverage_satisfaction: 1 if valid, 0 otherwise
    - uncovered_edges: number of edges not covered
    - λ: penalty weight
    
    Advantages:
    - Explicitly penalizes invalid covers
    - Guides search toward feasible region
    - Balances feasibility and optimality
    
    Disadvantages:
    - Requires parameter tuning (λ)
    - More complex evaluation
    """
    
    def __init__(self, problem: MinimumVertexCoverProblem, penalty_weight: float = 1.0):
        super().__init__(problem)
        self.penalty_weight = penalty_weight
    
    def count_uncovered_edges(self, cover: Set[int]) -> int:
        """Count edges not covered by the solution."""
        uncovered = 0
        for u, v in self.problem.edges:
            if u not in cover and v not in cover:
                uncovered += 1
        return uncovered
    
    def evaluate(self, cover: Set[int]) -> float:
        """
        Evaluate with constraint penalty.
        """
        is_valid = self.problem.is_valid_cover(cover)
        uncovered = self.count_uncovered_edges(cover)
        cover_size = len(cover)
        
        if is_valid:
            # Valid cover: penalize size
            return 1.0 - (cover_size / self.problem.num_nodes) * 0.5
        else:
            # Invalid cover: heavily penalize
            penalty = self.penalty_weight * (uncovered + cover_size / self.problem.num_nodes)
            return max(0.0, 1.0 - penalty)
    
    def get_name(self) -> str:
        return "ConstraintPenalty"


class EdgeCoverageOptimization(FitnessFunction):
    """
    Fitness Function 3: Multi-Objective Edge Coverage
    
    Fitness = (covered_edges / total_edges) - λ * (cover_size / num_nodes)
    
    This balances two objectives:
    1. Maximize edge coverage (feasibility)
    2. Minimize cover size (optimality)
    
    Advantages:
    - Multi-objective perspective
    - Natural progression from infeasible to optimal
    - Guided search through feasible region
    
    Disadvantages:
    - Requires weight tuning (λ)
    - May not strongly prefer smaller covers within feasible solutions
    """
    
    def __init__(self, problem: MinimumVertexCoverProblem, size_weight: float = 0.3):
        super().__init__(problem)
        self.size_weight = size_weight
    
    def count_covered_edges(self, cover: Set[int]) -> int:
        """Count edges covered by the solution."""
        covered = 0
        for u, v in self.problem.edges:
            if u in cover or v in cover:
                covered += 1
        return covered
    
    def evaluate(self, cover: Set[int]) -> float:
        """
        Evaluate with edge coverage and size minimization.
        """
        covered_edges = self.count_covered_edges(cover)
        coverage_ratio = covered_edges / self.problem.num_edges if self.problem.num_edges > 0 else 0
        
        cover_size = len(cover)
        size_penalty = (cover_size / self.problem.num_nodes) * self.size_weight
        
        fitness = coverage_ratio - size_penalty
        return max(0.0, fitness)
    
    def get_name(self) -> str:
        return "EdgeCoverageOptimization"


class FitnessFunctionFactory:
    """Factory for creating fitness functions."""
    
    @staticmethod
    def create_cover_size_minimization(problem: MinimumVertexCoverProblem) -> CoverSizeMinimization:
        return CoverSizeMinimization(problem)
    
    @staticmethod
    def create_constraint_penalty(problem: MinimumVertexCoverProblem, 
                                  penalty_weight: float = 1.0) -> ConstraintPenalty:
        return ConstraintPenalty(problem, penalty_weight)
    
    @staticmethod
    def create_edge_coverage_optimization(problem: MinimumVertexCoverProblem,
                                         size_weight: float = 0.3) -> EdgeCoverageOptimization:
        return EdgeCoverageOptimization(problem, size_weight)
    
    @staticmethod
    def get_all_functions(problem: MinimumVertexCoverProblem):
        """Return all fitness functions."""
        return [
            CoverSizeMinimization(problem),
            ConstraintPenalty(problem, penalty_weight=1.0),
            EdgeCoverageOptimization(problem, size_weight=0.3)
        ]


if __name__ == "__main__":
    from problem import InstanceGenerator
    
    # Test fitness functions
    problem, _ = InstanceGenerator.generate_benchmark_instances()[0]
    
    functions = FitnessFunctionFactory.get_all_functions(problem)
    
    # Create some test covers
    test_covers = [
        set(),  # Empty
        set(range(problem.num_nodes)),  # All nodes
        set(range(problem.num_nodes // 2)),  # Half nodes
    ]
    
    for func in functions:
        print(f"\n{func.get_name()}:")
        for cover in test_covers:
            fitness = func.evaluate(cover)
            valid = problem.is_valid_cover(cover)
            print(f"  Cover size {len(cover):2d} (valid={valid}): {fitness:.4f}")
