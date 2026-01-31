"""
Advanced GA-Specific Fitness Functions for Minimum Vertex Cover

These functions are designed with GA population dynamics in mind:
- Diversity preservation through fitness sharing
- Adaptive penalties that evolve with population
- Multi-objective balancing
- Phenotypic diversity rewards
"""

from abc import ABC, abstractmethod
from typing import Set, List, Dict, Tuple
from src.problem import MinimumVertexCoverProblem
import numpy as np


class GAFitnessFunction(ABC):
    """Base class for GA-specific fitness functions."""
    
    def __init__(self, problem: MinimumVertexCoverProblem):
        self.problem = problem
        self.generation = 0
        self.best_valid_size = float('inf')
    
    @abstractmethod
    def evaluate(self, cover: Set[int]) -> float:
        """Evaluate fitness. Higher is better."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return function name."""
        pass
    
    def update_generation(self, generation: int):
        """Called at each generation for adaptive parameters."""
        self.generation = generation
    
    def update_best(self, best_valid_size: int):
        """Update best valid solution found."""
        if best_valid_size < self.best_valid_size:
            self.best_valid_size = best_valid_size


class FitnessSharingGA(GAFitnessFunction):
    """
    Fitness Sharing: Reduces fitness of similar solutions to promote diversity.
    
    Best for: Maintaining diversity while optimizing
    Formula: shared_fitness = base_fitness / (1 + σ * similar_solutions)
    
    σ: sharing radius (controls diversity pressure)
    """
    
    def __init__(self, problem: MinimumVertexCoverProblem, sigma: float = 0.3):
        super().__init__(problem)
        self.sigma = sigma
        self.population = []
        self.base_fitness_cache = {}
    
    def set_population(self, population: List[Set[int]]):
        """Set current population for sharing calculations."""
        self.population = population
        self.base_fitness_cache = {}
    
    def _base_fitness(self, cover: Set[int]) -> float:
        """Compute raw fitness (size minimization)."""
        is_valid = self.problem.is_valid_cover(cover)
        
        if is_valid:
            size = len(cover)
            # Normalize by problem size
            return 1.0 / (1.0 + size / self.problem.num_nodes)
        else:
            # Penalty for invalid - scale by uncovered edges
            uncovered = sum(1 for u, v in self.problem.edges 
                          if u not in cover and v not in cover)
            return -(1000 + uncovered * 100)
    
    def _similarity(self, cover1: Set[int], cover2: Set[int]) -> float:
        """Jaccard similarity between covers."""
        if len(cover1) == 0 and len(cover2) == 0:
            return 1.0
        intersection = len(cover1 & cover2)
        union = len(cover1 | cover2)
        return intersection / union if union > 0 else 0.0
    
    def evaluate(self, cover: Set[int]) -> float:
        """Evaluate with fitness sharing."""
        base = self._base_fitness(cover)
        
        if not self.population:
            return base
        
        # Count similar solutions
        similar_count = sum(1 for ind in self.population 
                          if self._similarity(cover, ind) > (1 - self.sigma))
        
        # Apply sharing
        shared = base / (1.0 + self.sigma * similar_count)
        return shared
    
    def get_name(self) -> str:
        return "FitnessSharingGA"


class AdaptivePenaltyGA(GAFitnessFunction):
    """
    Adaptive Penalty: Penalty increases with generations to prioritize validity.
    
    Best for: Guided progression toward feasibility
    Formula: fitness = size_fitness - (penalty_weight * generation/max_gen) * uncovered
    """
    
    def __init__(self, problem: MinimumVertexCoverProblem, max_generations: int = 300,
                 initial_penalty: float = 100, final_penalty: float = 10000):
        super().__init__(problem)
        self.max_generations = max_generations
        self.initial_penalty = initial_penalty
        self.final_penalty = final_penalty
    
    def _current_penalty(self) -> float:
        """Penalty increases linearly with generation."""
        progress = self.generation / max(self.max_generations, 1)
        return self.initial_penalty + progress * (self.final_penalty - self.initial_penalty)
    
    def evaluate(self, cover: Set[int]) -> float:
        """Evaluate with generation-adaptive penalty."""
        is_valid = self.problem.is_valid_cover(cover)
        size = len(cover)
        
        if is_valid:
            # Valid: reward small covers
            return 1.0 / (1.0 + size / self.problem.num_nodes)
        else:
            # Invalid: apply adaptive penalty
            uncovered = sum(1 for u, v in self.problem.edges 
                          if u not in cover and v not in cover)
            penalty = self._current_penalty()
            return -penalty * uncovered
    
    def get_name(self) -> str:
        return "AdaptivePenaltyGA"


class MultiObjectiveGA(GAFitnessFunction):
    """
    Multi-Objective: Balance between cover size and edge coverage.
    
    Best for: Exploring feasible region thoroughly
    Formula: fitness = α * coverage_ratio - β * (size / num_nodes)
    
    For invalid: coverage_ratio = covered_edges / total_edges
    For valid: coverage_ratio = 1.0
    """
    
    def __init__(self, problem: MinimumVertexCoverProblem, 
                 size_weight: float = 0.5, coverage_weight: float = 0.5):
        super().__init__(problem)
        self.size_weight = size_weight
        self.coverage_weight = coverage_weight
    
    def _edge_coverage(self, cover: Set[int]) -> float:
        """Fraction of edges covered."""
        covered = sum(1 for u, v in self.problem.edges 
                     if u in cover or v in cover)
        return covered / self.problem.num_edges if self.problem.num_edges > 0 else 0.0
    
    def evaluate(self, cover: Set[int]) -> float:
        """Multi-objective evaluation."""
        is_valid = self.problem.is_valid_cover(cover)
        coverage = self._edge_coverage(cover)
        size_penalty = len(cover) / self.problem.num_nodes
        
        if is_valid:
            # All edges covered, optimize size
            return self.coverage_weight * 1.0 - self.size_weight * size_penalty
        else:
            # Partial coverage - penalize but not too harshly
            return self.coverage_weight * coverage - self.size_weight * size_penalty - 1000
    
    def get_name(self) -> str:
        return "MultiObjectiveGA"


class PhenotypePreservingGA(GAFitnessFunction):
    """
    Phenotype Preserving: Rewards solutions that are valid AND different from best.
    
    Best for: Exploring plateau of valid solutions
    Formula: fitness = size_fitness + diversity_bonus if valid else penalty
    
    diversity_bonus: 1.0 / (1 + distance_to_best)
    distance: |this_cover - best_cover| / avg_size
    """
    
    def __init__(self, problem: MinimumVertexCoverProblem):
        super().__init__(problem)
        self.best_valid_cover = None
    
    def set_best_valid(self, best_cover: Set[int]):
        """Set the best valid solution found so far."""
        self.best_valid_cover = best_cover
    
    def _distance_to_best(self, cover: Set[int]) -> float:
        """Symmetric difference between covers, normalized."""
        if self.best_valid_cover is None:
            return 0.0
        
        diff = len(cover ^ self.best_valid_cover)  # Symmetric difference
        avg_size = (len(cover) + len(self.best_valid_cover)) / 2
        
        return diff / avg_size if avg_size > 0 else 0.0
    
    def evaluate(self, cover: Set[int]) -> float:
        """Evaluate with phenotype preservation."""
        is_valid = self.problem.is_valid_cover(cover)
        size = len(cover)
        
        if not is_valid:
            uncovered = sum(1 for u, v in self.problem.edges 
                          if u not in cover and v not in cover)
            return -(1000 + uncovered * 100)
        
        # Valid: base fitness + diversity bonus
        base_fitness = 1.0 / (1.0 + size / self.problem.num_nodes)
        
        if self.best_valid_cover is not None:
            distance = self._distance_to_best(cover)
            diversity_bonus = 0.1 / (1.0 + distance)  # Small bonus
            return base_fitness + diversity_bonus
        
        return base_fitness
    
    def get_name(self) -> str:
        return "PhenotypePreservingGA"


class RestartGA(GAFitnessFunction):
    """
    Restart Signal: Periodically signal to restart when stagnation detected.
    
    Best for: Escaping local optima
    Returns lower fitness when stagnation detected, triggering restart.
    """
    
    def __init__(self, problem: MinimumVertexCoverProblem, stagnation_window: int = 30):
        super().__init__(problem)
        self.stagnation_window = stagnation_window
        self.best_fitness_history = []
    
    def _is_stagnated(self) -> bool:
        """Check if best fitness hasn't improved recently."""
        if len(self.best_fitness_history) < self.stagnation_window:
            return False
        
        recent = self.best_fitness_history[-self.stagnation_window:]
        return recent[-1] == recent[0]  # No improvement
    
    def evaluate(self, cover: Set[int]) -> float:
        """Evaluate with restart signal."""
        is_valid = self.problem.is_valid_cover(cover)
        size = len(cover)
        
        if is_valid:
            fitness = 1.0 / (1.0 + size / self.problem.num_nodes)
        else:
            uncovered = sum(1 for u, v in self.problem.edges 
                          if u not in cover and v not in cover)
            fitness = -100000 - uncovered * 1000
        
        # If stagnated, reduce fitness to signal restart
        if self._is_stagnated():
            fitness *= 0.5
        
        return fitness
    
    def get_name(self) -> str:
        return "RestartGA"
