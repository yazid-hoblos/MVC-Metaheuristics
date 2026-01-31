"""
Advanced GA-Specific Encodings for Minimum Vertex Cover

Encodings designed to work well with GA operators (crossover/mutation):
- Degree-biased binary: Favors high-degree nodes
- Greedy-initialization hybrid: Starts with greedy solution
- Adaptive threshold: Dynamic threshold based on degree
- Permutation-based: Rank nodes and cover top-k
"""

from typing import Set, List, Tuple
from abc import ABC, abstractmethod
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from problem import MinimumVertexCoverProblem
from mvc_encodings import Encoding
import random
import numpy as np


class DegreeBiasedBinary(Encoding):
    """
    Binary encoding with degree bias: High-degree nodes more likely to be in cover.
    
    Genotype: Binary string where bit i represents if node i is selected.
    Initialization: Probability of selection = degree(i) / max_degree
    
    Advantage: Initializes solutions closer to good feasible region
    """
    
    def __init__(self, problem: MinimumVertexCoverProblem):
        self.problem = problem
        # Precompute node degrees
        self.degree = {i: 0 for i in range(problem.num_nodes)}
        for u, v in problem.edges:
            self.degree[u] += 1
            self.degree[v] += 1
        self.max_degree = max(self.degree.values()) if self.degree else 1
    
    def solution_to_cover(self, solution: List[int]) -> Set[int]:
        """Convert binary string to cover."""
        return {i for i in range(self.problem.num_nodes) if solution[i] == 1}
    
    def cover_to_solution(self, cover: Set[int], num_nodes: int) -> List[int]:
        """Convert cover to binary string."""
        return [1 if i in cover else 0 for i in range(num_nodes)]
    
    def random_solution(self, num_nodes: int) -> List[int]:
        """Generate random solution with degree bias."""
        solution = []
        for i in range(num_nodes):
            # Probability of inclusion proportional to degree
            prob = self.degree[i] / self.max_degree if self.max_degree > 0 else 0.5
            solution.append(1 if random.random() < prob else 0)
        return solution
    
    def get_name(self) -> str:
        return "DegreeBiasedBinary"


class GreedyHybridEncoding(Encoding):
    """
    Hybrid encoding: Genotype represents deviations from greedy solution.
    
    Base: Greedy solution (cover all edges with minimum nodes)
    Genotype: Binary indicating which nodes to add/remove from base
    
    Advantage: Starts very close to good solution, mutations are small tweaks
    """
    
    def __init__(self, problem: MinimumVertexCoverProblem):
        self.problem = problem
        # Precompute node degrees
        self.degree = {i: 0 for i in range(problem.num_nodes)}
        for u, v in problem.edges:
            self.degree[u] += 1
            self.degree[v] += 1
        self.greedy_base = self._compute_greedy_base()
    
    def _compute_greedy_base(self) -> Set[int]:
        """Compute a greedy initial cover (approximation)."""
        cover = set()
        uncovered_edges = list(self.problem.edges)
        
        while uncovered_edges:
            # Pick node that covers most edges
            best_node = None
            best_count = 0
            
            for node in range(self.problem.num_nodes):
                if node in cover:
                    continue
                count = sum(1 for u, v in uncovered_edges if u == node or v == node)
                if count > best_count:
                    best_count = count
                    best_node = node
            
            if best_node is None:
                break
            
            cover.add(best_node)
            uncovered_edges = [(u, v) for u, v in uncovered_edges 
                              if u not in cover and v not in cover]
        
        return cover
    
    def solution_to_cover(self, solution: List[int]) -> Set[int]:
        """Decode from deviation genotype."""
        cover = set()
        for i in range(self.problem.num_nodes):
            if i in self.greedy_base:
                # Keep if not removed
                if solution[i] == 1:
                    cover.add(i)
            else:
                # Add if added
                if solution[i] == 1:
                    cover.add(i)
        return cover
    
    def cover_to_solution(self, cover: Set[int], num_nodes: int) -> List[int]:
        """Encode as deviation from greedy base."""
        solution = []
        for i in range(num_nodes):
            if i in self.greedy_base:
                # In base: 1 means keep, 0 means remove
                solution.append(1 if i in cover else 0)
            else:
                # Not in base: 1 means add, 0 means don't add
                solution.append(1 if i in cover else 0)
        return solution
    
    def random_solution(self, num_nodes: int) -> List[int]:
        """Generate random solution based on greedy base."""
        # Start with greedy base and randomly add/remove nodes
        solution = self.cover_to_solution(self.greedy_base, num_nodes)
        # Add some randomness
        for i in range(len(solution)):
            if random.random() < 0.1:  # 10% chance to flip
                solution[i] = 1 - solution[i]
        return solution
    
    def get_name(self) -> str:
        return "GreedyHybridEncoding"


class AdaptiveThresholdEncoding(Encoding):
    """
    Adaptive Threshold: Genotype is a threshold value T.
    Phenotype: Cover all nodes with degree >= T.
    
    Advantage: Reduces genotype complexity, captures degree-based structure
    Genotype: Single value representing threshold percentile
    """
    
    def __init__(self, problem: MinimumVertexCoverProblem):
        self.problem = problem
        # Precompute node degrees
        self.degree = {i: 0 for i in range(problem.num_nodes)}
        for u, v in problem.edges:
            self.degree[u] += 1
            self.degree[v] += 1
        self.sorted_degrees = sorted(set(self.degree.values()))
        self.max_degree = max(self.degree.values()) if self.degree else 1
    
    def solution_to_cover(self, solution: List) -> Set[int]:
        """Decode threshold to cover."""
        if not solution or len(solution) == 0:
            return set()
        threshold = max(0, min(solution[0], self.max_degree))
        cover = {i for i in range(self.problem.num_nodes) 
                if self.degree[i] >= threshold}
        return cover
    
    def cover_to_solution(self, cover: Set[int], num_nodes: int) -> List:
        """Encode cover as threshold."""
        # Try to infer threshold from cover
        degrees_in_cover = [self.degree[i] for i in cover]
        if degrees_in_cover:
            threshold = min(degrees_in_cover)
        else:
            threshold = 0
        return [float(threshold)]
    
    def random_solution(self, num_nodes: int) -> List:
        """Random threshold."""
        threshold = random.uniform(0, self.max_degree)
        return [threshold]
    
    def get_name(self) -> str:
        return "AdaptiveThresholdEncoding"


class PermutationEncoding(Encoding):
    """
    Permutation-based: Genotype is random permutation of nodes.
    Phenotype: Cover top-k nodes from permutation (k inferred from edges needed).
    
    Advantage: Natural genetic operators (partially mapped crossover, PMX)
    """
    
    def __init__(self, problem: MinimumVertexCoverProblem):
        self.problem = problem
        # Precompute node degrees
        self.degree = {i: 0 for i in range(problem.num_nodes)}
        for u, v in problem.edges:
            self.degree[u] += 1
            self.degree[v] += 1
        self.node_order = list(range(problem.num_nodes))
    
    def solution_to_cover(self, solution: List[int]) -> Set[int]:
        """Decode: take enough nodes from permutation to cover all edges."""
        cover = set()
        uncovered_edges = list(range(len(self.problem.edges)))
        
        # Greedy: add nodes from permutation until all edges covered
        for node in solution:
            if not uncovered_edges:
                break
            
            cover.add(node)
            # Remove edges covered by this node
            new_uncovered = []
            for edge_idx in uncovered_edges:
                u, v = self.problem.edges[edge_idx]
                if u != node and v != node:
                    new_uncovered.append(edge_idx)
            uncovered_edges = new_uncovered
        
        # If still uncovered edges, add remaining nodes greedily
        if uncovered_edges:
            for i in range(len(solution)):
                if i >= len(solution):
                    break
                cover.add(solution[i])
        
        return cover
    
    def cover_to_solution(self, cover: Set[int], num_nodes: int) -> List[int]:
        """Encode as permutation (order by degree, with cover first)."""
        # Put cover nodes first (sorted by degree), then others
        cover_nodes = sorted(cover, key=lambda i: -self.degree[i])
        other_nodes = sorted([i for i in range(num_nodes) if i not in cover],
                            key=lambda i: -self.degree[i])
        return cover_nodes + other_nodes
    
    def random_solution(self, num_nodes: int) -> List[int]:
        """Random permutation."""
        perm = list(range(num_nodes))
        random.shuffle(perm)
        return perm
    
    def get_name(self) -> str:
        return "PermutationEncoding"
