"""
Three Distinct Encodings for Minimum Vertex Cover Problem
"""

from abc import ABC, abstractmethod
from typing import Set, List, Dict, Tuple
import numpy as np
import copy


class Encoding(ABC):
    """Abstract base class for MVC encodings."""
    
    @abstractmethod
    def solution_to_cover(self, solution) -> Set[int]:
        """Convert solution representation to vertex cover (set of node indices)."""
        pass
    
    @abstractmethod
    def cover_to_solution(self, cover: Set[int], num_nodes: int):
        """Convert vertex cover to solution representation."""
        pass
    
    @abstractmethod
    def random_solution(self, num_nodes: int):
        """Generate a random solution in this encoding."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return name of this encoding."""
        pass


class BinaryEncoding(Encoding):
    """
    Encoding 1: Binary Vector
    
    Representation: List of binary values [b0, b1, ..., b_n-1]
    where b_i = 1 if node i is in the cover, 0 otherwise.
    
    Advantages:
    - Simple, standard for genetic algorithms
    - Easy mutation and crossover
    - Direct interpretation
    
    Disadvantages:
    - May produce invalid covers requiring repair
    - No implicit constraint satisfaction
    """
    
    def solution_to_cover(self, solution: List[int]) -> Set[int]:
        """Convert binary vector to cover."""
        return set(i for i, val in enumerate(solution) if val == 1)
    
    def cover_to_solution(self, cover: Set[int], num_nodes: int) -> List[int]:
        """Convert cover to binary vector."""
        solution = [0] * num_nodes
        for node in cover:
            solution[node] = 1
        return solution
    
    def random_solution(self, num_nodes: int) -> List[int]:
        """Generate random binary vector with ~50% density."""
        return [np.random.randint(0, 2) for _ in range(num_nodes)]
    
    def get_name(self) -> str:
        return "BinaryEncoding"


class SetEncoding(Encoding):
    """
    Encoding 2: Set-Based Representation
    
    Representation: List of node indices currently in the cover
    e.g., [0, 2, 5, 7] represents nodes {0, 2, 5, 7}
    
    Advantages:
    - More compact for sparse covers
    - Natural set semantics
    - Easier to reason about cover size
    
    Disadvantages:
    - Variable length representation
    - More complex crossover operations
    - Requires position-independent operators
    """
    
    def solution_to_cover(self, solution: List[int]) -> Set[int]:
        """Convert set representation to cover."""
        return set(solution)
    
    def cover_to_solution(self, cover: Set[int], num_nodes: int) -> List[int]:
        """Convert cover to sorted list."""
        return sorted(list(cover))
    
    def random_solution(self, num_nodes: int) -> List[int]:
        """Generate random set covering ~50% of nodes."""
        num_selected = np.random.randint(1, max(2, num_nodes // 2))
        nodes = np.random.choice(num_nodes, size=num_selected, replace=False)
        return sorted(nodes.tolist())
    
    def get_name(self) -> str:
        return "SetEncoding"


class EdgeCentricEncoding(Encoding):
    """
    Encoding 3: Edge-Centric Representation
    
    Representation: List of binary values [e0, e1, ..., e_m-1]
    where e_i = 1 if edge i contributes to the cover decision
    Nodes are derived from edge endpoints.
    
    Advantages:
    - Directly works with edge constraints
    - Natural for edge-based fitness functions
    - May guide search toward feasible regions
    
    Disadvantages:
    - Requires graph structure to decode
    - More complex interpretation
    - Not all solutions may be feasible
    
    Decoding: For each edge marked as "active", at least one endpoint must be in cover
    """
    
    def __init__(self, edges: List[Tuple[int, int]]):
        """
        Initialize with edge list for decoding.
        
        Args:
            edges: List of (u, v) edge tuples
        """
        self.edges = edges
        self.num_edges = len(edges)
    
    def solution_to_cover(self, solution: List[int]) -> Set[int]:
        """
        Decode edge-centric solution to vertex cover.
        For each edge, at least one endpoint must be selected.
        """
        cover = set()
        for i, edge_active in enumerate(solution):
            if edge_active == 1 and i < len(self.edges):
                u, v = self.edges[i]
                # Greedy selection: add endpoint with higher degree to cover
                cover.add(u)
        return cover
    
    def cover_to_solution(self, cover: Set[int], num_nodes: int) -> List[int]:
        """
        Convert cover to edge-centric representation.
        Mark edges that are covered by the solution.
        """
        solution = [0] * self.num_edges
        for i, (u, v) in enumerate(self.edges):
            if u in cover or v in cover:
                solution[i] = 1
        return solution
    
    def random_solution(self, num_nodes: int) -> List[int]:
        """Generate random edge selection."""
        return [np.random.randint(0, 2) for _ in range(self.num_edges)]
    
    def get_name(self) -> str:
        return "EdgeCentricEncoding"


class EncodingFactory:
    """Factory for creating and managing encodings."""
    
    @staticmethod
    def create_binary_encoding() -> BinaryEncoding:
        return BinaryEncoding()
    
    @staticmethod
    def create_set_encoding() -> SetEncoding:
        return SetEncoding()
    
    @staticmethod
    def create_edge_centric_encoding(edges: List[Tuple[int, int]]) -> EdgeCentricEncoding:
        return EdgeCentricEncoding(edges)
    
    @staticmethod
    def get_all_encodings(edges: List[Tuple[int, int]]):
        """Return all three encodings."""
        return [
            BinaryEncoding(),
            SetEncoding(),
            EdgeCentricEncoding(edges)
        ]


if __name__ == "__main__":
    # Test encodings
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    
    binary_enc = BinaryEncoding()
    set_enc = SetEncoding()
    edge_enc = EdgeCentricEncoding(edges)
    
    # Test binary
    binary_sol = binary_enc.random_solution(4)
    print(f"Binary solution: {binary_sol}")
    cover = binary_enc.solution_to_cover(binary_sol)
    print(f"Binary cover: {cover}")
    
    # Test set
    set_sol = set_enc.random_solution(4)
    print(f"\nSet solution: {set_sol}")
    cover = set_enc.solution_to_cover(set_sol)
    print(f"Set cover: {cover}")
    
    # Test edge-centric
    edge_sol = edge_enc.random_solution(4)
    print(f"\nEdge solution: {edge_sol}")
    cover = edge_enc.solution_to_cover(edge_sol)
    print(f"Edge cover: {cover}")
