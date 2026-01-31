"""
Minimum Vertex Cover Problem Definition and Instance Generation
"""

import random
import networkx as nx
from typing import Set, Tuple, List
import numpy as np


class MinimumVertexCoverProblem:
    """
    Minimum Vertex Cover Problem (MVC):
    Given an undirected graph G = (V, E), find the smallest subset C ⊆ V such that
    every edge (u, v) ∈ E has at least one endpoint in C.
    
    This is an NP-complete problem relevant to bioinformatics:
    - Protein interaction networks: find minimal set of proteins to cover all interactions
    - Network analysis: identify critical nodes in biological networks
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize MVC problem with a graph.
        
        Args:
            graph: NetworkX Graph object
        """
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()
        self.edges = list(graph.edges())
        
    def is_valid_cover(self, cover: Set[int]) -> bool:
        """
        Check if a given set of nodes is a valid vertex cover.
        
        Args:
            cover: Set of node indices
            
        Returns:
            True if all edges are covered, False otherwise
        """
        for u, v in self.edges:
            if u not in cover and v not in cover:
                return False
        return True
    
    def cover_size(self, cover: Set[int]) -> int:
        """Return the size of the cover."""
        return len(cover)

    def repair_cover(self, cover: Set[int]) -> Set[int]:
        """
        Repair an invalid cover by greedily adding endpoints of uncovered edges.

        This guarantees feasibility, though not minimality.
        """
        repaired = set(cover)
        for u, v in self.edges:
            if u not in repaired and v not in repaired:
                # Add endpoint with higher degree to cover more edges
                if self.graph.degree[u] >= self.graph.degree[v]:
                    repaired.add(u)
                else:
                    repaired.add(v)
        return repaired
    
    def get_graph_info(self) -> dict:
        """Return graph statistics."""
        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'density': nx.density(self.graph),
            'avg_degree': 2 * self.num_edges / self.num_nodes if self.num_nodes > 0 else 0
        }


class InstanceGenerator:
    """Generate random Minimum Vertex Cover instances of varying sizes."""
    
    @staticmethod
    def generate_random_graph(num_nodes: int, edge_probability: float, 
                             seed: int = None) -> MinimumVertexCoverProblem:
        """
        Generate a random graph using Erdős–Rényi model.
        
        Args:
            num_nodes: Number of nodes
            edge_probability: Probability of edge between any two nodes (0-1)
            seed: Random seed for reproducibility
            
        Returns:
            MinimumVertexCoverProblem instance
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        graph = nx.erdos_renyi_graph(num_nodes, edge_probability)
        return MinimumVertexCoverProblem(graph)
    
    @staticmethod
    def generate_scale_free_graph(num_nodes: int, m: int = 2, 
                                  seed: int = None) -> MinimumVertexCoverProblem:
        """
        Generate a scale-free graph (Barabási–Albert model).
        More realistic for biological networks.
        
        Args:
            num_nodes: Number of nodes
            m: Number of edges to attach from new node to existing nodes
            seed: Random seed
            
        Returns:
            MinimumVertexCoverProblem instance
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        graph = nx.barabasi_albert_graph(num_nodes, m)
        return MinimumVertexCoverProblem(graph)
    
    @staticmethod
    def generate_benchmark_instances() -> List[Tuple[MinimumVertexCoverProblem, str]]:
        """
        Generate a set of benchmark instances at different sizes.
        Sizes: small (20), medium (50), large (100)
        
        Returns:
            List of (problem, name) tuples
        """
        instances = []
        
        # Small instance
        problem_small = InstanceGenerator.generate_random_graph(
            num_nodes=20, edge_probability=0.3, seed=42
        )
        instances.append((problem_small, "small_20nodes"))
        
        # Medium instance
        problem_medium = InstanceGenerator.generate_random_graph(
            num_nodes=50, edge_probability=0.25, seed=43
        )
        instances.append((problem_medium, "medium_50nodes"))
        
        # Large instance
        problem_large = InstanceGenerator.generate_random_graph(
            num_nodes=100, edge_probability=0.15, seed=44
        )
        instances.append((problem_large, "large_100nodes"))
        
        # Scale-free (bioinformatics-inspired)
        problem_scale_free = InstanceGenerator.generate_scale_free_graph(
            num_nodes=50, m=3, seed=45
        )
        instances.append((problem_scale_free, "scale_free_50nodes"))
        
        return instances


if __name__ == "__main__":
    # Test instance generation
    instances = InstanceGenerator.generate_benchmark_instances()
    for problem, name in instances:
        info = problem.get_graph_info()
        print(f"\n{name}:")
        print(f"  Nodes: {info['num_nodes']}, Edges: {info['num_edges']}")
        print(f"  Density: {info['density']:.4f}, Avg Degree: {info['avg_degree']:.2f}")
