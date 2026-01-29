"""
Quick test to verify all components work
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from problem import InstanceGenerator, MinimumVertexCoverProblem
from mvc_encodings import EncodingFactory, BinaryEncoding, SetEncoding
from mvc_fitness import FitnessFunctionFactory, CoverSizeMinimization, ConstraintPenalty
from ga import GeneticAlgorithm, GAParams
from sa import SimulatedAnnealing, SAParams
from ts import TabuSearch, TSParams


def test_basic_functionality():
    """Test that all components work together."""
    print("="*60)
    print("TESTING MINIMUM VERTEX COVER META-HEURISTICS")
    print("="*60)
    
    # Generate small test instance
    print("\n1. Generating test instance...")
    problem, name = InstanceGenerator.generate_benchmark_instances()[0]
    print(f"   Instance: {name}")
    print(f"   Nodes: {problem.num_nodes}, Edges: {problem.num_edges}")
    
    # Test encodings
    print("\n2. Testing encodings...")
    encodings = EncodingFactory.get_all_encodings(problem.edges)
    for enc in encodings:
        sol = enc.random_solution(problem.num_nodes)
        cover = enc.solution_to_cover(sol)
        is_valid = problem.is_valid_cover(cover)
        print(f"   {enc.get_name():20s}: cover_size={len(cover):2d}, valid={is_valid}")
    
    # Test fitness functions
    print("\n3. Testing fitness functions...")
    fitness_funcs = FitnessFunctionFactory.get_all_functions(problem)
    test_cover = set(range(problem.num_nodes // 2))
    for ff in fitness_funcs:
        fitness = ff.evaluate(test_cover)
        print(f"   {ff.get_name():30s}: fitness={fitness:.6f}")
    
    # Test GA
    print("\n4. Testing Genetic Algorithm...")
    encoding = BinaryEncoding()
    fitness = CoverSizeMinimization(problem)
    params = GAParams(population_size=50, generations=50)
    ga = GeneticAlgorithm(problem, encoding, fitness, params)
    result = ga.run()
    print(f"   GA Result: cover_size={result['best_cover_size']}, "
          f"valid={result['is_valid']}, fitness={result['best_fitness']:.6f}")
    
    # Test SA
    print("\n5. Testing Simulated Annealing...")
    encoding = SetEncoding()
    fitness = ConstraintPenalty(problem)
    params = SAParams(max_iterations=1000)
    sa = SimulatedAnnealing(problem, encoding, fitness, params)
    result = sa.run()
    print(f"   SA Result: cover_size={result['best_cover_size']}, "
          f"valid={result['is_valid']}, fitness={result['best_fitness']:.6f}")
    
    # Test TS
    print("\n6. Testing Tabu Search...")
    encoding = SetEncoding()
    fitness = CoverSizeMinimization(problem)
    params = TSParams(max_iterations=500)
    ts = TabuSearch(problem, encoding, fitness, params)
    result = ts.run()
    print(f"   TS Result: cover_size={result['best_cover_size']}, "
          f"valid={result['is_valid']}, fitness={result['best_fitness']:.6f}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_basic_functionality()
