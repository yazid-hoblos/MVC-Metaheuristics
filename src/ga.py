"""
Genetic Algorithm for Minimum Vertex Cover Problem
"""

import random
import numpy as np
from typing import List, Set, Tuple, Callable
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from mvc_encodings import Encoding
from mvc_fitness import FitnessFunction


@dataclass
class GAParams:
    """Genetic Algorithm parameters."""
    population_size: int = 100
    generations: int = 500
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_type: str = "tournament"  # tournament or roulette
    tournament_size: int = 3
    elitism_rate: float = 0.05  # Keep top 5% unchanged


class SelectionOperator(ABC):
    """Abstract base class for selection operators."""
    
    @abstractmethod
    def select(self, population: List, fitness_scores: List[float]) -> List:
        """Select individuals from population based on fitness."""
        pass


class TournamentSelection(SelectionOperator):
    """Tournament selection: randomly choose k individuals and pick best."""
    
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    def select(self, population: List, fitness_scores: List[float]) -> List:
        """Select an individual via tournament."""
        indices = random.sample(range(len(population)), self.tournament_size)
        best_idx = max(indices, key=lambda i: fitness_scores[i])
        return deepcopy(population[best_idx])


class RouletteWheelSelection(SelectionOperator):
    """Roulette wheel selection: probability proportional to fitness."""
    
    def select(self, population: List, fitness_scores: List[float]) -> List:
        """Select an individual via roulette wheel."""
        # Normalize fitness scores (shift to positive range)
        min_fitness = min(fitness_scores)
        adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_scores]
        
        total_fitness = sum(adjusted_fitness)
        if total_fitness <= 0:
            # Uniform selection if all fitness is zero
            return deepcopy(random.choice(population))
        
        probabilities = [f / total_fitness for f in adjusted_fitness]
        selected_idx = np.random.choice(len(population), p=probabilities)
        return deepcopy(population[selected_idx])


class CrossoverOperator(ABC):
    """Abstract base class for crossover operators."""
    
    @abstractmethod
    def crossover(self, parent1, parent2) -> Tuple:
        """Perform crossover between two parents."""
        pass


class UniformCrossover(CrossoverOperator):
    """Uniform crossover: each gene randomly inherited from either parent."""
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform uniform crossover."""
        if len(parent1) != len(parent2):
            # For variable-length representations (set encoding)
            return deepcopy(parent1), deepcopy(parent2)
        
        child1 = []
        child2 = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        
        return child1, child2


class SinglePointCrossover(CrossoverOperator):
    """Single-point crossover: split at random point and recombine."""
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform single-point crossover."""
        if len(parent1) != len(parent2):
            return deepcopy(parent1), deepcopy(parent2)
        
        if len(parent1) <= 1:
            return deepcopy(parent1), deepcopy(parent2)
        
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        return child1, child2


class MutationOperator(ABC):
    """Abstract base class for mutation operators."""
    
    @abstractmethod
    def mutate(self, individual, num_nodes: int) -> None:
        """Mutate an individual in-place."""
        pass


class BitFlipMutation(MutationOperator):
    """Bit-flip mutation: flip bits in binary encoding."""
    
    def __init__(self, mutation_rate: float = 0.1):
        self.mutation_rate = mutation_rate
    
    def mutate(self, individual: List[int], num_nodes: int) -> None:
        """Mutate by flipping bits with given probability."""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]


class SetMutation(MutationOperator):
    """Mutation for set-based encoding: add/remove nodes."""
    
    def __init__(self, mutation_rate: float = 0.1):
        self.mutation_rate = mutation_rate
    
    def mutate(self, individual: List[int], num_nodes: int) -> None:
        """Mutate by adding/removing nodes from set."""
        if random.random() < self.mutation_rate / 2:
            # Add a random node
            node = random.randint(0, num_nodes - 1)
            if node not in individual:
                individual.append(node)
                individual.sort()
        
        if random.random() < self.mutation_rate / 2:
            # Remove a random node from cover
            if individual:
                idx = random.randint(0, len(individual) - 1)
                individual.pop(idx)


class GeneticAlgorithm:
    """
    Genetic Algorithm for Minimum Vertex Cover
    
    Core Components:
    1. Population initialization
    2. Fitness evaluation
    3. Selection (tournament or roulette)
    4. Crossover (uniform or single-point)
    5. Mutation (bit-flip for binary, add/remove for set)
    6. Elitism (preserve best solutions)
    """
    
    def __init__(self,
                 problem,
                 encoding: Encoding,
                 fitness_func: FitnessFunction,
                 params: GAParams = None):
        """
        Initialize Genetic Algorithm.
        
        Args:
            problem: MinimumVertexCoverProblem instance
            encoding: Encoding strategy
            fitness_func: Fitness function
            params: GA parameters
        """
        self.problem = problem
        self.encoding = encoding
        self.fitness_func = fitness_func
        self.params = params or GAParams()
        
        # Select operators based on encoding type
        encoding_name = encoding.get_name()
        if encoding_name == "BinaryEncoding":
            self.mutation = BitFlipMutation(self.params.mutation_rate)
            self.crossover = UniformCrossover()
        elif encoding_name == "SetEncoding":
            self.mutation = SetMutation(self.params.mutation_rate)
            self.crossover = SinglePointCrossover()
        else:  # EdgeCentricEncoding
            self.mutation = BitFlipMutation(self.params.mutation_rate)
            self.crossover = UniformCrossover()
        
        # Selection operator
        if self.params.selection_type == "tournament":
            self.selection = TournamentSelection(self.params.tournament_size)
        else:
            self.selection = RouletteWheelSelection()
        
        # Statistics
        self.generation_stats = []
    
    def initialize_population(self) -> List:
        """Initialize population with random solutions."""
        population = []
        for _ in range(self.params.population_size):
            individual = self.encoding.random_solution(self.problem.num_nodes)
            population.append(individual)
        return population
    
    def evaluate_population(self, population: List) -> List[float]:
        """Evaluate fitness of entire population."""
        fitness_scores = []
        for individual in population:
            cover = self.encoding.solution_to_cover(individual)
            fitness = self.fitness_func.evaluate(cover)
            fitness_scores.append(fitness)
        return fitness_scores
    
    def select_parents(self, population: List, fitness_scores: List[float]) -> Tuple:
        """Select two parents for reproduction."""
        parent1 = self.selection.select(population, fitness_scores)
        parent2 = self.selection.select(population, fitness_scores)
        return parent1, parent2
    
    def create_offspring(self, parent1, parent2) -> Tuple:
        """Create offspring from parents via crossover and mutation."""
        # Crossover
        if random.random() < self.params.crossover_rate:
            child1, child2 = self.crossover.crossover(parent1, parent2)
        else:
            child1, child2 = deepcopy(parent1), deepcopy(parent2)
        
        # Mutation
        self.mutation.mutate(child1, self.problem.num_nodes)
        self.mutation.mutate(child2, self.problem.num_nodes)
        
        return child1, child2
    
    def apply_elitism(self, population: List, fitness_scores: List[float]) -> Tuple:
        """
        Preserve best individuals for next generation.
        """
        num_elites = max(1, int(self.params.population_size * self.params.elitism_rate))
        
        # Get indices of best individuals
        elite_indices = sorted(range(len(fitness_scores)), 
                              key=lambda i: fitness_scores[i], 
                              reverse=True)[:num_elites]
        
        elites = [deepcopy(population[i]) for i in elite_indices]
        elite_fitness = [fitness_scores[i] for i in elite_indices]
        
        return elites, elite_fitness
    
    def run(self, max_generations: int = None) -> dict:
        """
        Run genetic algorithm.
        
        Returns:
            Dictionary with results:
            - best_solution: best individual found
            - best_fitness: best fitness value
            - best_cover: vertex cover (set of nodes)
            - generation_stats: statistics per generation
        """
        max_generations = max_generations or self.params.generations
        
        population = self.initialize_population()
        
        best_individual = None
        best_fitness = float('-inf')
        best_cover = None
        
        for generation in range(max_generations):
            # Evaluate population
            fitness_scores = self.evaluate_population(population)
            
            # Track best solution
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            gen_best_cover = self.encoding.solution_to_cover(population[gen_best_idx])
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = deepcopy(population[gen_best_idx])
                best_cover = self.encoding.solution_to_cover(best_individual)
            
            # Store statistics
            self.generation_stats.append({
                'generation': generation,
                'best_fitness': max(fitness_scores),
                'avg_fitness': np.mean(fitness_scores),
                'std_fitness': np.std(fitness_scores),
                'best_cover_size': len(best_cover),
                'gen_best_cover': gen_best_cover,
                'best_cover_so_far': deepcopy(best_cover)
            })
            
            # Apply elitism
            elites, elite_fitness = self.apply_elitism(population, fitness_scores)
            
            # Create new population through selection and reproduction
            new_population = deepcopy(elites)
            
            while len(new_population) < self.params.population_size:
                parent1, parent2 = self.select_parents(population, fitness_scores)
                child1, child2 = self.create_offspring(parent1, parent2)
                new_population.append(child1)
                if len(new_population) < self.params.population_size:
                    new_population.append(child2)
            
            # Trim to exact population size
            population = new_population[:self.params.population_size]
        
        return {
            'best_solution': best_individual,
            'best_fitness': best_fitness,
            'best_cover': best_cover,
            'best_cover_size': len(best_cover),
            'is_valid': self.problem.is_valid_cover(best_cover),
            'generation_stats': self.generation_stats
        }


if __name__ == "__main__":
    from problem import InstanceGenerator
    from encodings import BinaryEncoding
    from fitness import CoverSizeMinimization
    
    # Test GA
    problem, _ = InstanceGenerator.generate_benchmark_instances()[0]
    encoding = BinaryEncoding()
    fitness = CoverSizeMinimization(problem)
    
    params = GAParams(population_size=50, generations=100)
    ga = GeneticAlgorithm(problem, encoding, fitness, params)
    
    result = ga.run()
    
    print(f"Best cover size: {result['best_cover_size']}")
    print(f"Is valid: {result['is_valid']}")
    print(f"Best fitness: {result['best_fitness']:.6f}")
