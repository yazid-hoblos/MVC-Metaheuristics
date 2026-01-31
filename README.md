# Minimum Vertex Cover Optimization using Meta-Heuristics

## Project Overview

This project implements and compares three meta-heuristic optimization algorithms for the Minimum Vertex Cover (MVC) problem:
- **Genetic Algorithm (GA)**
- **Simulated Annealing (SA)**
- **Tabu Search (TS)**

The study investigates three fundamentally distinct problem encodings and three different fitness functions to assess their impact on algorithm performance.

## Problem Description

**Minimum Vertex Cover**: Given an undirected graph G = (V, E), find the smallest subset C ⊆ V such that every edge (u,v) ∈ E has at least one endpoint in C.

This is an NP-complete problem with applications in:
- Bioinformatics: protein interaction networks, pathway analysis
- Network design: communication networks, sensor placement
- Resource allocation: minimum set cover variants

## Project Structure

```
minimum_vertex_cover/
├── src/                          # Core implementation
│   ├── problem.py                # Problem definition and instance generation
│   ├── mvc_encodings.py          # Three distinct problem encodings
│   ├── mvc_fitness.py            # Three fitness functions
│   ├── ga.py                     # Genetic Algorithm implementation
│   ├── sa.py                     # Simulated Annealing implementation
│   ├── ts.py                     # Tabu Search implementation
│   └── experiments.py            # Comprehensive experiment runner
├── report/
│   ├── main.tex                  # Main LaTeX report (5+ pages)
│   └── appendix.tex              # Pseudocode and implementation details
├── results/                      # Experiment results and CSV output
├── instances/                    # Generated benchmark instances
├── run_experiments.py            # Quick experiment runner
├── test_quick.py                 # Quick verification test
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Problem Encodings

### 1. Binary Vector Encoding
- **Representation**: [b₀, b₁, ..., b_n-1] where bᵢ ∈ {0,1}
- **Advantages**: Standard GA operators, intuitive interpretation
- **Disadvantages**: May encode infeasible solutions, requires repair/penalty

### 2. Set-Based Encoding
- **Representation**: [n₁, n₂, ..., n_k] list of selected nodes
- **Advantages**: Compact for sparse covers, natural set semantics
- **Disadvantages**: Variable length, requires specialized crossover

### 3. Edge-Centric Encoding
- **Representation**: [e₀, e₁, ..., e_m-1] marking which edges participate
- **Advantages**: Directly addresses edge constraints
- **Disadvantages**: Requires graph structure for decoding, less intuitive

## Fitness Functions

### 1. Cover Size Minimization
```
f(C) = 1 / (1 + |C|/n)
```
Simple direct optimization of cover size.

### 2. Constraint Penalty
```
f(C) = { 1.0 - 0.5·(|C|/n)           if C is valid
       { max(0, 1 - λ(u + |C|/n))    otherwise
```
Penalizes infeasible solutions; balances feasibility with optimality.

### 3. Edge Coverage Optimization
```
f(C) = covered_edges/total_edges - 0.3·(|C|/n)
```
Multi-objective: maximize edge coverage, minimize cover size.

## Installation

### Requirements
- Python 3.8+
- NumPy >= 1.20.0
- NetworkX >= 2.6
- Matplotlib >= 3.4.0 (for plotting)

### Setup
```bash
cd minimum_vertex_cover
pip install -r requirements.txt
```

## Usage

### Quick Test
Verify all components work correctly:
```bash
python test_quick.py
```

Expected output: Tests GA, SA, TS on a small instance with quick parameter settings.

### Run Experiments
```bash
# Run with 3 independent runs per configuration (quick testing)
python run_experiments.py

# Run with more iterations for thorough testing
python -c "
import sys
sys.path.insert(0, 'src')
from run_experiments import FastExperimentRunner
runner = FastExperimentRunner()
runner.run_experiments(num_runs=10)
runner.save_results()
runner.print_summary()
"
```

Results are saved to `results/` directory as CSV files.

### Export Instances (Optional)
```bash
python export_instances.py
```

Exports benchmark graphs to `instances/` as edge lists and metadata JSON files.

### Generate Plots
```bash
python generate_plots.py
```

Generates basic figures in `report/figures/` for inclusion in the report.

### Generate Advanced Plots (Comprehensive Analysis)
```bash
python generate_advanced_plots.py
```

Generates 8 advanced publication-quality plots:
- Performance profiles (cumulative distribution)
- Algorithm × Encoding heatmap
- Convergence curves by instance
- Box plot distributions
- Runtime vs quality trade-off
- Fitness function impact heatmap
- Feasibility comparison
- Instance difficulty analysis

### Generate GA Solution Animation
```bash
python animate_ga_solution.py
```

Generates an animation (MP4 or GIF) showing:
- GA solution evolution on graph visualization
- Nodes highlighted in cover (red) vs not in cover (teal)
- Convergence curve in parallel
- Perfect for presentations and understanding algorithm dynamics

### Custom Experiments
```python
import sys
sys.path.insert(0, 'src')

from problem import InstanceGenerator
from mvc_encodings import BinaryEncoding
from mvc_fitness import CoverSizeMinimization
from ga import GeneticAlgorithm, GAParams

# Load instance
problem, name = InstanceGenerator.generate_benchmark_instances()[0]

# Setup algorithm
encoding = BinaryEncoding()
fitness = CoverSizeMinimization(problem)
params = GAParams(population_size=100, generations=300)

# Run optimization
ga = GeneticAlgorithm(problem, encoding, fitness, params)
result = ga.run()

print(f"Best cover size: {result['best_cover_size']}")
print(f"Is valid: {result['is_valid']}")
print(f"Fitness: {result['best_fitness']:.6f}")
```

## Experimental Results Summary

**Performance Comparison (over 108 tests per algorithm, 5 runs each)**:

| Algorithm | Valid Solutions | Avg Cover Size | Std Dev | Min/Max | Avg Time |
|-----------|-----------------|----------------|---------|---------|----------|
| GA        | 23/108 (21%)    | 32.26 ± 21.99  | -       | 13/85   | 1.334s   |
| SA        | 11/108 (10%)    | 24.55 ± 12.52  | -       | 14/44   | 0.108s   |
| TS        | 11/108 (10%)    | 23.00 ± 4.71   | -       | 13/28   | 0.247s   |

**Key Findings**:
1. **Tabu Search** achieves best solution quality (smallest cover, lowest variance)
2. **Genetic Algorithm** finds most valid solutions but with higher variance
3. **Simulated Annealing** is fastest but has low feasibility
4. **Encoding choice** is secondary to algorithm selection (differences ≤ 2 nodes)
5. **Fitness function** significantly affects feasibility rates

## Algorithm Parameters

### Genetic Algorithm
- Population size: 100
- Generations: 300
- Mutation rate: 0.1
- Crossover rate: 0.8
- Selection: Tournament (size 3)
- Elitism: Top 5%

### Simulated Annealing
- Initial temperature: 100.0
- Cooling rate: 0.95
- Iterations per temperature: 50
- Minimum temperature: 0.01
- Max iterations: 5000

### Tabu Search
- Tabu list size: 50
- Max iterations: 5000
- Aspiration criteria: Enabled
- Early stopping: 100 iterations without improvement

## Implementation Notes

### Design Choices

1. **Three distinct encodings** rather than encoding variants
   - Tests fundamental representation differences
   - Validates MVC's robustness to encoding choice
   - Provides broader algorithmic insights

2. **Three meta-heuristics** representing different paradigms
   - GA: Population-based evolution
   - SA: Trajectory-based probabilistic acceptance
   - TS: Adaptive memory with full neighborhood search

3. **Random instance generation** for controlled experiments
   - Enables fair algorithm comparison
   - Better reproducibility than DIMACS benchmarks
   - Controlled instance scaling

4. **Conservative parameters** for realistic time constraints
   - Focus on algorithm behavior under practical limits
   - Avoids parameter overfitting to instances

### Limitations & Future Work

1. **Parameter tuning**: Full grid search over population size, mutation rate, temperature schedules
2. **Large-scale instances**: Test on 500-10,000 node graphs
3. **Hybrid approaches**: GA + local search, SA + TS refinement
4. **Statistical testing**: Wilcoxon signed-rank tests, ANOVA
5. **Domain knowledge**: Problem-specific enhancements, greedy initialization
6. **DIMACS benchmarks**: Compare against known solutions and approximation algorithms

## References

1. Kirkpatrick et al. (1983) - Simulated Annealing
2. Holland (1975) - Genetic Algorithms
3. Glover (1989, 1990) - Tabu Search
4. Garey & Johnson (1976) - NP-completeness of VC
5. Hochbaum (1983) - Approximation algorithms for Set Cover
6. Blum & Roli (2003) - Metaheuristics overview
7. Gendreau et al. (2010) - Metaheuristics for hard combinatorial problems
8. Additional citations in report
