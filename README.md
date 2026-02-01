# Minimum Vertex Cover Optimization using Meta-Heuristics

[![MVC-Metaheuristics](https://img.shields.io/badge/GitHub-MVC--Metaheuristics-blue)](https://github.com/yazid-hoblos/MVC-Metaheuristics)
[![ENGA Framework](https://img.shields.io/badge/GitHub-ENGA-green)](https://github.com/yazid-hoblos/ENGA)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)

## Overview

This project implements and compares three meta-heuristic algorithms for solving the **Minimum Vertex Cover (MVC)** problem:
- **Genetic Algorithm (GA)** - Population-based evolutionary optimization
- **Simulated Annealing (SA)** - Probabilistic trajectory-based search  
- **Tabu Search (TS)** - Adaptive memory-based local search

The study evaluates **3 problem encodings** (binary vector, set-based, edge-centric) and **3 fitness function classes** (original hard-penalty and optimized smooth-landscape functions) across 12 benchmark graph instances.

**Key Finding**: Set-based encoding with edge coverage fitness achieves the best balance of solution quality and feasibility. Optimized fitness functions (smooth penalties, adaptive weighting, repair-based) significantly improve search landscape navigability.

### Genetic Algorithm in Action

<p align="center">
  <img src="report/figures/ga_animation_small.gif" alt="GA Solution Evolution" width="600"/>
</p>

*Genetic Algorithm evolution on a small MVC instance showing convergence from random initialization to optimal vertex cover (red nodes). The animation demonstrates population-based search dynamics and fitness improvement over generations.*

## Problem Definition

**Minimum Vertex Cover**: Given graph $G = (V, E)$, find the smallest subset $C \subseteq V$ such that every edge $(u,v) \in E$ has at least one endpoint in $C$.

- **Complexity**: NP-complete
- **Applications**: Network design, bioinformatics, resource allocation

## Repository Structure

```
minimum_vertex_cover/
├── src/
│   ├── problem.py              # MVC problem definition and instance generator
│   ├── mvc_encodings.py        # Three problem encodings (binary, set, edge)
│   ├── mvc_fitness.py          # Original fitness functions (hard penalties)
│   ├── mvc_fitness_opt.py      # Optimized fitness functions (smooth landscape)
│   ├── ga.py                   # Genetic Algorithm
│   ├── sa.py                   # Simulated Annealing
│   ├── ts.py                   # Tabu Search
│   └── experiments.py          # Experiment orchestration
├── report/
│   ├── main.tex                # Technical report (LaTeX)
│   └── appendix.tex            # Algorithm pseudocode
├── results/                    # Experimental results (CSV files)
├── instances/                  # Benchmark graph instances
├── run_experiments.py          # Main experiment runner
└── requirements.txt            # Python dependencies
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Experiments
```bash
# Default: 3 runs per configuration, results saved to results/results.csv
python run_experiments.py

# Custom output path
python run_experiments.py --output results/my_experiment.csv

# Quick test to verify setup
python test_quick.py
```

## Key Results

**Baseline Experiments** (324 runs total: 3 algorithms × 3 encodings × 3 fitness × 12 instances × 3 runs):

| Algorithm | Valid Solutions | Avg Cover Size | Runtime |
|-----------|-----------------|----------------|---------|
| **GA**    | 48.1%           | 34.78 ± 24.78  | 2.66s   |
| **SA**    | 55.6%           | 35.58 ± 25.70  | 0.14s   |
| **TS**    | 34.3%           | 34.14 ± 24.13  | 2.25s   |

**Optimized Fitness Functions** (same experimental design with smooth-landscape functions):

| Algorithm | Valid Solutions | Avg Cover Size | Runtime |
|-----------|-----------------|----------------|---------|
| **GA**    | 33.6%           | 32.65 ± 23.07  | 2.60s   |
| **SA**    | 37.0%           | 32.65 ± 23.14  | 0.14s   |
| **TS**    | 43.5%           | 32.28 ± 22.64  | 2.18s   |

**Key Findings**:
- **Set encoding + edge fitness**: Best overall performance across algorithms
- **Optimized fitness functions**: Eliminate death-penalty cliffs, improve TS validity by 9.2%
- **SA**: Fastest algorithm (100ms), excellent quality when feasible
- **TS**: Benefits most from smooth fitness landscapes
- **Instance size**: TS outperforms on large graphs (>50 nodes)

### Visual Results

<p align="center">
  <img src="report/stratified_analysis/01_stratified_quality_by_instance.png" alt="Stratified Quality by Instance" width="700"/>
</p>

*Solution quality stratified by instance size (small, medium, large). Clear progression showing algorithm behavior across different graph scales.*

<p align="center">
  <img src="report/stratified_analysis/07_cover_size_heatmap_per_instance.png" alt="Cover Size Heatmap" width="700"/>
</p>

*Heatmap of average cover sizes across algorithm-encoding-fitness combinations for each instance. Darker colors indicate smaller (better) vertex covers.*

<p align="center">
  <img src="report/stratified_analysis/09_fitness_function_comparison.png" alt="Fitness Function Comparison" width="700"/>
</p>

*Comprehensive comparison of fitness functions showing both solution quality and validity rates. Optimized functions provide smoother search landscapes.*


## Reproducibility

### System Requirements
- **Python**: 3.8 or higher
- **OS**: Linux, macOS, or Windows (WSL recommended)
- **RAM**: 2GB minimum
- **Storage**: 500MB for results and plots

### Dependencies
```bash
numpy>=1.20.0
networkx>=2.6
matplotlib>=3.4.0
seaborn>=0.11.0
pandas>=1.3.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Reproducing Baseline Results
```bash
# Run baseline experiments (3 algorithms × 3 encodings × 3 fitness × 12 instances × 3 runs)
python run_experiments.py --output results/baseline_results.csv

# Expected runtime: ~45 minutes
# Expected output: CSV with 324 rows
```

### Reproducing Optimized Fitness Results
```bash
# First, ensure optimized fitness module exists
ls src/mvc_fitness_opt.py

# Run experiments with optimized fitness functions
# (Modify run_experiments.py to import from mvc_fitness_opt instead of mvc_fitness)
python run_experiments_opt.py --output results/optimized_results.csv

# Expected runtime: ~45 minutes
# Expected output: CSV with 324 rows
```

### Reproducing GA Parameter Analysis
```bash
# Analyze GA sensitivity to generations and mutation rates
python ga_parameter_analysis.py

# Expected runtime: ~30 minutes
# Expected output: Results in ga_analysis_results_opt/
# - validity_progression.png: Validity vs generations (0% → 100%)
# - mutation_rate_impact.png: Mutation rate sensitivity analysis
```

### Reproducing Report Figures
```bash
# Generate all stratified analysis plots
python generate_stratified_plots.py

# Expected output: Plots in test_stratified/ including:
# - plot_1.png: Feasibility by fitness function
# - plot_6.png: Quality by instance size
# - plot_7.png: Runtime by algorithm
# - plot_11.png: Convergence by instance difficulty

# Generate comprehensive 2-panel GA analysis
cd report
python plot_ga_analysis.py

# Expected output: ga_analysis_combined.png (2 panels showing validity progression)
```

### Verification
After running experiments, verify results:
```bash
# Check CSV structure
head -n 5 results/results.csv

# Expected columns: algorithm,encoding,fitness,instance,run,best_cover_size,
#                   is_valid,best_fitness,total_time,evaluations

# Verify instance generation is deterministic
python -c "
import sys; sys.path.insert(0, 'src')
from problem import InstanceGenerator
instances = InstanceGenerator.generate_benchmark_instances()
print(f'Generated {len(instances)} instances')
print(f'First instance: {instances[0][1]} with {instances[0][0].n_nodes} nodes')
"
# Expected: 12 instances, deterministic output
```

### Random Seed Control
The codebase uses fixed random seeds for reproducibility:
- **Instance generation**: `seed=42` in `InstanceGenerator`
- **Algorithm runs**: `seed=42 + run_id` per experiment
- **To change seeds**: Modify `experiments.py` line 15

### Common Issues

**Issue**: Import error for `mvc_fitness_opt`
**Solution**: Ensure `src/mvc_fitness_opt.py` exists. If not, use baseline fitness from `mvc_fitness.py`.

**Issue**: Memory error during large experiments
**Solution**: Reduce `num_runs` in `run_experiments.py` from 3 to 1.

**Issue**: Plots look different from report
**Solution**: Ensure matplotlib version ≥3.4.0 and seaborn ≥0.11.0 for consistent styling.

### Dataset Access
Benchmark instances are generated programmatically with deterministic seeds:
- **Small**: 10-20 nodes, 15-50 edges (Erdős-Rényi, p=0.3)
- **Medium**: 30-50 nodes, 90-300 edges (Erdős-Rényi, p=0.2)  
- **Large**: 60-80 nodes, 360-1000 edges (Erdős-Rényi, p=0.15)

Export instances to disk:
```bash
python export_instances.py
# Output: instances/ directory with edge lists and metadata JSON
```

### Related Projects
- **MVC-Metaheuristics**: [github.com/yazid-hoblos/MVC-Metaheuristics](https://github.com/yazid-hoblos/MVC-Metaheuristics) - This repository
- **ENGA Framework**: [github.com/yazid-hoblos/ENGA](https://github.com/yazid-hoblos/ENGA) - Evolving Network Generation with Augmentation

### Support
- **Issues**: [MVC-Metaheuristics Issues](https://github.com/yazid-hoblos/MVC-Metaheuristics/issues)
- **Detailed Methodology**: See `report/main.tex` for technical report



## References

Complete bibliography available in [report/main.tex](report/main.tex). Key references:

- **Karp (1972)**: Reducibility among combinatorial problems - NP-completeness of Vertex Cover
- **Kirkpatrick et al. (1983)**: Optimization by simulated annealing - SA algorithm foundation
- **Holland (1975)**: Adaptation in natural and artificial systems - GA theoretical basis  
- **Glover (1989, 1990)**: Tabu search fundamentals and applications
- **Karakostas (2005)**: Better approximation algorithm for vertex cover  
- **Hoblos (2025)**: Evolving Network Generation with Augmentation (ENGA) preprint

## License

This project is part of academic research. For reuse, please cite the associated technical report.

---

**Author**: Yazid Hoblos  
**Institution**: M2 Research Project  
**Year**: 2025
