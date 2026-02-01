"""
Run ENGA (Enhanced Networked Genetic Algorithm) on the MVC benchmark instances
and export results in the same CSV schema as the existing experiments.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import networkx as nx

# Use a non-interactive backend to avoid GUI issues during ENGA draws
import matplotlib

matplotlib.use("Agg")


def _append_sys_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.append(path)


def _load_graph(edgelist_path: Path) -> nx.Graph:
    return nx.read_edgelist(edgelist_path, nodetype=int)


def _create_random_individual(n_genes: int, domain, decimal_precision: int):
    return [random.randint(0, 1) for _ in range(n_genes)]


def _coerce_binary(individual: List[float]) -> List[int]:
    return [1 if val >= 0.5 else 0 for val in individual]


def _get_algorithms() -> dict:
    from enhanced_networked_genetic_algorithm import EnhancedNetworkGeneticAlgorithm
    from networked_genetic_algorithm import NetworkGeneticAlgorithm
    from genetic_algorithm import GeneticAlgorithm

    return {
        "enhanced": ("ENGA", EnhancedNetworkGeneticAlgorithm),
        "networked": ("NGA", NetworkGeneticAlgorithm),
        "ga": ("GA", GeneticAlgorithm),
    }


def _disable_enga_drawing() -> None:
    from utils.drawable import DrawManager
    from genetic_algorithm import GAHistory

    DrawManager.draw_network = lambda *args, **kwargs: None
    DrawManager.draw_animated_network = lambda *args, **kwargs: None

    def _safe_str(self: GAHistory) -> str:
        if not self._best_fitnesses:
            return (
                "\n\n---------------------------------------\n\n"
                "Number of generations: 0\n"
                "Total time: 0.00 seconds\n"
                "Average fitness: n/a\n"
                "Best fitness: n/a\n"
                "Best individual: n/a\n"
                "\n\n---------------------------------------\n\n"
            )

        avg_fitnesses = float(np.mean(self._best_fitnesses))
        best_fitness = float(np.max(self._best_fitnesses) if self._is_maximization else np.min(self._best_fitnesses))
        best_index = int(np.argmax(self._fitnesses[-1]) if self._is_maximization else np.argmin(self._fitnesses[-1]))
        best_individual = self._populations[-1][best_index]

        return (
            "\n\n---------------------------------------\n\n"
            f"Number of generations: {self._number_of_generations}\n"
            f"Total time: {self.get_total_time()}\n"
            f"Average fitness: {avg_fitnesses}\n"
            f"Best fitness: {best_fitness}\n"
            f"Best individual: {best_individual}"
            "\n\n---------------------------------------\n\n"
        )

    GAHistory.__str__ = _safe_str


def _get_fitness_functions(problem):
    from src.mvc_fitness import FitnessFunctionFactory

    return [
        FitnessFunctionFactory.create_cover_size_minimization(problem),
        FitnessFunctionFactory.create_constraint_penalty(problem, penalty_weight=1.0),
        FitnessFunctionFactory.create_edge_coverage_optimization(problem, size_weight=0.3),
    ]


def _run_single(
    ga_class,
    problem,
    fitness_function,
    runs: int,
    generations: int,
    population_size: int,
    elites: int,
    crossover_prob: float,
    mutation_prob: float,
    seed_base: int,
) -> List[dict]:
    from src.mvc_encodings import BinaryEncoding

    encoding = BinaryEncoding()

    def fitness(individual):
        cover = encoding.solution_to_cover(_coerce_binary(individual))
        return fitness_function.evaluate(cover)

    rows = []
    for run_idx in range(1, runs + 1):
        seed = seed_base + run_idx
        random.seed(seed)
        np.random.seed(seed)

        ga = ga_class(
            number_of_genes=problem.num_nodes,
            domain=[0, 1],
            number_of_generations=generations,
            population_size=population_size,
            number_elites=elites,
            probability_crossover=crossover_prob,
            probability_mutation=mutation_prob,
            decimal_precision=0,
            create_random_individual=_create_random_individual,
            fitness_function=fitness,
            is_maximization=True,
            verbose=False,
            random_seed=seed,
        )

        history = ga.run()
        populations = history.get_populations()
        fitnesses = history.get_fitnesses()

        if populations and fitnesses:
            last_pop = populations[-1]
            last_fit = fitnesses[-1]
            best_idx = int(np.argmax(last_fit))
            best_individual = last_pop[best_idx]
            best_fitness = float(last_fit[best_idx])
        else:
            best_individual = [0] * problem.num_nodes
            best_fitness = float("-inf")

        best_cover = encoding.solution_to_cover(_coerce_binary(best_individual))
        is_valid = problem.is_valid_cover(best_cover)
        cover_size = len(best_cover)

        rows.append(
            {
                "run": run_idx,
                "cover_size": cover_size,
                "is_valid": is_valid,
                "fitness": best_fitness,
                "time_sec": float(getattr(history, "_total_time", 0.0)),
            }
        )

    return rows


def _write_results(output_path: Path, rows: List[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instance",
        "algorithm",
        "encoding",
        "fitness_func",
        "run",
        "cover_size",
        "is_valid",
        "fitness",
        "time_sec",
    ]

    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _compare_to_baseline(baseline_path: Path, output_path: Path) -> None:
    if not baseline_path.exists() or not output_path.exists():
        return

    def _load(path: Path) -> List[dict]:
        with path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    baseline = _load(baseline_path)
    enga = _load(output_path)

    def _summarize(rows: List[dict], algorithm: str, encoding: str = "BinaryEncoding") -> Tuple[float, float]:
        filtered = [r for r in rows if r["algorithm"] == algorithm and r["encoding"] == encoding]
        if not filtered:
            return 0.0, 0.0
        valid_rows = [r for r in filtered if r["is_valid"].lower() in {"true", "1"}]
        valid_rate = 100.0 * len(valid_rows) / len(filtered)
        if valid_rows:
            avg_cover = sum(float(r["cover_size"]) for r in valid_rows) / len(valid_rows)
        else:
            avg_cover = 0.0
        return valid_rate, avg_cover

    enga_algos = sorted({r["algorithm"] for r in enga})
    for alg in enga_algos:
        enga_valid, enga_cover = _summarize(enga, alg)
        ga_valid, ga_cover = _summarize(baseline, "GA")
        print(
            f"[Comparison] {alg} vs GA (BinaryEncoding): "
            f"validity {enga_valid:.1f}% vs {ga_valid:.1f}%, "
            f"avg cover {enga_cover:.2f} vs {ga_cover:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ENGA on MVC benchmark instances")
    parser.add_argument(
        "--enga-path",
        default="/mnt/f/archives/projects_archive/ENGA/src",
        help="Path to ENGA src directory",
    )
    parser.add_argument(
        "--instances-dir",
        default=str(Path(__file__).resolve().parent / "instances"),
        help="Directory containing benchmark instances (.edgelist)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "results" / "enga_results.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--elites", type=int, default=5)
    parser.add_argument("--crossover", type=float, default=0.8)
    parser.add_argument("--mutation", type=float, default=0.1)
    parser.add_argument(
        "--algorithm",
        choices=["enhanced", "networked", "ga"],
        default="enhanced",
        help="ENGA variant to run",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare against baseline results.csv (GA, BinaryEncoding)",
    )
    parser.add_argument(
        "--baseline",
        default=str(Path(__file__).resolve().parent / "results" / "results.csv"),
        help="Baseline results CSV for comparison",
    )

    args = parser.parse_args()

    root_path = Path(__file__).resolve().parent
    _append_sys_path(args.enga_path)
    _append_sys_path(str(root_path))
    _append_sys_path(str(root_path / "src"))
    _disable_enga_drawing()

    algorithms = _get_algorithms()
    algo_label, algo_class = algorithms[args.algorithm]

    from src.problem import MinimumVertexCoverProblem

    instances_dir = Path(args.instances_dir)
    instance_files = [
        "small_20nodes.edgelist",
        "medium_50nodes.edgelist",
        "large_100nodes.edgelist",
        "scale_free_50nodes.edgelist",
    ]

    output_path = Path(args.output)

    for instance_file in instance_files:
        name = instance_file.replace(".edgelist", "")
        graph = _load_graph(instances_dir / instance_file)
        problem = MinimumVertexCoverProblem(graph)
        fitness_functions = _get_fitness_functions(problem)

        for fitness_function in fitness_functions:
            rows = _run_single(
                algo_class,
                problem,
                fitness_function,
                runs=args.runs,
                generations=args.generations,
                population_size=args.population_size,
                elites=args.elites,
                crossover_prob=args.crossover,
                mutation_prob=args.mutation,
                seed_base=1000,
            )

            enriched_rows = []
            for row in rows:
                enriched_rows.append(
                    {
                        "instance": name,
                        "algorithm": algo_label,
                        "encoding": "BinaryEncoding",
                        "fitness_func": fitness_function.get_name(),
                        **row,
                    }
                )

            _write_results(output_path, enriched_rows)

    if args.compare:
        _compare_to_baseline(Path(args.baseline), output_path)


if __name__ == "__main__":
    main()