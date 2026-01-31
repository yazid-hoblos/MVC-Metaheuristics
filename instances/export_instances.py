"""
Export benchmark instances to instances/ as edge lists and metadata.
"""

import json
from pathlib import Path
import networkx as nx

from src.problem import InstanceGenerator


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "instances"


def export_instances():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    instances = InstanceGenerator.generate_benchmark_instances()

    metadata = []

    for problem, name in instances:
        graph = problem.graph
        edge_path = OUT_DIR / f"{name}.edgelist"
        meta_path = OUT_DIR / f"{name}.json"

        nx.write_edgelist(graph, edge_path, data=False)

        info = {
            "name": name,
            "num_nodes": problem.num_nodes,
            "num_edges": problem.num_edges,
        }
        metadata.append(info)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    summary_path = OUT_DIR / "instances_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported {len(metadata)} instances to: {OUT_DIR}")


if __name__ == "__main__":
    export_instances()
