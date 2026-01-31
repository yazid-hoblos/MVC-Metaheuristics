"""
Animate GA solution convergence on graph visualization.
Shows how the cover evolves over generations.
"""

import sys
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch

# Adjust path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from problem import InstanceGenerator
from mvc_encodings import BinaryEncoding
from mvc_fitness import ConstraintPenalty
from ga import GeneticAlgorithm, GAParams


def create_animation(instance_name: str = "small_20nodes", 
                     output_file: str = None,
                     fps: int = 5,
                     generations: int = 50):
    """
    Create animation of GA solution evolution.
    
    Args:
        instance_name: Which instance to animate
        output_file: Output file path (GIF recommended)
        fps: Frames per second
        generations: Number of GA generations (lower = faster)
    """
    print(f"Running GA for {generations} generations on {instance_name}...")

    # Generate instance
    instances = {name: problem for problem, name in InstanceGenerator.generate_benchmark_instances()}
    problem = instances[instance_name]
    
    # Setup GA
    encoding = BinaryEncoding()
    fitness = ConstraintPenalty(problem, penalty_weight=1.0)
    params = GAParams(population_size=50, generations=generations)
    
    ga = GeneticAlgorithm(problem, encoding, fitness, params)
    result = ga.run()
    
    # Create graph layout
    G = nx.Graph()
    G.add_nodes_from(range(problem.num_nodes))
    G.add_edges_from(problem.edges)
    pos = nx.spring_layout(G, seed=42, iterations=50)
    
    # Setup figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Graph visualization
    ax_graph = axes[0]
    ax_graph.set_title("GA Solution Evolution: Vertex Cover", fontsize=12, fontweight='bold')
    
    # Right: Convergence curve
    ax_conv = axes[1]
    ax_conv2 = ax_conv.twinx()
    ax_conv.set_xlabel("Generation")
    ax_conv.set_ylabel("Best Cover Size")
    ax_conv.set_title("Convergence: Best Cover Size Over Generations", fontsize=12, fontweight='bold')
    ax_conv.grid(True, alpha=0.3)
    
    legend_box = fig.text(0.7, 0.92, "", ha='center', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Extract generation data
    generations_list = [s['generation'] for s in result['generation_stats']]
    cover_sizes = [s['best_cover_size'] for s in result['generation_stats']]
    best_fitness = [s['best_fitness'] for s in result['generation_stats']]
    
    # Use per-generation covers from GA run to ensure plot consistency
    generation_covers = [s.get('best_cover_so_far') or s.get('gen_best_cover')
                         for s in result['generation_stats']]
    
    def update_frame(frame):
        """Update animation frame."""
        ax_graph.clear()
        ax_conv.clear()
        ax_conv2.clear()
        
        # Get cover for this generation
        if frame < len(generation_covers):
            cover = generation_covers[frame]
            cover_size = len(cover)
        else:
            cover = result['best_cover']
            cover_size = len(cover)
        
        # Draw graph
        node_colors = []
        for node in G.nodes():
            if node in cover:
                node_colors.append('#FF6B6B')  # Red: in cover
            else:
                node_colors.append('#4ECDC4')  # Teal: not in cover
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=500, ax=ax_graph)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax_graph)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax_graph)
        
        ax_graph.set_title(
            f"GA Solution Evolution (Gen {frame}/{len(result['generation_stats'])-1})\n"
            f"Cover Size: {cover_size} | Valid: {problem.is_valid_cover(cover)}",
            fontsize=11, fontweight='bold'
        )
        ax_graph.axis('off')
        
        # Draw convergence curve
        end_gen = min(frame + 1, len(generations_list))
        line1 = ax_conv.plot(generations_list[:end_gen], cover_sizes[:end_gen], 
                     'b-o', linewidth=2, markersize=4, label='Best Cover Size')
        
        # Add best fitness as secondary y-axis
        line2 = ax_conv2.plot(generations_list[:end_gen], best_fitness[:end_gen], 
                      'g--s', linewidth=2, markersize=4, alpha=0.7, label='Best Fitness')
        ax_conv2.set_ylabel("Best Fitness", color='g')
        ax_conv2.tick_params(axis='y', labelcolor='g')
        
        ax_conv.set_xlabel("Generation")
        ax_conv.set_ylabel("Best Cover Size", color='b')
        ax_conv.tick_params(axis='y', labelcolor='b')
        ax_conv.set_xlim(-2, len(generations_list) + 2)
        ax_conv.set_ylim(min(cover_sizes) - 2, max(cover_sizes) + 2)
        ax_conv.grid(True, alpha=0.3)
        handles = line1 + line2
        labels = [h.get_label() for h in handles]
        ax_conv.legend(handles, labels, loc='upper right',
                   bbox_to_anchor=(1.02, 1.12), fontsize=9, frameon=True)
        
        # Legend
        legend_text = (
            f"Generation: {frame}/{len(result['generation_stats'])-1}\n"
            f"Nodes: {problem.num_nodes}, Edges: {problem.num_edges}\n"
            f"Best Size Found: {result['best_cover_size']}"
        )
        legend_box.set_text(legend_text)
    
    # Create animation
    num_frames = len(result['generation_stats'])
    anim = animation.FuncAnimation(fig, update_frame, frames=num_frames,
                                   interval=1000 // fps, repeat=True)
    
    # Save animation (GIF via Pillow, avoids ffmpeg dependency)
    if output_file is None:
        output_file = f"report/figures/ga_animation_{instance_name}.gif"
    
    output_path = Path(output_file)
    if output_path.suffix.lower() != ".gif":
        output_path = output_path.with_suffix('.gif')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        anim.save(str(output_path), writer='pillow', fps=fps, dpi=80)
        print(f"Animation saved as GIF: {output_path}")
    except Exception as e:
        print(f"Could not save GIF: {e}")
    
    plt.close(fig)


if __name__ == "__main__":
    print("Generating GA solution animation...")
    
    # Animate on small instance
    create_animation(instance_name="medium_50nodes",
                    output_file="report/figures/ga_animation_medium.gif",
                    fps=5,
                    generations=50)
    
    print("Done!")
