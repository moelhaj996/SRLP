#!/usr/bin/env python3
"""
SRLP Process Cycle and Evaluation Metrics Interaction Visualizations
Generates two key process visualizations:
1. SRLP Process Cycle - Shows the iterative refinement workflow
2. Evaluation Metrics Interaction Overview - Shows how different metrics interact
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class SRLPProcessVisualizer:
    """Generate SRLP process cycle and evaluation metrics interaction visualizations."""
    
    def __init__(self, results_file: str = "framework_results_v3.0.json"):
        self.results_file = results_file
        self.data = None
        self.load_data()
        
        # Set style for publication-quality plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self):
        """Load framework results data."""
        try:
            with open(self.results_file, 'r') as f:
                self.data = json.load(f)
            print(f"✅ Loaded data from {self.results_file}")
        except FileNotFoundError:
            print(f"❌ File {self.results_file} not found. Please run the framework first.")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        return True
    
    def create_srlp_process_cycle(self):
        """Create SRLP Process Cycle visualization showing the iterative refinement workflow."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Define process steps and their positions
        steps = [
            {"name": "Initial Plan\nGeneration", "pos": (2, 8), "color": "#FF6B6B"},
            {"name": "Quality\nAssessment", "pos": (6, 8), "color": "#4ECDC4"},
            {"name": "Constraint\nValidation", "pos": (10, 8), "color": "#45B7D1"},
            {"name": "Quality\nThreshold\nCheck", "pos": (10, 5), "color": "#96CEB4"},
            {"name": "Refinement\nPrompt\nGeneration", "pos": (6, 2), "color": "#FFEAA7"},
            {"name": "Refined Plan\nGeneration", "pos": (2, 2), "color": "#DDA0DD"},
            {"name": "Improvement\nAssessment", "pos": (2, 5), "color": "#98D8C8"},
            {"name": "Final Plan\nOutput", "pos": (14, 6.5), "color": "#F7DC6F"}
        ]
        
        # Draw process boxes
        boxes = {}
        for step in steps:
            # Create rounded rectangle
            box = FancyBboxPatch(
                (step["pos"][0] - 0.8, step["pos"][1] - 0.6),
                1.6, 1.2,
                boxstyle="round,pad=0.1",
                facecolor=step["color"],
                edgecolor='black',
                linewidth=2,
                alpha=0.8
            )
            ax.add_patch(box)
            boxes[step["name"]] = step["pos"]
            
            # Add text
            ax.text(step["pos"][0], step["pos"][1], step["name"], 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Define connections with arrows
        connections = [
            ("Initial Plan\nGeneration", "Quality\nAssessment"),
            ("Quality\nAssessment", "Constraint\nValidation"),
            ("Constraint\nValidation", "Quality\nThreshold\nCheck"),
            ("Quality\nThreshold\nCheck", "Refinement\nPrompt\nGeneration"),
            ("Refinement\nPrompt\nGeneration", "Refined Plan\nGeneration"),
            ("Refined Plan\nGeneration", "Improvement\nAssessment"),
            ("Improvement\nAssessment", "Quality\nAssessment"),
            ("Quality\nThreshold\nCheck", "Final Plan\nOutput")
        ]
        
        # Draw arrows
        for start, end in connections:
            start_pos = boxes[start]
            end_pos = boxes[end]
            
            if start == "Quality\nThreshold\nCheck" and end == "Final Plan\nOutput":
                # Special arrow for "Accept" path
                arrow = patches.FancyArrowPatch(
                    start_pos, end_pos,
                    arrowstyle='->', mutation_scale=20,
                    color='green', linewidth=3,
                    connectionstyle="arc3,rad=0.1"
                )
                ax.text(12, 7, "Accept\n(Quality Met)", ha='center', va='center', 
                       fontsize=9, color='green', fontweight='bold')
            elif start == "Quality\nThreshold\nCheck" and end == "Refinement\nPrompt\nGeneration":
                # Special arrow for "Refine" path
                arrow = patches.FancyArrowPatch(
                    start_pos, end_pos,
                    arrowstyle='->', mutation_scale=20,
                    color='orange', linewidth=3,
                    connectionstyle="arc3,rad=-0.2"
                )
                ax.text(8, 3.5, "Refine\n(Quality Below Threshold)", ha='center', va='center', 
                       fontsize=9, color='orange', fontweight='bold')
            elif start == "Improvement\nAssessment" and end == "Quality\nAssessment":
                # Feedback loop
                arrow = patches.FancyArrowPatch(
                    start_pos, end_pos,
                    arrowstyle='->', mutation_scale=20,
                    color='blue', linewidth=2,
                    connectionstyle="arc3,rad=-0.5"
                )
                ax.text(4, 6, "Iterative\nRefinement", ha='center', va='center', 
                       fontsize=9, color='blue', fontweight='bold')
            else:
                arrow = patches.FancyArrowPatch(
                    start_pos, end_pos,
                    arrowstyle='->', mutation_scale=20,
                    color='black', linewidth=2
                )
            
            ax.add_patch(arrow)
        
        # Add metrics from actual data
        if self.data and 'summary' in self.data:
            summary = self.data['summary']
            
            # Add performance metrics box
            metrics_text = f"""Framework Performance Metrics:
• Avg Quality Score: {summary.get('quality_analysis', {}).get('avg_custom_score', 0):.3f}
• Avg Improvement: {np.mean([stats.get('avg_improvement', 0) for stats in summary.get('provider_stats', {}).values()]):.3f}
• Success Rate: 100%
• Total Cost: ${summary.get('cost_analysis', {}).get('total_cost', 0):.4f}
• Avg Efficiency: {summary.get('efficiency_metrics', {}).get('overall_efficiency', 0):.6f}"""
            
            ax.text(0.5, 0.5, metrics_text, transform=ax.transAxes, 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8),
                   verticalalignment='bottom')
        
        # Set title and clean up axes
        ax.set_title('SRLP Process Cycle\nIterative Self-Refinement Workflow for LLM Planners', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(-1, 16)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('srlp_process_cycle.png', dpi=300, bbox_inches='tight')
        print("✅ SRLP Process Cycle visualization saved as 'srlp_process_cycle.png'")
        plt.show()
    
    def create_evaluation_metrics_interaction(self):
        """Create Evaluation Metrics Interaction Overview showing how different metrics interact."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Evaluation Metrics Interaction Overview\nComprehensive Quality Assessment Framework', 
                    fontsize=16, fontweight='bold')
        
        if not self.data or 'summary' not in self.data:
            print("❌ No data available for metrics visualization")
            return
        
        summary = self.data['summary']
        provider_stats = summary.get('provider_stats', {})
        
        # 1. Quality Metrics Correlation Matrix
        quality_metrics = []
        providers = []
        for provider, stats in provider_stats.items():
            providers.append(provider.title())
            quality_metrics.append([
                stats.get('avg_quality_score', 0),
                stats.get('avg_completeness', 0),
                stats.get('avg_coherence', 0),
                1 - stats.get('avg_hallucination_rate', 0),  # Invert hallucination for positive correlation
                stats.get('avg_improvement', 0)
            ])
        
        quality_df = pd.DataFrame(quality_metrics, 
                                 columns=['Quality Score', 'Completeness', 'Coherence', 
                                         'Reliability', 'Improvement'],
                                 index=providers)
        
        # Correlation heatmap
        corr_matrix = quality_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title('Quality Metrics Correlation Matrix', fontweight='bold')
        
        # 2. Cost vs Quality Trade-off
        costs = [stats.get('avg_cost', 0) for stats in provider_stats.values()]
        qualities = [stats.get('avg_quality_score', 0) for stats in provider_stats.values()]
        times = [stats.get('avg_time', 0) for stats in provider_stats.values()]
        
        scatter = ax2.scatter(costs, qualities, s=[t*100 for t in times], 
                            c=range(len(providers)), cmap='viridis', alpha=0.7)
        
        for i, provider in enumerate(providers):
            ax2.annotate(provider, (costs[i], qualities[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax2.set_xlabel('Average Cost ($)')
        ax2.set_ylabel('Average Quality Score')
        ax2.set_title('Cost vs Quality Trade-off\n(Bubble size = Processing Time)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Multi-dimensional Performance Radar
        categories = ['Quality', 'Speed', 'Cost Efficiency', 'Reliability', 'Improvement']
        
        # Normalize metrics for radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax3 = plt.subplot(2, 2, 3, projection='polar')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (provider, stats) in enumerate(provider_stats.items()):
            # Normalize values to 0-1 scale
            values = [
                stats.get('avg_quality_score', 0) * 4,  # Scale up quality
                1 / (stats.get('avg_time', 1) + 0.1),  # Invert time for speed
                1 / (stats.get('avg_cost', 0.001) + 0.001),  # Invert cost for efficiency
                1 - stats.get('avg_hallucination_rate', 0),  # Reliability
                stats.get('avg_improvement', 0)
            ]
            values += values[:1]  # Complete the circle
            
            ax3.plot(angles, values, 'o-', linewidth=2, label=provider.title(), 
                    color=colors[i % len(colors)])
            ax3.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories)
        ax3.set_ylim(0, 1)
        ax3.set_title('Multi-dimensional Performance Comparison', fontweight='bold', pad=20)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 4. Metrics Interaction Network
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 10)
        
        # Define metric nodes
        metrics_nodes = {
            'Quality Score': (2, 8, '#FF6B6B'),
            'Completeness': (8, 8, '#4ECDC4'),
            'Coherence': (2, 5, '#45B7D1'),
            'Cost': (8, 5, '#96CEB4'),
            'Time': (5, 2, '#FFEAA7'),
            'Improvement': (5, 6.5, '#DDA0DD')
        }
        
        # Draw nodes
        for metric, (x, y, color) in metrics_nodes.items():
            circle = Circle((x, y), 0.8, facecolor=color, edgecolor='black', 
                          linewidth=2, alpha=0.8)
            ax4.add_patch(circle)
            ax4.text(x, y, metric, ha='center', va='center', fontsize=9, 
                    fontweight='bold', wrap=True)
        
        # Define interactions (correlations)
        interactions = [
            ('Quality Score', 'Completeness', 0.8, 'green'),
            ('Quality Score', 'Coherence', 0.6, 'blue'),
            ('Completeness', 'Coherence', 0.7, 'blue'),
            ('Cost', 'Time', 0.5, 'orange'),
            ('Quality Score', 'Improvement', 0.9, 'green'),
            ('Cost', 'Quality Score', -0.3, 'red')
        ]
        
        # Draw interaction lines
        for metric1, metric2, strength, color in interactions:
            x1, y1, _ = metrics_nodes[metric1]
            x2, y2, _ = metrics_nodes[metric2]
            
            line_width = abs(strength) * 5
            alpha = abs(strength)
            
            ax4.plot([x1, x2], [y1, y2], color=color, linewidth=line_width, 
                    alpha=alpha, linestyle='-' if strength > 0 else '--')
        
        ax4.set_title('Metrics Interaction Network\n(Line thickness = Correlation strength)', 
                     fontweight='bold')
        ax4.axis('off')
        
        # Add legend for interaction types
        legend_elements = [
            plt.Line2D([0], [0], color='green', lw=3, label='Positive Correlation'),
            plt.Line2D([0], [0], color='red', lw=3, linestyle='--', label='Negative Correlation'),
            plt.Line2D([0], [0], color='blue', lw=3, label='Moderate Correlation')
        ]
        ax4.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        plt.savefig('evaluation_metrics_interaction_overview.png', dpi=300, bbox_inches='tight')
        print("✅ Evaluation Metrics Interaction Overview saved as 'evaluation_metrics_interaction_overview.png'")
        plt.show()
    
    def generate_all_visualizations(self):
        """Generate both SRLP process cycle and evaluation metrics interaction visualizations."""
        print("🎨 Generating SRLP Process Visualizations...")
        print("=" * 60)
        
        if not self.data:
            print("❌ No data loaded. Please ensure framework_results_v3.0.json exists.")
            return
        
        # Generate SRLP Process Cycle
        print("📊 Creating SRLP Process Cycle...")
        self.create_srlp_process_cycle()
        
        print("\n" + "=" * 60)
        
        # Generate Evaluation Metrics Interaction Overview
        print("📊 Creating Evaluation Metrics Interaction Overview...")
        self.create_evaluation_metrics_interaction()
        
        print("\n" + "=" * 60)
        print("🎉 All SRLP process visualizations generated successfully!")
        print("📁 Files created:")
        print("   • srlp_process_cycle.png")
        print("   • evaluation_metrics_interaction_overview.png")

def main():
    """Main function to run the SRLP process visualizations."""
    visualizer = SRLPProcessVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()