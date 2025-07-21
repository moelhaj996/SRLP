#!/usr/bin/env python3
"""
Comprehensive Analysis Visualizations for SRLP Framework
Generates four key analysis visualizations:
1. Provider comparison and ranking
2. Cost vs quality trade-off analysis
3. Scenario complexity evaluation
4. Retry and error pattern analysis
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveAnalysisVisualizer:
    """Generate comprehensive analysis visualizations for SRLP Framework."""
    
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
    
    def extract_provider_metrics(self) -> pd.DataFrame:
        """Extract provider-level metrics for analysis."""
        provider_data = []
        
        # Use summary data if available
        if 'summary' in self.data and 'provider_stats' in self.data['summary']:
            for provider, stats in self.data['summary']['provider_stats'].items():
                provider_data.append({
                    'Provider': provider.title(),
                    'Avg_Quality': stats.get('avg_quality_score', 0),
                    'Avg_Cost': stats.get('avg_cost', 0),
                    'Success_Rate': 100.0,  # Default success rate from summary
                    'Avg_Wall_Time': stats.get('avg_wall_time', 0),
                    'Total_Retries': stats.get('total_retries', 0),
                    'Avg_Improvement': stats.get('avg_improvement', 0)
                })
        else:
            # Fallback: extract from detailed results (list format)
            detailed_results = self.data.get('detailed_results', [])
            provider_stats = {}
            
            for result in detailed_results:
                provider = result.get('provider', 'unknown')
                if provider not in provider_stats:
                    provider_stats[provider] = {
                        'quality_scores': [],
                        'costs': [],
                        'wall_times': [],
                        'retries': [],
                        'improvements': [],
                        'successes': []
                    }
                
                provider_stats[provider]['quality_scores'].append(
                    result.get('quality_metrics', {}).get('custom_score', 0)
                )
                provider_stats[provider]['costs'].append(
                    result.get('cost_metrics', {}).get('total_cost', 0)
                )
                provider_stats[provider]['wall_times'].append(
                    result.get('wall_clock_time', 0)
                )
                provider_stats[provider]['retries'].append(
                    result.get('retry_attempts', 0)
                )
                provider_stats[provider]['improvements'].append(
                    result.get('improvement_score', 0)
                )
                provider_stats[provider]['successes'].append(
                    1 if result.get('success', True) else 0
                )
            
            # Calculate averages
            for provider, stats in provider_stats.items():
                provider_data.append({
                    'Provider': provider.title(),
                    'Avg_Quality': np.mean(stats['quality_scores']) if stats['quality_scores'] else 0,
                    'Avg_Cost': np.mean(stats['costs']) if stats['costs'] else 0,
                    'Success_Rate': np.mean(stats['successes']) * 100 if stats['successes'] else 0,
                    'Avg_Wall_Time': np.mean(stats['wall_times']) if stats['wall_times'] else 0,
                    'Total_Retries': sum(stats['retries']) if stats['retries'] else 0,
                    'Avg_Improvement': np.mean(stats['improvements']) if stats['improvements'] else 0
                })
        
        return pd.DataFrame(provider_data)
    
    def extract_scenario_metrics(self) -> pd.DataFrame:
        """Extract scenario-level metrics for complexity analysis."""
        scenario_data = []
        
        # Use summary data if available
        if 'summary' in self.data and 'scenario_stats' in self.data['summary']:
            for scenario, stats in self.data['summary']['scenario_stats'].items():
                scenario_data.append({
                    'Scenario': scenario.replace('_', ' ').title(),
                    'Avg_Quality': stats.get('avg_quality_score', 0),
                    'Avg_Cost': stats.get('avg_cost', 0),
                    'Avg_Wall_Time': stats.get('avg_time', 0),
                    'Total_Retries': 0,  # Not available in summary, use 0
                    'Complexity_Score': stats.get('complexity', 0.5) * stats.get('avg_time', 0)
                })
        else:
            # Fallback: extract from detailed results (list format)
            detailed_results = self.data.get('detailed_results', [])
            scenario_stats = {}
            
            for result in detailed_results:
                scenario = result.get('scenario', 'unknown')
                if scenario not in scenario_stats:
                    scenario_stats[scenario] = {
                        'quality_scores': [],
                        'costs': [],
                        'wall_times': [],
                        'retries': []
                    }
                
                scenario_stats[scenario]['quality_scores'].append(
                    result.get('quality_metrics', {}).get('custom_score', 0)
                )
                scenario_stats[scenario]['costs'].append(
                    result.get('cost_metrics', {}).get('total_cost', 0)
                )
                scenario_stats[scenario]['wall_times'].append(
                    result.get('wall_clock_time', 0)
                )
                scenario_stats[scenario]['retries'].append(
                    result.get('retry_attempts', 0)
                )
            
            # Calculate averages
            for scenario, stats in scenario_stats.items():
                avg_quality = np.mean(stats['quality_scores']) if stats['quality_scores'] else 0
                avg_cost = np.mean(stats['costs']) if stats['costs'] else 0
                avg_wall_time = np.mean(stats['wall_times']) if stats['wall_times'] else 0
                total_retries = sum(stats['retries']) if stats['retries'] else 0
                
                scenario_data.append({
                    'Scenario': scenario.replace('_', ' ').title(),
                    'Avg_Quality': avg_quality,
                    'Avg_Cost': avg_cost,
                    'Avg_Wall_Time': avg_wall_time,
                    'Total_Retries': total_retries,
                    'Complexity_Score': avg_wall_time + total_retries * 0.1
                })
        
        return pd.DataFrame(scenario_data)
    
    def generate_provider_comparison(self):
        """Generate provider comparison and ranking visualization."""
        df = self.extract_provider_metrics()
        
        if df.empty:
            print("❌ No provider data available")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Provider Comparison and Ranking Analysis', fontsize=16, fontweight='bold')
        
        # 1. Quality vs Success Rate
        scatter = ax1.scatter(df['Avg_Quality'], df['Success_Rate'], 
                            s=df['Avg_Wall_Time']*10, alpha=0.7, c=range(len(df)), cmap='viridis')
        for i, provider in enumerate(df['Provider']):
            ax1.annotate(provider, (df['Avg_Quality'].iloc[i], df['Success_Rate'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax1.set_xlabel('Average Quality Score')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Quality vs Success Rate\n(Bubble size = Wall Time)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Provider Ranking Bar Chart
        df_sorted = df.sort_values('Avg_Quality', ascending=True)
        bars = ax2.barh(df_sorted['Provider'], df_sorted['Avg_Quality'], 
                       color=plt.cm.viridis(np.linspace(0, 1, len(df_sorted))))
        ax2.set_xlabel('Average Quality Score')
        ax2.set_title('Provider Ranking by Quality')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        # 3. Performance Radar Chart
        categories = ['Quality', 'Success Rate', 'Speed', 'Efficiency']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax3 = plt.subplot(2, 2, 3, projection='polar')
        
        for i, provider in enumerate(df['Provider']):
            # Normalize metrics to 0-1 scale
            quality_norm = df['Avg_Quality'].iloc[i] / df['Avg_Quality'].max() if df['Avg_Quality'].max() > 0 else 0
            success_norm = df['Success_Rate'].iloc[i] / 100
            speed_norm = 1 - (df['Avg_Wall_Time'].iloc[i] / df['Avg_Wall_Time'].max()) if df['Avg_Wall_Time'].max() > 0 else 0
            efficiency_norm = 1 - (df['Total_Retries'].iloc[i] / df['Total_Retries'].max()) if df['Total_Retries'].max() > 0 else 0
            
            values = [quality_norm, success_norm, speed_norm, efficiency_norm]
            values += values[:1]  # Complete the circle
            
            ax3.plot(angles, values, 'o-', linewidth=2, label=provider)
            ax3.fill(angles, values, alpha=0.25)
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories)
        ax3.set_ylim(0, 1)
        ax3.set_title('Provider Performance Radar', y=1.08)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 4. Improvement vs Retries
        ax4.scatter(df['Total_Retries'], df['Avg_Improvement'], 
                   s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
        for i, provider in enumerate(df['Provider']):
            ax4.annotate(provider, (df['Total_Retries'].iloc[i], df['Avg_Improvement'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax4.set_xlabel('Total Retries')
        ax4.set_ylabel('Average Improvement')
        ax4.set_title('Improvement vs Retry Patterns')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('provider_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Provider comparison visualization saved as 'provider_comparison_analysis.png'")
    
    def generate_cost_quality_analysis(self):
        """Generate cost vs quality trade-off analysis."""
        df = self.extract_provider_metrics()
        
        if df.empty:
            print("❌ No provider data available")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cost vs Quality Trade-off Analysis', fontsize=16, fontweight='bold')
        
        # 1. Cost vs Quality Scatter
        scatter = ax1.scatter(df['Avg_Cost'], df['Avg_Quality'], 
                            s=df['Success_Rate']*3, alpha=0.7, c=range(len(df)), cmap='RdYlBu_r')
        for i, provider in enumerate(df['Provider']):
            ax1.annotate(provider, (df['Avg_Cost'].iloc[i], df['Avg_Quality'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax1.set_xlabel('Average Cost')
        ax1.set_ylabel('Average Quality Score')
        ax1.set_title('Cost vs Quality Trade-off\n(Bubble size = Success Rate)')
        ax1.grid(True, alpha=0.3)
        
        # Add efficiency frontier line
        if len(df) > 1:
            z = np.polyfit(df['Avg_Cost'], df['Avg_Quality'], 1)
            p = np.poly1d(z)
            ax1.plot(df['Avg_Cost'], p(df['Avg_Cost']), "r--", alpha=0.8, label='Trend Line')
            ax1.legend()
        
        # 2. Value Score (Quality/Cost ratio)
        df['Value_Score'] = df['Avg_Quality'] / (df['Avg_Cost'] + 0.001)  # Avoid division by zero
        df_sorted = df.sort_values('Value_Score', ascending=True)
        
        bars = ax2.barh(df_sorted['Provider'], df_sorted['Value_Score'], 
                       color=plt.cm.RdYlBu_r(np.linspace(0, 1, len(df_sorted))))
        ax2.set_xlabel('Value Score (Quality/Cost)')
        ax2.set_title('Provider Value Ranking')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center', fontsize=10)
        
        # 3. Cost Distribution
        ax3.hist(df['Avg_Cost'], bins=max(3, len(df)), alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(df['Avg_Cost'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["Avg_Cost"].mean():.4f}')
        ax3.set_xlabel('Average Cost')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Cost Distribution Across Providers')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Quality vs Time vs Cost (3D-like visualization)
        # Normalize time for color mapping
        time_norm = (df['Avg_Wall_Time'] - df['Avg_Wall_Time'].min()) / (df['Avg_Wall_Time'].max() - df['Avg_Wall_Time'].min() + 0.001)
        
        scatter = ax4.scatter(df['Avg_Quality'], df['Avg_Cost'], 
                            s=df['Avg_Wall_Time']*20, c=time_norm, cmap='plasma', alpha=0.7)
        
        for i, provider in enumerate(df['Provider']):
            ax4.annotate(provider, (df['Avg_Quality'].iloc[i], df['Avg_Cost'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax4.set_xlabel('Average Quality Score')
        ax4.set_ylabel('Average Cost')
        ax4.set_title('Quality vs Cost vs Time\n(Bubble size = Time, Color = Time intensity)')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Time Intensity')
        
        plt.tight_layout()
        plt.savefig('cost_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Cost vs quality analysis saved as 'cost_quality_analysis.png'")
    
    def generate_scenario_complexity_analysis(self):
        """Generate scenario complexity evaluation."""
        df = self.extract_scenario_metrics()
        
        if df.empty:
            print("❌ No scenario data available")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Scenario Complexity Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Complexity Score Ranking
        df_sorted = df.sort_values('Complexity_Score', ascending=True)
        bars = ax1.barh(df_sorted['Scenario'], df_sorted['Complexity_Score'], 
                       color=plt.cm.Reds(np.linspace(0.3, 1, len(df_sorted))))
        ax1.set_xlabel('Complexity Score')
        ax1.set_title('Scenario Complexity Ranking')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center', fontsize=10)
        
        # 2. Time vs Retries Scatter
        scatter = ax2.scatter(df['Avg_Wall_Time'], df['Total_Retries'], 
                            s=df['Avg_Quality']*200, alpha=0.7, c=range(len(df)), cmap='viridis')
        for i, scenario in enumerate(df['Scenario']):
            ax2.annotate(scenario, (df['Avg_Wall_Time'].iloc[i], df['Total_Retries'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9, rotation=15)
        ax2.set_xlabel('Average Wall Time (seconds)')
        ax2.set_ylabel('Total Retries')
        ax2.set_title('Time vs Retries by Scenario\n(Bubble size = Quality)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Quality vs Complexity
        ax3.scatter(df['Complexity_Score'], df['Avg_Quality'], 
                   s=100, alpha=0.7, c=range(len(df)), cmap='RdYlBu')
        for i, scenario in enumerate(df['Scenario']):
            ax3.annotate(scenario, (df['Complexity_Score'].iloc[i], df['Avg_Quality'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax3.set_xlabel('Complexity Score')
        ax3.set_ylabel('Average Quality')
        ax3.set_title('Quality vs Complexity Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df['Complexity_Score'], df['Avg_Quality'], 1)
            p = np.poly1d(z)
            ax3.plot(df['Complexity_Score'], p(df['Complexity_Score']), "r--", alpha=0.8)
        
        # 4. Scenario Performance Heatmap
        metrics_matrix = df[['Avg_Quality', 'Avg_Cost', 'Avg_Wall_Time', 'Total_Retries']].T
        metrics_matrix.columns = df['Scenario']
        
        # Normalize for better visualization
        metrics_normalized = (metrics_matrix - metrics_matrix.min(axis=1).values.reshape(-1, 1)) / \
                           (metrics_matrix.max(axis=1) - metrics_matrix.min(axis=1)).values.reshape(-1, 1)
        
        sns.heatmap(metrics_normalized, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   ax=ax4, cbar_kws={'label': 'Normalized Score'})
        ax4.set_title('Scenario Performance Heatmap\n(Normalized Metrics)')
        ax4.set_ylabel('Metrics')
        
        plt.tight_layout()
        plt.savefig('scenario_complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Scenario complexity analysis saved as 'scenario_complexity_analysis.png'")
    
    def generate_retry_error_analysis(self):
        """Generate retry and error pattern analysis."""
        # Extract retry data from detailed results
        retry_data = []
        
        detailed_results = self.data.get('detailed_results', [])
        
        for result in detailed_results:
            scenario = result.get('scenario', 'unknown')
            provider = result.get('provider', 'unknown')
            retry_count = result.get('retry_attempts', 0)
            wall_time = result.get('wall_clock_time', 0)
            quality = result.get('quality_metrics', {}).get('custom_score', 0)
            violations = result.get('quality_metrics', {}).get('constraint_violations', 0)
            
            retry_data.append({
                'Scenario': scenario.replace('_', ' ').title(),
                'Provider': provider.title(),
                'Retries': retry_count,
                'Wall_Time': wall_time,
                'Quality': quality,
                'Violations': violations,
                'Has_Retries': 1 if retry_count > 0 else 0
            })
        
        if not retry_data:
            print("❌ No retry data available")
            return
        
        df = pd.DataFrame(retry_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Retry and Error Pattern Analysis', fontsize=16, fontweight='bold')
        
        # 1. Retry Distribution by Provider
        retry_by_provider = df.groupby('Provider')['Retries'].agg(['sum', 'mean', 'count']).reset_index()
        
        bars = ax1.bar(retry_by_provider['Provider'], retry_by_provider['sum'], 
                      alpha=0.7, color=plt.cm.Reds(np.linspace(0.3, 1, len(retry_by_provider))))
        ax1.set_xlabel('Provider')
        ax1.set_ylabel('Total Retries')
        ax1.set_title('Total Retries by Provider')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # 2. Retry Rate by Scenario
        retry_by_scenario = df.groupby('Scenario').agg({
            'Has_Retries': 'mean',
            'Retries': 'mean'
        }).reset_index()
        retry_by_scenario['Retry_Rate'] = retry_by_scenario['Has_Retries'] * 100
        
        bars = ax2.barh(retry_by_scenario['Scenario'], retry_by_scenario['Retry_Rate'], 
                       color=plt.cm.Oranges(np.linspace(0.3, 1, len(retry_by_scenario))))
        ax2.set_xlabel('Retry Rate (%)')
        ax2.set_title('Retry Rate by Scenario')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center', fontsize=10)
        
        # 3. Retries vs Quality Relationship
        ax3.scatter(df['Retries'], df['Quality'], alpha=0.6, s=50)
        ax3.set_xlabel('Number of Retries')
        ax3.set_ylabel('Quality Score')
        ax3.set_title('Retries vs Quality Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df['Retries'], df['Quality'], 1)
            p = np.poly1d(z)
            ax3.plot(df['Retries'], p(df['Retries']), "r--", alpha=0.8, 
                    label=f'Trend: slope={z[0]:.3f}')
            ax3.legend()
        
        # 4. Error Pattern Heatmap (Violations by Provider and Scenario)
        violation_pivot = df.pivot_table(values='Violations', index='Provider', 
                                       columns='Scenario', aggfunc='mean', fill_value=0)
        
        if not violation_pivot.empty:
            sns.heatmap(violation_pivot, annot=True, fmt='.1f', cmap='Reds', 
                       ax=ax4, cbar_kws={'label': 'Avg Violations'})
            ax4.set_title('Average Constraint Violations\nby Provider and Scenario')
            ax4.set_xlabel('Scenario')
            ax4.set_ylabel('Provider')
        else:
            ax4.text(0.5, 0.5, 'No violation data available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Constraint Violations Analysis')
        
        plt.tight_layout()
        plt.savefig('retry_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Retry and error analysis saved as 'retry_error_analysis.png'")
    
    def generate_all_analyses(self):
        """Generate all four analysis visualizations."""
        print("🚀 Generating Comprehensive Analysis Visualizations...\n")
        
        if not self.data:
            print("❌ No data loaded. Cannot generate visualizations.")
            return
        
        try:
            print("📊 1. Generating Provider Comparison Analysis...")
            self.generate_provider_comparison()
            print()
            
            print("💰 2. Generating Cost vs Quality Analysis...")
            self.generate_cost_quality_analysis()
            print()
            
            print("🎯 3. Generating Scenario Complexity Analysis...")
            self.generate_scenario_complexity_analysis()
            print()
            
            print("🔄 4. Generating Retry and Error Pattern Analysis...")
            self.generate_retry_error_analysis()
            print()
            
            print("✅ All analyses completed successfully!")
            print("\n📁 Generated Files:")
            print("   • provider_comparison_analysis.png")
            print("   • cost_quality_analysis.png")
            print("   • scenario_complexity_analysis.png")
            print("   • retry_error_analysis.png")
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function."""
    print("=" * 60)
    print("SRLP Framework - Comprehensive Analysis Visualizations")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = ComprehensiveAnalysisVisualizer()
    
    # Generate all analyses
    visualizer.generate_all_analyses()
    
    print("\n" + "=" * 60)
    print("Analysis Complete! Check the generated PNG files.")
    print("=" * 60)

if __name__ == "__main__":
    main()