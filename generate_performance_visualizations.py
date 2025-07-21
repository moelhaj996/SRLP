#!/usr/bin/env python3
"""
SRLP Framework v3.0 - Performance Visualization Generator

This script generates comprehensive visualizations based on the actionable suggestions:
- Average Wall_Clock_Time per provider
- Retry_Time frequency by config
- Scenario complexity analysis
- Provider efficiency comparisons
- Bottleneck identification charts

Author: AI Research Team
Version: 1.0
Date: 2025-07-12
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any

class PerformanceVisualizer:
    """Generate comprehensive performance visualizations for SRLP Framework analysis."""
    
    def __init__(self, csv_file: str = "visualization_data.csv", 
                 results_file: str = "enhanced_framework_results_v2.1.json"):
        self.csv_file = csv_file
        self.results_file = results_file
        self.df = None
        self.results = None
        
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self):
        """Load data from CSV and JSON files."""
        try:
            # Load CSV data
            if Path(self.csv_file).exists():
                self.df = pd.read_csv(self.csv_file)
                print(f"✅ Loaded {len(self.df)} records from {self.csv_file}")
            else:
                print(f"❌ CSV file {self.csv_file} not found")
                return False
                
            # Load JSON results
            if Path(self.results_file).exists():
                with open('framework_results_v3.0.json', 'r') as f:
                    self.results = json.load(f)
                print(f"✅ Loaded results from {self.results_file}")
            else:
                print(f"⚠️  JSON file {self.results_file} not found - some visualizations may be limited")
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def calculate_provider_averages(self) -> Dict[str, float]:
        """Calculate average Wall_Clock_Time per provider as mentioned in suggestions."""
        if self.df is None:
            return {}
            
        provider_stats = self.df.groupby('Provider').agg({
            'Wall_Clock_Time': ['mean', 'std', 'count'],
            'Retry_Time': 'sum',
            'Improvement_Score': 'mean'
        }).round(6)
        
        # Flatten column names
        provider_stats.columns = ['_'.join(col).strip() for col in provider_stats.columns]
        
        print("\n📊 Provider Performance Analysis (as suggested):")
        print("Average Wall_Clock_Time per Provider:")
        
        averages = {}
        for provider in provider_stats.index:
            avg_time = provider_stats.loc[provider, 'Wall_Clock_Time_mean']
            count = provider_stats.loc[provider, 'Wall_Clock_Time_count']
            total_time = avg_time * count
            
            averages[provider] = avg_time
            
            if provider == 'mock':
                print(f"   • {provider.capitalize()}: ~{avg_time:.5f}s ({avg_time*1000000:.0f}µs) – negligible due to simulation")
            else:
                print(f"   • {provider.capitalize()}: ~{avg_time:.2f}s (sum: {total_time:.2f}s / {count} entries)")
        
        return averages
    
    def generate_provider_comparison_chart(self):
        """Generate comprehensive provider comparison chart."""
        if self.df is None:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Provider Performance Analysis - Enhanced SRLP Framework', fontsize=16, fontweight='bold')
        
        # 1. Average Wall Clock Time per Provider
        provider_times = self.df.groupby('Provider')['Wall_Clock_Time'].mean().sort_values(ascending=True)
        
        ax1 = axes[0, 0]
        bars = ax1.bar(provider_times.index, provider_times.values, 
                      color=['lightcoral', 'skyblue', 'lightgreen', 'gold'])
        ax1.set_title('Average Wall Clock Time per Provider')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, provider_times.values):
            if value < 0.001:  # For mock provider
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(provider_times.values)*0.01,
                        f'{value*1000000:.0f}µs', ha='center', va='bottom', fontweight='bold')
            else:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(provider_times.values)*0.01,
                        f'{value:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # 2. Retry Time Distribution
        retry_data = self.df.groupby('Provider')['Retry_Time'].sum()
        
        ax2 = axes[0, 1]
        if retry_data.sum() > 0:
            wedges, texts, autotexts = ax2.pie(retry_data.values, labels=retry_data.index, autopct='%1.1f%%')
            ax2.set_title('Total Retry Time Distribution by Provider')
        else:
            ax2.text(0.5, 0.5, 'No Retry Time\nRecorded', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14, fontweight='bold')
            ax2.set_title('Retry Time Distribution (None Recorded)')
        
        # 3. Provider Efficiency Box Plot
        ax3 = axes[1, 0]
        providers = self.df['Provider'].unique()
        efficiency_data = []
        labels = []
        
        for provider in providers:
            provider_data = self.df[self.df['Provider'] == provider]
            if len(provider_data) > 0:
                # Calculate efficiency as Framework_Time / Wall_Clock_Time
                efficiency = provider_data['Framework_Time'] / provider_data['Wall_Clock_Time']
                efficiency_data.append(efficiency.values)
                labels.append(provider)
        
        if efficiency_data:
            ax3.boxplot(efficiency_data, labels=labels)
            ax3.set_title('Provider Efficiency Distribution')
            ax3.set_ylabel('Efficiency Ratio (Framework/Wall Time)')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Scenario Performance by Provider
        ax4 = axes[1, 1]
        scenario_provider = self.df.pivot_table(values='Wall_Clock_Time', 
                                               index='Scenario', 
                                               columns='Provider', 
                                               aggfunc='mean')
        
        im = ax4.imshow(scenario_provider.values, cmap='YlOrRd', aspect='auto')
        ax4.set_xticks(range(len(scenario_provider.columns)))
        ax4.set_yticks(range(len(scenario_provider.index)))
        ax4.set_xticklabels(scenario_provider.columns, rotation=45)
        ax4.set_yticklabels([s.replace('_', ' ').title() for s in scenario_provider.index])
        ax4.set_title('Scenario Performance Heatmap (Wall Clock Time)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Average Time (seconds)')
        
        plt.tight_layout()
        plt.savefig('comprehensive_provider_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Generated: comprehensive_provider_analysis.png")
    
    def generate_retry_analysis_chart(self):
        """Generate detailed retry analysis as suggested."""
        if self.df is None:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Retry Analysis - Framework Optimization Insights', fontsize=16, fontweight='bold')
        
        # 1. Retry Time Frequency by Config
        config_retry = self.df.groupby('Config')['Retry_Time'].agg(['sum', 'count', 'mean'])
        
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(config_retry.index)), config_retry['sum'].values)
        ax1.set_title('Total Retry Time by Configuration')
        ax1.set_ylabel('Total Retry Time (seconds)')
        ax1.set_xticks(range(len(config_retry.index)))
        ax1.set_xticklabels(config_retry.index, rotation=45, ha='right')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, config_retry['sum'].values)):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(config_retry['sum'].values)*0.01,
                        f'{value:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 2. Retry Impact on Performance
        ax2 = axes[0, 1]
        
        # Scatter plot: Wall Clock Time vs Retry Time
        for provider in self.df['Provider'].unique():
            provider_data = self.df[self.df['Provider'] == provider]
            ax2.scatter(provider_data['Retry_Time'], provider_data['Wall_Clock_Time'], 
                       label=provider, alpha=0.7, s=60)
        
        ax2.set_xlabel('Retry Time (seconds)')
        ax2.set_ylabel('Wall Clock Time (seconds)')
        ax2.set_title('Retry Time Impact on Total Execution Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Provider Retry Patterns
        ax3 = axes[1, 0]
        
        retry_by_provider = self.df.groupby('Provider')['Retry_Time'].agg(['sum', 'count', 'mean'])
        
        x = np.arange(len(retry_by_provider.index))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, retry_by_provider['sum'], width, label='Total Retry Time', alpha=0.8)
        bars2 = ax3.bar(x + width/2, retry_by_provider['count'], width, label='Retry Count', alpha=0.8)
        
        ax3.set_xlabel('Provider')
        ax3.set_ylabel('Retry Metrics')
        ax3.set_title('Provider Retry Patterns')
        ax3.set_xticks(x)
        ax3.set_xticklabels(retry_by_provider.index)
        ax3.legend()
        
        # 4. Efficiency vs Retry Correlation
        ax4 = axes[1, 1]
        
        # Calculate efficiency and plot against retry time
        self.df['Efficiency'] = self.df['Framework_Time'] / self.df['Wall_Clock_Time']
        
        ax4.scatter(self.df['Retry_Time'], self.df['Efficiency'], alpha=0.6, s=50)
        ax4.set_xlabel('Retry Time (seconds)')
        ax4.set_ylabel('Efficiency Ratio')
        ax4.set_title('Retry Time vs Efficiency Correlation')
        ax4.grid(True, alpha=0.3)
        
        # Add trend line
        if len(self.df[self.df['Retry_Time'] > 0]) > 1:
            retry_data = self.df[self.df['Retry_Time'] > 0]
            z = np.polyfit(retry_data['Retry_Time'], retry_data['Efficiency'], 1)
            p = np.poly1d(z)
            ax4.plot(retry_data['Retry_Time'], p(retry_data['Retry_Time']), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig('retry_analysis_insights.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Generated: retry_analysis_insights.png")
    
    def generate_scenario_complexity_analysis(self):
        """Generate scenario-specific analysis for tuning insights."""
        if self.df is None:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Scenario Complexity Analysis - Optimization Opportunities', fontsize=16, fontweight='bold')
        
        # 1. Scenario Performance Ranking
        scenario_stats = self.df.groupby('Scenario').agg({
            'Wall_Clock_Time': ['mean', 'std'],
            'Framework_Time': 'mean',
            'Improvement_Score': 'mean',
            'Iterations': 'mean'
        })
        
        scenario_stats.columns = ['_'.join(col).strip() for col in scenario_stats.columns]
        scenario_stats = scenario_stats.sort_values('Wall_Clock_Time_mean', ascending=False)
        
        ax1 = axes[0, 0]
        bars = ax1.barh(range(len(scenario_stats.index)), scenario_stats['Wall_Clock_Time_mean'].values)
        ax1.set_title('Scenario Performance Ranking (Avg Wall Clock Time)')
        ax1.set_xlabel('Average Time (seconds)')
        ax1.set_yticks(range(len(scenario_stats.index)))
        ax1.set_yticklabels([s.replace('_', ' ').title() for s in scenario_stats.index])
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, scenario_stats['Wall_Clock_Time_mean'].values)):
            ax1.text(value + max(scenario_stats['Wall_Clock_Time_mean'].values)*0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{value:.2f}s', ha='left', va='center', fontweight='bold')
        
        # 2. Scenario Complexity vs Performance
        ax2 = axes[0, 1]
        
        # Define complexity factors (as suggested in the analysis)
        complexity_factors = {
            'travel_planning': 0.8,
            'cooking_dinner': 0.4,
            'software_project': 0.9,
            'conference_planning': 0.6,
            'kitchen_renovation': 1.0
        }
        
        scenario_perf = self.df.groupby('Scenario')['Wall_Clock_Time'].mean()
        complexities = [complexity_factors.get(scenario, 0.5) for scenario in scenario_perf.index]
        
        scatter = ax2.scatter(complexities, scenario_perf.values, s=100, alpha=0.7, c=range(len(scenario_perf)))
        ax2.set_xlabel('Complexity Factor')
        ax2.set_ylabel('Average Wall Clock Time (seconds)')
        ax2.set_title('Scenario Complexity vs Performance')
        
        # Add scenario labels
        for i, (complexity, time, scenario) in enumerate(zip(complexities, scenario_perf.values, scenario_perf.index)):
            ax2.annotate(scenario.replace('_', '\n'), (complexity, time), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Add trend line
        z = np.polyfit(complexities, scenario_perf.values, 1)
        p = np.poly1d(z)
        ax2.plot(complexities, p(complexities), "r--", alpha=0.8, linewidth=2)
        
        # 3. Provider Performance by Scenario
        ax3 = axes[1, 0]
        
        scenario_provider_perf = self.df.pivot_table(values='Wall_Clock_Time', 
                                                    index='Scenario', 
                                                    columns='Provider', 
                                                    aggfunc='mean')
        
        # Create stacked bar chart
        scenario_provider_perf.plot(kind='bar', stacked=True, ax=ax3, alpha=0.8)
        ax3.set_title('Provider Performance by Scenario (Stacked)')
        ax3.set_ylabel('Wall Clock Time (seconds)')
        ax3.set_xlabel('Scenario')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Provider', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Bottleneck Identification
        ax4 = axes[1, 1]
        
        # Calculate bottleneck score (high time + high variance)
        bottleneck_scores = []
        scenario_names = []
        
        for scenario in scenario_stats.index:
            mean_time = scenario_stats.loc[scenario, 'Wall_Clock_Time_mean']
            std_time = scenario_stats.loc[scenario, 'Wall_Clock_Time_std']
            # Normalize and combine (higher = more bottleneck)
            score = (mean_time / scenario_stats['Wall_Clock_Time_mean'].max()) + \
                   (std_time / scenario_stats['Wall_Clock_Time_std'].max())
            bottleneck_scores.append(score)
            scenario_names.append(scenario.replace('_', ' ').title())
        
        colors = ['red' if score > 1.5 else 'orange' if score > 1.0 else 'green' for score in bottleneck_scores]
        bars = ax4.bar(range(len(scenario_names)), bottleneck_scores, color=colors, alpha=0.7)
        
        ax4.set_title('Bottleneck Analysis (High Score = Optimization Target)')
        ax4.set_ylabel('Bottleneck Score')
        ax4.set_xticks(range(len(scenario_names)))
        ax4.set_xticklabels(scenario_names, rotation=45, ha='right')
        
        # Add threshold line
        ax4.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='High Priority')
        ax4.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Medium Priority')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('scenario_complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Generated: scenario_complexity_analysis.png")
    
    def generate_improvement_recommendations(self):
        """Generate actionable improvement recommendations chart."""
        if self.df is None:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Actionable Improvement Recommendations', fontsize=16, fontweight='bold')
        
        # 1. Provider Efficiency Ranking
        ax1 = axes[0, 0]
        
        provider_efficiency = self.df.groupby('Provider').agg({
            'Wall_Clock_Time': 'mean',
            'Retry_Time': 'sum',
            'Improvement_Score': 'mean'
        })
        
        # Calculate efficiency score (lower time + higher improvement = better)
        max_time = provider_efficiency['Wall_Clock_Time'].max()
        efficiency_scores = []
        
        for provider in provider_efficiency.index:
            time_score = 1 - (provider_efficiency.loc[provider, 'Wall_Clock_Time'] / max_time)
            improvement_score = provider_efficiency.loc[provider, 'Improvement_Score']
            retry_penalty = provider_efficiency.loc[provider, 'Retry_Time'] / 10  # Penalty for retries
            
            total_score = time_score + improvement_score - retry_penalty
            efficiency_scores.append(max(0, total_score))
        
        colors = ['green' if score > 0.8 else 'orange' if score > 0.5 else 'red' for score in efficiency_scores]
        bars = ax1.bar(provider_efficiency.index, efficiency_scores, color=colors, alpha=0.7)
        
        ax1.set_title('Provider Efficiency Ranking')
        ax1.set_ylabel('Efficiency Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add recommendations as text
        for i, (bar, score, provider) in enumerate(zip(bars, efficiency_scores, provider_efficiency.index)):
            if score < 0.5:
                recommendation = "Optimize"
            elif score < 0.8:
                recommendation = "Improve"
            else:
                recommendation = "Excellent"
            
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    recommendation, ha='center', va='bottom', fontweight='bold')
        
        # 2. Retry Optimization Targets
        ax2 = axes[0, 1]
        
        retry_analysis = self.df.groupby('Provider')['Retry_Time'].agg(['sum', 'count'])
        retry_analysis['avg_retry'] = retry_analysis['sum'] / retry_analysis['count']
        
        # Only show providers with retries
        retry_providers = retry_analysis[retry_analysis['sum'] > 0]
        
        if len(retry_providers) > 0:
            bars = ax2.bar(retry_providers.index, retry_providers['sum'], alpha=0.7, color='coral')
            ax2.set_title('Retry Optimization Targets')
            ax2.set_ylabel('Total Retry Time (seconds)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add optimization suggestions
            for bar, provider in zip(bars, retry_providers.index):
                total_retry = retry_providers.loc[provider, 'sum']
                if total_retry > 5:
                    suggestion = "High Priority"
                elif total_retry > 1:
                    suggestion = "Medium Priority"
                else:
                    suggestion = "Low Priority"
                
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(retry_providers['sum'])*0.02,
                        suggestion, ha='center', va='bottom', fontweight='bold', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No Retry Issues\nDetected', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14, fontweight='bold', color='green')
            ax2.set_title('Retry Analysis - All Good!')
        
        # 3. Mock Provider Enhancement Suggestions
        ax3 = axes[1, 0]
        
        mock_data = self.df[self.df['Provider'] == 'mock']
        if len(mock_data) > 0:
            current_times = mock_data['Wall_Clock_Time'].values
            suggested_times = np.random.uniform(1, 5, len(current_times))  # 1-5s as suggested
            
            ax3.hist(current_times * 1000000, bins=20, alpha=0.7, label='Current (µs)', color='lightblue')
            ax3.hist(suggested_times, bins=20, alpha=0.7, label='Suggested (s)', color='lightgreen')
            
            ax3.set_title('Mock Provider Latency Enhancement')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            
            # Add recommendation text
            ax3.text(0.05, 0.95, 'Recommendation:\nAdd 1-5s latency\nfor realism', 
                    transform=ax3.transAxes, fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 4. Overall Optimization Priority Matrix
        ax4 = axes[1, 1]
        
        # Create priority matrix based on time and retry issues
        providers = self.df['Provider'].unique()
        x_coords = []  # Retry time
        y_coords = []  # Average wall time
        labels = []
        colors = []
        
        for provider in providers:
            provider_data = self.df[self.df['Provider'] == provider]
            avg_time = provider_data['Wall_Clock_Time'].mean()
            total_retry = provider_data['Retry_Time'].sum()
            
            x_coords.append(total_retry)
            y_coords.append(avg_time)
            labels.append(provider)
            
            # Color based on priority (high time + high retry = red)
            if avg_time > 5 and total_retry > 1:
                colors.append('red')
            elif avg_time > 2 or total_retry > 0.5:
                colors.append('orange')
            else:
                colors.append('green')
        
        scatter = ax4.scatter(x_coords, y_coords, c=colors, s=200, alpha=0.7)
        
        for i, label in enumerate(labels):
            ax4.annotate(label, (x_coords[i], y_coords[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax4.set_xlabel('Total Retry Time (seconds)')
        ax4.set_ylabel('Average Wall Clock Time (seconds)')
        ax4.set_title('Optimization Priority Matrix')
        
        # Add quadrant lines
        ax4.axhline(y=np.mean(y_coords), color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=np.mean(x_coords), color='gray', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax4.text(0.02, 0.98, 'Low Time\nHigh Retry', transform=ax4.transAxes, 
                fontsize=9, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        ax4.text(0.98, 0.98, 'High Time\nHigh Retry', transform=ax4.transAxes, 
                fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
        ax4.text(0.02, 0.02, 'Low Time\nLow Retry', transform=ax4.transAxes, 
                fontsize=9, ha='left', va='bottom', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
        ax4.text(0.98, 0.02, 'High Time\nLow Retry', transform=ax4.transAxes, 
                fontsize=9, ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('improvement_recommendations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Generated: improvement_recommendations.png")
    
    def print_actionable_insights(self):
        """Print actionable insights based on the analysis."""
        if self.df is None:
            return
            
        print("\n" + "=" * 80)
        print("🎯 ACTIONABLE INSIGHTS & RECOMMENDATIONS")
        print("=" * 80)
        
        # Calculate key metrics
        provider_averages = self.calculate_provider_averages()
        
        print("\n🔧 OPTIMIZATION PRIORITIES:")
        
        # Identify slowest providers
        sorted_providers = sorted(provider_averages.items(), key=lambda x: x[1], reverse=True)
        
        print("\n1️⃣  Provider Optimization (by avg time):")
        for i, (provider, avg_time) in enumerate(sorted_providers):
            if provider == 'mock':
                continue
            
            if i == 0:
                print(f"   🔴 HIGH PRIORITY: {provider} ({avg_time:.2f}s avg)")
                print(f"      → Implement request batching")
                print(f"      → Add connection pooling")
                print(f"      → Consider parallel processing")
            elif i == 1:
                print(f"   🟡 MEDIUM PRIORITY: {provider} ({avg_time:.2f}s avg)")
                print(f"      → Optimize API call patterns")
                print(f"      → Review timeout settings")
            else:
                print(f"   🟢 LOW PRIORITY: {provider} ({avg_time:.2f}s avg)")
                print(f"      → Monitor for regression")
        
        # Retry analysis
        total_retries = self.df['Retry_Time'].sum()
        if total_retries > 0:
            print("\n2️⃣  Retry Logic Optimization:")
            retry_by_provider = self.df.groupby('Provider')['Retry_Time'].sum().sort_values(ascending=False)
            
            for provider, retry_time in retry_by_provider.items():
                if retry_time > 0:
                    print(f"   • {provider}: {retry_time:.1f}s total retry time")
                    if retry_time > 5:
                        print(f"     → URGENT: Implement exponential backoff")
                        print(f"     → Set maximum retry limit (2 attempts)")
                    elif retry_time > 1:
                        print(f"     → Review retry triggers")
                        print(f"     → Adjust timeout thresholds")
        else:
            print("\n2️⃣  Retry Logic: ✅ No issues detected")
        
        # Mock provider enhancement
        mock_data = self.df[self.df['Provider'] == 'mock']
        if len(mock_data) > 0:
            avg_mock_time = mock_data['Wall_Clock_Time'].mean()
            print("\n3️⃣  Mock Provider Enhancement:")
            print(f"   • Current avg time: {avg_mock_time*1000000:.0f}µs (unrealistic)")
            print(f"   • Recommended: Add 1-5s latency simulation")
            print(f"   • Benefits: Better testing of retry logic and timeouts")
        
        # Scenario-specific recommendations
        scenario_times = self.df.groupby('Scenario')['Wall_Clock_Time'].mean().sort_values(ascending=False)
        print("\n4️⃣  Scenario-Specific Tuning:")
        
        for i, (scenario, avg_time) in enumerate(scenario_times.items()):
            if i < 2:  # Top 2 slowest scenarios
                print(f"   🎯 {scenario.replace('_', ' ').title()}: {avg_time:.2f}s avg")
                if 'travel' in scenario:
                    print(f"      → Simplify travel planning inputs")
                    print(f"      → Cache common routes/destinations")
                elif 'software' in scenario:
                    print(f"      → Optimize code analysis algorithms")
                    print(f"      → Implement incremental processing")
                elif 'kitchen' in scenario:
                    print(f"      → Break down complex renovation tasks")
                    print(f"      → Use parallel planning for different rooms")
                else:
                    print(f"      → Profile resource usage for bottlenecks")
                    print(f"      → Consider task decomposition")
        
        print("\n📊 IMPLEMENTATION ROADMAP:")
        print("   Phase 1: Cap retries at 2 attempts (immediate)")
        print("   Phase 2: Add realistic mock latency (1-2 days)")
        print("   Phase 3: Optimize slowest provider (1 week)")
        print("   Phase 4: Implement parallel processing (2 weeks)")
        print("   Phase 5: Scenario-specific optimizations (ongoing)")
        
        print("\n🎉 Expected Improvements:")
        print(f"   • 50-70% reduction in retry delays")
        print(f"   • 20-30% improvement in provider efficiency")
        print(f"   • Better testing coverage with realistic mock latency")
        print(f"   • More accurate performance benchmarks")
        
        print("=" * 80)
    
    def generate_all_visualizations(self):
        """Generate all visualization charts."""
        print("🎨 Generating comprehensive performance visualizations...")
        print("=" * 60)
        
        if not self.load_data():
            print("❌ Failed to load data - cannot generate visualizations")
            return False
        
        try:
            # Generate all charts
            self.generate_provider_comparison_chart()
            self.generate_retry_analysis_chart()
            self.generate_scenario_complexity_analysis()
            self.generate_improvement_recommendations()
            
            # Print insights
            self.print_actionable_insights()
            
            print("\n✅ All visualizations generated successfully!")
            print("\n📁 Generated Files:")
            print("   • comprehensive_provider_analysis.png")
            print("   • retry_analysis_insights.png")
            print("   • scenario_complexity_analysis.png")
            print("   • improvement_recommendations.png")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            return False

def main():
    """Main execution function."""
    print("🎨 Performance Visualization Generator for Enhanced SRLP Framework")
    print("📊 Implementing actionable suggestions for comprehensive analysis")
    print("=" * 70)
    
    # Try different file combinations
    file_combinations = [
        ("visualization_data.csv", "enhanced_framework_results_v2.1.json"),
        ("enhanced_visualization_data_v2.2.csv", "enhanced_framework_results_v2.2.json"),
        ("visualization_data.csv", "enhanced_framework_results_v2.json")
    ]
    
    visualizer = None
    for csv_file, json_file in file_combinations:
        if Path(csv_file).exists():
            print(f"📂 Found data file: {csv_file}")
            visualizer = PerformanceVisualizer(csv_file, json_file)
            break
    
    if visualizer is None:
        print("❌ No visualization data files found.")
        print("   Please run the Enhanced SRLP Framework first to generate data.")
        return False
    
    # Generate all visualizations
    success = visualizer.generate_all_visualizations()
    
    if success:
        print("\n🎉 Visualization generation completed successfully!")
        print("📈 Use these charts to implement the suggested optimizations.")
    else:
        print("\n❌ Some visualizations failed to generate.")
    
    return success

if __name__ == "__main__":
    main()