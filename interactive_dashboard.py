#!/usr/bin/env python3
"""
Interactive Dashboard for Enhanced SRLP Framework v3.0
Streamlit-based real-time visualization and analysis

Features:
- Real-time metrics visualization
- Provider comparison sliders
- Scenario drill-down analysis
- Cost vs quality trade-off charts
- Interactive filtering and exploration

Author: Enhanced SRLP Framework Team
Date: 2025-01-12
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
from typing import Dict, List, Any

# Page configuration
st.set_page_config(
    page_title="Enhanced SRLP Framework v3.0 Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SRLPDashboard:
    """Interactive dashboard for SRLP Framework analysis"""
    
    def __init__(self):
        self.data = None
        self.results_df = None
        self.load_data()
    
    def load_data(self):
        """Load the latest framework results"""
        try:
            # Try to load the latest results
            if os.path.exists('framework_results_v3.0.json'):
                with open('framework_results_v3.0.json', 'r') as f:
                    self.data = json.load(f)
                self.prepare_dataframes()
            elif os.path.exists('enhanced_framework_results_v3.0.json'):
                with open('enhanced_framework_results_v3.0.json', 'r') as f:
                    self.data = json.load(f)
                self.prepare_dataframes()
            else:
                st.error("No results file found. Please run the framework first.")
                return False
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
        return True
    
    def prepare_dataframes(self):
        """Prepare pandas DataFrames for analysis"""
        if not self.data or 'detailed_results' not in self.data:
            return
        
        # Convert detailed results to DataFrame
        results = []
        for result in self.data['detailed_results']:
            row = {
                'scenario': result['scenario'],
                'provider': result['provider'],
                'success': result['success'],
                'time_accuracy': result['time_accuracy'],
                'improvement_score': result['improvement_score'],
                'framework_time': result['framework_time'],
                'wall_clock_time': result['wall_clock_time'],
                'efficiency': result['efficiency'],
                'retry_attempts': result['retry_attempts'],
                'total_cost': result['cost_metrics']['total_cost'],
                'input_tokens': result['cost_metrics']['input_tokens'],
                'output_tokens': result['cost_metrics']['output_tokens'],
                'quality_score': result['quality_metrics']['custom_score'],
                'completeness': result['quality_metrics']['plan_completeness'],
                'coherence': result['quality_metrics']['logical_coherence'],
                'hallucination_rate': result['quality_metrics']['hallucination_rate'],
                'constraint_violations': result['quality_metrics']['constraint_violations']
            }
            results.append(row)
        
        self.results_df = pd.DataFrame(results)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Data refresh
        if st.sidebar.button("🔄 Refresh Data"):
            self.load_data()
            st.rerun()
        
        # Filters
        st.sidebar.subheader("📊 Filters")
        
        # Provider filter
        providers = ['All'] + list(self.results_df['provider'].unique()) if self.results_df is not None else ['All']
        selected_provider = st.sidebar.selectbox("Provider", providers)
        
        # Scenario filter
        scenarios = ['All'] + list(self.results_df['scenario'].unique()) if self.results_df is not None else ['All']
        selected_scenario = st.sidebar.selectbox("Scenario", scenarios)
        
        # Success filter
        success_filter = st.sidebar.selectbox("Success Status", ['All', 'Success Only', 'Failed Only'])
        
        # Quality threshold
        quality_threshold = st.sidebar.slider("Min Quality Score", 0.0, 1.0, 0.0, 0.1)
        
        # Cost threshold
        max_cost = float(self.results_df['total_cost'].max()) if self.results_df is not None and not self.results_df.empty else 1.0
        cost_threshold = st.sidebar.slider("Max Cost ($)", 0.0, max_cost, max_cost, 0.001)
        
        return {
            'provider': selected_provider,
            'scenario': selected_scenario,
            'success': success_filter,
            'quality_threshold': quality_threshold,
            'cost_threshold': cost_threshold
        }
    
    def apply_filters(self, filters):
        """Apply filters to the DataFrame"""
        if self.results_df is None or self.results_df.empty:
            return pd.DataFrame()
        
        df = self.results_df.copy()
        
        # Provider filter
        if filters['provider'] != 'All':
            df = df[df['provider'] == filters['provider']]
        
        # Scenario filter
        if filters['scenario'] != 'All':
            df = df[df['scenario'] == filters['scenario']]
        
        # Success filter
        if filters['success'] == 'Success Only':
            df = df[df['success'] == True]
        elif filters['success'] == 'Failed Only':
            df = df[df['success'] == False]
        
        # Quality threshold
        df = df[df['quality_score'] >= filters['quality_threshold']]
        
        # Cost threshold
        df = df[df['total_cost'] <= filters['cost_threshold']]
        
        return df
    
    def render_overview(self, df):
        """Render overview metrics"""
        st.header("📊 Framework Overview")
        
        if df.empty:
            st.warning("No data matches the current filters.")
            return
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Tests",
                len(df),
                delta=None
            )
        
        with col2:
            success_rate = (df['success'].sum() / len(df) * 100) if len(df) > 0 else 0
            st.metric(
                "Success Rate",
                f"{success_rate:.1f}%",
                delta=None
            )
        
        with col3:
            avg_quality = df['quality_score'].mean() if not df.empty else 0
            st.metric(
                "Avg Quality",
                f"{avg_quality:.3f}",
                delta=None
            )
        
        with col4:
            total_cost = df['total_cost'].sum() if not df.empty else 0
            st.metric(
                "Total Cost",
                f"${total_cost:.4f}",
                delta=None
            )
        
        with col5:
            avg_time = df['framework_time'].mean() if not df.empty else 0
            st.metric(
                "Avg Time",
                f"{avg_time:.2f}s",
                delta=None
            )
    
    def render_provider_comparison(self, df):
        """Render provider comparison charts"""
        st.header("🏆 Provider Performance Comparison")
        
        if df.empty:
            return
        
        # Provider metrics
        provider_metrics = df.groupby('provider').agg({
            'framework_time': 'mean',
            'quality_score': 'mean',
            'total_cost': 'mean',
            'efficiency': 'mean',
            'success': 'mean',
            'hallucination_rate': 'mean'
        }).round(4)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Average Time', 'Quality Score', 'Cost per Test', 
                          'Efficiency', 'Success Rate', 'Hallucination Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        providers = provider_metrics.index
        colors = px.colors.qualitative.Set1
        
        # Add traces
        fig.add_trace(
            go.Bar(x=providers, y=provider_metrics['framework_time'], 
                   name='Time', marker_color=colors[0 % len(colors)]),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=providers, y=provider_metrics['quality_score'], 
                   name='Quality', marker_color=colors[1 % len(colors)]),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=providers, y=provider_metrics['total_cost'], 
                   name='Cost', marker_color=colors[2 % len(colors)]),
            row=1, col=3
        )
        
        fig.add_trace(
            go.Bar(x=providers, y=provider_metrics['efficiency'], 
                   name='Efficiency', marker_color=colors[3 % len(colors)]),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=providers, y=provider_metrics['success'], 
                   name='Success Rate', marker_color=colors[4 % len(colors)]),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(x=providers, y=provider_metrics['hallucination_rate'], 
                   name='Hallucination', marker_color=colors[5 % len(colors)]),
            row=2, col=3
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Provider Performance Metrics")
        st.plotly_chart(fig, use_container_width=True)
        
        # Provider ranking table
        st.subheader("📈 Provider Rankings")
        
        ranking_df = provider_metrics.copy()
        ranking_df['overall_score'] = (
            (1 - ranking_df['framework_time'] / ranking_df['framework_time'].max()) * 0.2 +
            ranking_df['quality_score'] * 0.3 +
            (1 - ranking_df['total_cost'] / ranking_df['total_cost'].max()) * 0.2 +
            ranking_df['efficiency'] * 0.2 +
            ranking_df['success'] * 0.1
        )
        
        ranking_df = ranking_df.sort_values('overall_score', ascending=False)
        st.dataframe(ranking_df, use_container_width=True)
    
    def render_cost_quality_analysis(self, df):
        """Render cost vs quality analysis"""
        st.header("💰 Cost vs Quality Analysis")
        
        if df.empty:
            return
        
        # Cost vs Quality scatter plot
        fig = px.scatter(
            df, 
            x='total_cost', 
            y='quality_score',
            color='provider',
            size='framework_time',
            hover_data=['scenario', 'efficiency', 'success'],
            title="Cost vs Quality Trade-off",
            labels={
                'total_cost': 'Total Cost ($)',
                'quality_score': 'Quality Score',
                'framework_time': 'Framework Time (s)'
            }
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost efficiency analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost by provider
            cost_by_provider = df.groupby('provider')['total_cost'].sum().sort_values(ascending=False)
            fig_cost = px.pie(
                values=cost_by_provider.values,
                names=cost_by_provider.index,
                title="Total Cost Distribution by Provider"
            )
            st.plotly_chart(fig_cost, use_container_width=True)
        
        with col2:
            # Quality by scenario
            quality_by_scenario = df.groupby('scenario')['quality_score'].mean().sort_values(ascending=False)
            fig_quality = px.bar(
                x=quality_by_scenario.index,
                y=quality_by_scenario.values,
                title="Average Quality by Scenario",
                labels={'x': 'Scenario', 'y': 'Average Quality Score'}
            )
            fig_quality.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_quality, use_container_width=True)
    
    def render_scenario_analysis(self, df):
        """Render scenario complexity analysis"""
        st.header("📋 Scenario Analysis")
        
        if df.empty:
            return
        
        # Scenario performance heatmap
        scenario_provider_metrics = df.pivot_table(
            index='scenario',
            columns='provider',
            values='quality_score',
            aggfunc='mean'
        )
        
        fig = px.imshow(
            scenario_provider_metrics.values,
            x=scenario_provider_metrics.columns,
            y=scenario_provider_metrics.index,
            color_continuous_scale='RdYlGn',
            title="Quality Score Heatmap: Scenario vs Provider",
            labels={'color': 'Quality Score'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Scenario complexity vs performance
        scenario_stats = df.groupby('scenario').agg({
            'quality_score': 'mean',
            'framework_time': 'mean',
            'total_cost': 'mean',
            'success': 'mean'
        }).round(4)
        
        # Add complexity data (from framework)
        complexity_map = {
            'Travel Planning': 0.8,
            'Software Project': 0.9,
            'Event Organization': 0.7,
            'Research Study': 0.85,
            'Business Launch': 0.95
        }
        
        scenario_stats['complexity'] = [complexity_map.get(scenario, 0.5) for scenario in scenario_stats.index]
        
        # Complexity vs Performance scatter
        fig = px.scatter(
            scenario_stats.reset_index(),
            x='complexity',
            y='quality_score',
            size='framework_time',
            color='success',
            hover_data=['scenario', 'total_cost'],
            title="Scenario Complexity vs Quality Performance",
            labels={
                'complexity': 'Scenario Complexity',
                'quality_score': 'Average Quality Score',
                'framework_time': 'Avg Time (s)',
                'success': 'Success Rate'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Scenario statistics table
        st.subheader("📊 Scenario Statistics")
        st.dataframe(scenario_stats, use_container_width=True)
    
    def render_retry_analysis(self, df):
        """Render retry and error analysis"""
        st.header("🔄 Retry & Error Analysis")
        
        if df.empty:
            return
        
        # Retry statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Retry attempts by provider
            retry_by_provider = df.groupby('provider')['retry_attempts'].sum()
            fig = px.bar(
                x=retry_by_provider.index,
                y=retry_by_provider.values,
                title="Total Retry Attempts by Provider",
                labels={'x': 'Provider', 'y': 'Total Retries'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Success rate vs retry attempts
            retry_success = df.groupby('retry_attempts').agg({
                'success': 'mean',
                'provider': 'count'
            }).rename(columns={'provider': 'count'})
            
            fig = px.bar(
                retry_success.reset_index(),
                x='retry_attempts',
                y='success',
                title="Success Rate by Retry Attempts",
                labels={'retry_attempts': 'Retry Attempts', 'success': 'Success Rate'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Error analysis from network events
        if self.data and 'retry_events' in self.data:
            retry_events = pd.DataFrame(self.data['retry_events'])
            if not retry_events.empty:
                st.subheader("🚨 Error Type Analysis")
                
                error_counts = retry_events['error_type'].value_counts()
                fig = px.pie(
                    values=error_counts.values,
                    names=error_counts.index,
                    title="Distribution of Error Types"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_detailed_data(self, df):
        """Render detailed data table"""
        st.header("📋 Detailed Test Results")
        
        if df.empty:
            st.warning("No data to display.")
            return
        
        # Add download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"srlp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Display data with formatting
        st.dataframe(
            df.style.format({
                'framework_time': '{:.2f}',
                'quality_score': '{:.3f}',
                'total_cost': '${:.4f}',
                'efficiency': '{:.3f}',
                'hallucination_rate': '{:.3f}'
            }),
            use_container_width=True
        )
    
    def run(self):
        """Main dashboard execution"""
        # Title and description
        st.title("🚀 Enhanced SRLP Framework v3.0 Dashboard")
        st.markdown("""
        **Interactive Analysis Dashboard for Self-Refinement LLM Planners Framework**
        
        This dashboard provides comprehensive analysis of framework performance, including:
        - Provider comparison and ranking
        - Cost vs quality trade-off analysis
        - Scenario complexity evaluation
        - Retry and error pattern analysis
        """)
        
        if self.data is None:
            st.error("Please run the framework first to generate data.")
            return
        
        # Sidebar controls
        filters = self.render_sidebar()
        
        # Apply filters
        filtered_df = self.apply_filters(filters)
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "🏆 Providers", 
            "💰 Cost/Quality", 
            "📋 Scenarios", 
            "🔄 Retries", 
            "📋 Data"
        ])
        
        with tab1:
            self.render_overview(filtered_df)
        
        with tab2:
            self.render_provider_comparison(filtered_df)
        
        with tab3:
            self.render_cost_quality_analysis(filtered_df)
        
        with tab4:
            self.render_scenario_analysis(filtered_df)
        
        with tab5:
            self.render_retry_analysis(filtered_df)
        
        with tab6:
            self.render_detailed_data(filtered_df)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "**Enhanced SRLP Framework v3.0** | "
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

def main():
    """Main function"""
    dashboard = SRLPDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()