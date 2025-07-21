"""
Main CLI interface for the SRLP framework with LLM provider support.
Provides command-line access to all framework functionality with any LLM provider.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append('/Users/mohamedelhajsuliman/Desktop/Mohamed 2025 summer thesis')

# Original path (commented out)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from srlp_framework.core.evaluator import Evaluator
from srlp_framework.core.metrics_calculator import BasicMetricsCalculator
from srlp_framework.core.refinement_engine import RefinementEngine
from refinement_engine import create_refinement_engine
from srlp_framework.test_scenarios import get_scenario_by_name, get_all_test_scenarios, load_scenario_from_file
from srlp_framework.utils.visualization import generate_all_visualizations
from srlp_framework.llm_providers import LLMFactory, list_available_providers


def run_single_evaluation(problem_file: str = None, scenario: str = None, 
                         evaluate: bool = True, export: str = None, 
                         visualize: bool = False, provider: str = "mock",
                         model: str = None, **llm_kwargs) -> Dict[str, Any]:
    """
    Run a single evaluation with the SRLP framework using specified LLM provider.
    
    Args:
        problem_file: Path to JSON file containing problem specification
        scenario: Name of predefined scenario to run
        evaluate: Whether to run full evaluation with metrics
        export: Path to export results
        visualize: Whether to generate visualizations
        provider: LLM provider name ('openai', 'claude', 'llama', 'huggingface', 'mock')
        model: Model name (optional, uses provider default)
        **llm_kwargs: Additional LLM configuration parameters
        
    Returns:
        Dictionary with evaluation results
    """
    
    # Load problem
    if problem_file:
        if not os.path.exists(problem_file):
            raise FileNotFoundError(f"Problem file not found: {problem_file}")
        
        with open(problem_file, 'r') as f:
            scenario_data = json.load(f)
        
        scenario_name = scenario_data.get('name', 'custom_scenario')
        problem = scenario_data.get('problem', scenario_data)
        
    elif scenario:
        scenario_data = get_scenario_by_name(scenario)
        scenario_name = scenario_data['name']
        problem = scenario_data['problem']
        
    else:
        # Default to travel scenario
        scenario_data = get_scenario_by_name('travel')
        scenario_name = scenario_data['name']
        problem = scenario_data['problem']
    
    print(f"Running SRLP evaluation: {scenario_name}")
    print(f"Problem: {problem.get('goal', 'No goal specified')}")
    print(f"Type: {problem.get('type', 'general')}")
    print(f"LLM Provider: {provider}")
    if model:
        print(f"Model: {model}")
    print()
    
    # Create LLM-enabled components
    try:
        # Create refinement engine with the specified provider
        refinement_engine = create_refinement_engine(provider=provider, model=model, **llm_kwargs)
        
        # Get provider info
        provider_info = refinement_engine.llm.get_provider_info()
        print(f"Using LLM: {provider_info}")
        
    except Exception as e:
        print(f"Error initializing {provider} provider: {e}")
        print("Falling back to mock provider...")
        refinement_engine = create_refinement_engine(provider="mock")
    
    # Initialize other components
    calculator = BasicMetricsCalculator()
    
    # Run evaluation
    if evaluate:
        print("Running comprehensive evaluation...")
        
        # Use the LLM-enabled refinement engine
        refinement_result = refinement_engine.refine_plan(problem)
        
        # Extract metrics
        initial_plan = refinement_result.initial_plan
        final_plan = refinement_result.final_plan
        
        # Get check results
        initial_check_data = refinement_result.refinement_history[0]['check_result']
        final_check_data = refinement_result.refinement_history[-1]['check_result']
        
        # Convert to CheckResult objects
        from srlp_framework.core.self_checker import CheckResult
        
        initial_check = CheckResult(
            overall_score=initial_check_data['overall_score'],
            error_count=initial_check_data['error_count'],
            errors=initial_check_data['errors'],
            constraint_violations=initial_check_data['constraint_violations'],
            uncertainty_scores=initial_check_data['uncertainty_scores'],
            semantic_consistency=initial_check_data['semantic_consistency'],
            completeness_score=initial_check_data['completeness_score']
        )
        
        final_check = CheckResult(
            overall_score=final_check_data['overall_score'],
            error_count=final_check_data['error_count'],
            errors=final_check_data['errors'],
            constraint_violations=final_check_data['constraint_violations'],
            uncertainty_scores=final_check_data['uncertainty_scores'],
            semantic_consistency=final_check_data['semantic_consistency'],
            completeness_score=final_check_data['completeness_score']
        )
        
        # Calculate metrics
        metrics_before = calculator.calculate_metrics(initial_plan, problem, initial_check)
        metrics_after = calculator.calculate_metrics(final_plan, problem, final_check)
        improvement_metrics = calculator.compare_metrics(metrics_before, metrics_after)
        
        # Display results
        print("Results:")
        print("-" * 40)
        print(f"Initial Quality: {metrics_before.quality_metrics['overall_quality_score']:.3f}")
        print(f"Final Quality: {metrics_after.quality_metrics['overall_quality_score']:.3f}")
        print(f"Improvement: {improvement_metrics['overall_quality_score_absolute_improvement']:+.3f}")
        print(f"Relative Improvement: {improvement_metrics['overall_quality_score_relative_improvement']:+.1f}%")
        print(f"Iterations: {refinement_result.iterations}")
        print(f"Converged: {'Yes' if refinement_result.converged else 'No'}")
        print(f"Processing Time: {refinement_result.total_time:.2f}s")
        provider_info = refinement_engine.llm.get_provider_info()
        print(f"LLM Provider: {provider_info.get('provider', 'unknown')}")
        print(f"LLM Model: {provider_info.get('model', 'unknown')}")
        
        # Prepare results
        results = {
            'scenario': scenario_name,
            'problem': problem,
            'refinement_result': refinement_result.to_dict(),
            'metrics_before': metrics_before.to_dict(),
            'metrics_after': metrics_after.to_dict(),
            'improvement_metrics': improvement_metrics,
            'llm_info': provider_info
        }
        
    else:
        # Simple refinement without full evaluation
        print("Running basic refinement...")
        refinement_result = refinement_engine.refine_plan(problem)
        
        print("Results:")
        print("-" * 40)
        print(f"Iterations: {refinement_result.iterations}")
        print(f"Converged: {'Yes' if refinement_result.converged else 'No'}")
        print(f"Improvement: {refinement_result.improvement_score:.3f}")
        print(f"Processing Time: {refinement_result.total_time:.2f}s")
        provider_info = refinement_engine.llm.get_provider_info()
        print(f"LLM Provider: {provider_info.get('provider', 'unknown')}")
        print(f"LLM Model: {provider_info.get('model', 'unknown')}")
        
        results = {
            'scenario': scenario_name,
            'problem': problem,
            'refinement_result': refinement_result.to_dict(),
            'llm_info': provider_info
        }
    
    # Export results
    if export:
        export_results(results, export, evaluate)
    
    # Generate visualizations
    if visualize and evaluate:
        print("\nGenerating visualizations...")
        viz_dir = os.path.join(os.path.dirname(export) if export else 'results', 'visualizations')
        generate_all_visualizations([results], viz_dir)
        print(f"Visualizations saved to: {viz_dir}")
    
    return results


def run_multiple_evaluations(scenarios: List[str] = None, export: str = None, 
                           visualize: bool = False, provider: str = "mock",
                           model: str = None, **llm_kwargs) -> List[Dict[str, Any]]:
    """
    Run evaluations on multiple scenarios with specified LLM provider.
    
    Args:
        scenarios: List of scenario names to evaluate
        export: Path to export aggregate results
        visualize: Whether to generate visualizations
        provider: LLM provider name
        model: Model name (optional)
        **llm_kwargs: Additional LLM configuration parameters
        
    Returns:
        List of evaluation results
    """
    
    if scenarios is None:
        all_scenarios = get_all_test_scenarios()
        scenarios = [s['name'] for s in all_scenarios]
    
    print(f"Running SRLP evaluation on {len(scenarios)} scenarios")
    print(f"LLM Provider: {provider}")
    if model:
        print(f"Model: {model}")
    print("=" * 60)
    
    results = []
    
    for i, scenario_name in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Evaluating: {scenario_name}")
        print("-" * 40)
        
        try:
            result = run_single_evaluation(
                scenario=scenario_name, 
                evaluate=True,
                provider=provider,
                model=model,
                **llm_kwargs
            )
            results.append(result)
            
            # Brief summary
            before_quality = result['metrics_before']['quality_metrics']['overall_quality_score']
            after_quality = result['metrics_after']['quality_metrics']['overall_quality_score']
            improvement = after_quality - before_quality
            converged = result['refinement_result']['converged']
            
            print(f"Quality: {before_quality:.3f} → {after_quality:.3f} ({improvement:+.3f})")
            print(f"Converged: {'Yes' if converged else 'No'}")
            
        except Exception as e:
            print(f"Error evaluating {scenario_name}: {e}")
            continue
    
    # Generate summary
    if results:
        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        
        avg_initial = sum(r['metrics_before']['quality_metrics']['overall_quality_score'] 
                         for r in results) / len(results)
        avg_final = sum(r['metrics_after']['quality_metrics']['overall_quality_score'] 
                       for r in results) / len(results)
        avg_improvement = avg_final - avg_initial
        success_rate = sum(1 for r in results if r['refinement_result']['converged']) / len(results)
        
        print(f"Scenarios Evaluated: {len(results)}")
        print(f"Average Initial Quality: {avg_initial:.3f}")
        print(f"Average Final Quality: {avg_final:.3f}")
        print(f"Average Improvement: {avg_improvement:+.3f} ({(avg_improvement/avg_initial)*100:+.1f}%)")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"LLM Provider: {provider}")
        if model:
            print(f"Model: {model}")
    
    # Export results
    if export and results:
        export_aggregate_results(results, export)
    
    # Generate visualizations
    if visualize and results:
        print("\nGenerating aggregate visualizations...")
        viz_dir = os.path.join(os.path.dirname(export) if export else 'results', 'visualizations')
        generate_all_visualizations(results, viz_dir)
        print(f"Visualizations saved to: {viz_dir}")
    
    return results


def export_results(results: Dict[str, Any], export_path: str, full_evaluation: bool = True):
    """Export results to specified format."""
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    if export_path.endswith('.json'):
        # Export as JSON
        with open(export_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to: {export_path}")
        
    elif export_path.endswith('.csv') and full_evaluation:
        # Export metrics comparison as CSV
        calculator = BasicMetricsCalculator()
        
        # Convert back to MetricsResult objects
        from srlp_framework.core.metrics_calculator import MetricsResult
        
        metrics_before = MetricsResult(
            plan_metrics=results['metrics_before']['plan_metrics'],
            quality_metrics=results['metrics_before']['quality_metrics'],
            performance_metrics=results['metrics_before']['performance_metrics'],
            comparison_metrics=results['metrics_before']['comparison_metrics'],
            composite_scores=results['metrics_before']['composite_scores']
        )
        
        metrics_after = MetricsResult(
            plan_metrics=results['metrics_after']['plan_metrics'],
            quality_metrics=results['metrics_after']['quality_metrics'],
            performance_metrics=results['metrics_after']['performance_metrics'],
            comparison_metrics=results['metrics_after']['comparison_metrics'],
            composite_scores=results['metrics_after']['composite_scores']
        )
        
        calculator.export_comparison_to_csv(metrics_before, metrics_after, export_path)
        print(f"Metrics comparison exported to: {export_path}")
        
    else:
        # Default to JSON
        json_path = export_path.replace('.csv', '.json') if export_path.endswith('.csv') else export_path + '.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to: {json_path}")


def export_aggregate_results(results: List[Dict[str, Any]], export_path: str):
    """Export aggregate results from multiple evaluations."""
    
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    if export_path.endswith('.csv'):
        # Create summary CSV
        import csv
        
        summary_data = []
        for result in results:
            scenario = result['scenario']
            before_quality = result['metrics_before']['quality_metrics']['overall_quality_score']
            after_quality = result['metrics_after']['quality_metrics']['overall_quality_score']
            improvement = after_quality - before_quality
            converged = result['refinement_result']['converged']
            iterations = result['refinement_result']['iterations']
            time_taken = result['refinement_result']['total_time']
            llm_info = result.get('llm_info', {})
            
            summary_data.append({
                'scenario': scenario,
                'initial_quality': before_quality,
                'final_quality': after_quality,
                'improvement': improvement,
                'improvement_percent': (improvement / max(0.001, before_quality)) * 100,
                'converged': converged,
                'iterations': iterations,
                'time_seconds': time_taken,
                'llm_provider': llm_info.get('provider', 'unknown'),
                'llm_model': llm_info.get('model', 'unknown')
            })
        
        with open(export_path, 'w', newline='') as csvfile:
            fieldnames = ['scenario', 'initial_quality', 'final_quality', 'improvement', 
                         'improvement_percent', 'converged', 'iterations', 'time_seconds',
                         'llm_provider', 'llm_model']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)
            
    else:
        # Export as JSON
        aggregate_data = {
            'summary': {
                'total_scenarios': len(results),
                'avg_initial_quality': sum(r['metrics_before']['quality_metrics']['overall_quality_score'] 
                                         for r in results) / len(results),
                'avg_final_quality': sum(r['metrics_after']['quality_metrics']['overall_quality_score'] 
                                       for r in results) / len(results),
                'success_rate': sum(1 for r in results if r['refinement_result']['converged']) / len(results),
                'llm_provider': results[0].get('llm_info', {}).get('provider', 'unknown') if results else 'unknown'
            },
            'detailed_results': results
        }
        
        json_path = export_path.replace('.csv', '.json') if export_path.endswith('.csv') else export_path
        with open(json_path, 'w') as f:
            json.dump(aggregate_data, f, indent=2)
    
    print(f"Aggregate results exported to: {export_path}")


def main():
    """Main CLI entry point with LLM provider support."""
    parser = argparse.ArgumentParser(
        description='SRLP Framework - Self-Refinement for LLM Planners via Self-Checking Feedback',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run travel scenario with OpenAI GPT-4
  python -m srlp_framework.cli.main --scenario travel --provider openai --model gpt-4 --evaluate --export results.csv
  
  # Run with Claude
  python -m srlp_framework.cli.main --scenario cooking --provider claude --model claude-3-sonnet-20240229 --evaluate
  
  # Run with local LLaMA via Ollama
  python -m srlp_framework.cli.main --scenario project --provider llama --model llama2 --evaluate
  
  # Run all scenarios with HuggingFace
  python -m srlp_framework.cli.main --all --provider huggingface --model gpt2 --evaluate --export results.csv
  
  # List available providers
  python -m srlp_framework.cli.main --list-providers
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--problem-file', type=str,
                           help='JSON file containing problem specification')
    input_group.add_argument('--scenario', type=str, default='travel',
                           help='Predefined scenario to run (travel, cooking, project, etc.)')
    input_group.add_argument('--scenarios', nargs='+',
                           help='Multiple scenarios to run')
    input_group.add_argument('--all', action='store_true',
                           help='Run all available scenarios')
    
    # LLM Provider options
    llm_group = parser.add_argument_group('LLM Provider Options')
    llm_group.add_argument('--provider', type=str, default='mock',
                          help='LLM provider (openai, claude, llama, huggingface, mock)')
    llm_group.add_argument('--model', type=str,
                          help='Model name (uses provider default if not specified)')
    llm_group.add_argument('--api-key', type=str,
                          help='API key for real providers (optional - mock provider works without keys)')
    llm_group.add_argument('--base-url', type=str,
                          help='Base URL for the provider API')
    llm_group.add_argument('--temperature', type=float, default=0.7,
                          help='Temperature for text generation (default: 0.7)')
    llm_group.add_argument('--max-tokens', type=int, default=1000,
                          help='Maximum tokens to generate (default: 1000)')
    
    # Processing options
    parser.add_argument('--evaluate', action='store_true', default=True,
                       help='Run full evaluation with metrics (default: True)')
    parser.add_argument('--no-evaluate', dest='evaluate', action='store_false',
                       help='Skip full evaluation, run basic refinement only')
    
    # Output options
    parser.add_argument('--export', type=str,
                       help='Export results to file (CSV or JSON)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization charts')
    
    # Framework options
    parser.add_argument('--iterations', type=int, default=5,
                       help='Maximum refinement iterations (default: 5)')
    parser.add_argument('--quality-threshold', type=float, default=0.8,
                       help='Quality threshold for convergence (default: 0.8)')
    
    # Utility options
    parser.add_argument('--list-scenarios', action='store_true',
                       help='List all available scenarios')
    parser.add_argument('--list-providers', action='store_true',
                       help='List all available LLM providers')
    parser.add_argument('--test-provider', type=str,
                       help='Test connection to specified provider')
    parser.add_argument('--version', action='version', version='SRLP Framework 1.0.0')
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.list_scenarios:
        scenarios = get_all_test_scenarios()
        print("Available scenarios:")
        print("=" * 40)
        for scenario in scenarios:
            print(f"Name: {scenario['name']}")
            print(f"Type: {scenario['problem']['type']}")
            print(f"Goal: {scenario['problem']['goal']}")
            print("-" * 40)
        return
    
    if args.list_providers:
        providers = list_available_providers()
        print("Available LLM providers:")
        print("=" * 50)
        for provider_name, info in providers.items():
            print(f"Provider: {provider_name}")
            print(f"Description: {info['description']}")
            print(f"Default Model: {info['default_model']}")
            print(f"Requires API Key: {info['requires_api_key']}")
            if info.get('api_key_env_var'):
                print(f"API Key Environment Variable: {info['api_key_env_var']}")
            print("-" * 50)
        return
    
    if args.test_provider:
        print(f"Testing connection to {args.test_provider} provider...")
        try:
            llm_kwargs = {}
            if args.api_key:
                llm_kwargs['api_key'] = args.api_key
            if args.base_url:
                llm_kwargs['base_url'] = args.base_url
            
            success = LLMFactory.test_provider(args.test_provider, args.model, **llm_kwargs)
            if success:
                print(f"✅ Connection to {args.test_provider} successful!")
            else:
                print(f"❌ Connection to {args.test_provider} failed!")
        except Exception as e:
            print(f"❌ Error testing {args.test_provider}: {e}")
        return
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Prepare LLM configuration
    llm_kwargs = {
        'temperature': args.temperature,
        'max_tokens': args.max_tokens
    }
    
    if args.api_key:
        llm_kwargs['api_key'] = args.api_key
    if args.base_url:
        llm_kwargs['base_url'] = args.base_url
    
    try:
        if args.all:
            # Run all scenarios
            results = run_multiple_evaluations(
                export=args.export,
                visualize=args.visualize,
                provider=args.provider,
                model=args.model,
                **llm_kwargs
            )
            
        elif args.scenarios:
            # Run specified scenarios
            results = run_multiple_evaluations(
                scenarios=args.scenarios,
                export=args.export,
                visualize=args.visualize,
                provider=args.provider,
                model=args.model,
                **llm_kwargs
            )
            
        else:
            # Run single scenario
            result = run_single_evaluation(
                problem_file=args.problem_file,
                scenario=args.scenario,
                evaluate=args.evaluate,
                export=args.export,
                visualize=args.visualize,
                provider=args.provider,
                model=args.model,
                **llm_kwargs
            )
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_cli():
    """Alternative entry point for programmatic use."""
    main()


if __name__ == "__main__":
    main()

