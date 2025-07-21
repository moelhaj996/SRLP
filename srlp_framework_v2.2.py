#!/usr/bin/env python3
"""SRLP Framework v2.2 - Self-Refinement for LLM Planners

This script implements the actionable suggestions for enhancing the Enhanced SRLP Framework v2.1:

🔧 Key Improvements in v2.2:
- Optimized Retry Logic: Exponential backoff with intelligent retry triggers
- Enhanced Provider Efficiency: Parallel processing and request optimization
- Realistic Mock Provider: Configurable latency simulation (1-5s)
- Scenario-Specific Tuning: Adaptive processing based on complexity
- Dynamic Improvement Scoring: Variable scoring based on accuracy/efficiency
- Auto-Generated Visualizations: Matplotlib charts for comprehensive analysis
- Capped Retry Limits: Maximum 2 attempts to prevent excessive delays
- Detailed Analytics: Provider efficiency metrics and bottleneck analysis

Author: AI Research Team
Version: 2.2
Date: 2025-07-12 19:00:00 CEST
"""

import os
import sys
import json
import time
import csv
import random
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set API keys for testing
# Load API keys from environment variables (DO NOT hardcode in production!)
# Set these in your environment or .env file
if not os.getenv('GOOGLE_API_KEY'):
    print("⚠️  Warning: GOOGLE_API_KEY not found in environment variables")
if not os.getenv('OPENAI_API_KEY'):
    print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
if not os.getenv('ANTHROPIC_API_KEY'):
    print("⚠️  Warning: ANTHROPIC_API_KEY not found in environment variables")

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from refinement_engine import RefinementEngine, create_refinement_engine
    from test_scenarios import get_all_test_scenarios
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure refinement_engine.py and test_scenarios.py are in the same directory.")
    sys.exit(1)

class EnhancedSRLPFrameworkV22:
    """Enhanced SRLP Framework v2.2 with optimized retry logic and provider efficiency."""
    
    def __init__(self):
        self.results = {
            "metadata": {
                "framework_version": "Enhanced v2.2",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
                "total_tests": 0,
                "successful_tests": 0,
                "failed_tests": 0,
                "total_duration": 0.0,
                "scenarios_tested": [],
                "configurations_tested": [],
                "providers_tested": []
            },
            "test_results": [],
            "provider_stats": {},
            "scenario_stats": {},
            "quota_monitoring": {},
            "retry_statistics": {
                "total_retries": 0,
                "retry_success_rate": 0.0,
                "avg_retry_duration": 0.0,
                "retry_reasons": {}
            },
            "efficiency_metrics": {
                "provider_efficiency": {},
                "scenario_complexity": {},
                "bottleneck_analysis": {}
            }
        }
        
        # Enhanced quota limits with safety margins
        self.quota_limits = {
            "gemini": {"daily_limit": 1000, "current_usage": 0, "warning_threshold": 0.8},
            "openai": {"daily_limit": 500, "current_usage": 0, "warning_threshold": 0.8},
            "claude": {"daily_limit": 300, "current_usage": 0, "warning_threshold": 0.8},
            "mock": {"daily_limit": float('inf'), "current_usage": 0, "warning_threshold": 1.0}
        }
        
        # Optimized retry settings
        self.retry_config = {
            "max_retries": 2,  # Capped at 2 attempts to prevent excessive delays
            "base_delay": 1.0,
            "max_delay": 8.0,
            "exponential_base": 2.0,
            "timeout_threshold": 15.0  # Increased from 10s for better tolerance
        }
        
        # Enhanced mock latency simulation
        self.mock_config = {
            "enabled": True,
            "scenario_latencies": {
                "travel_planning": {"min": 2.0, "max": 3.5},
                "cooking_dinner": {"min": 1.0, "max": 2.0},
                "software_project": {"min": 2.5, "max": 4.0},
                "conference_planning": {"min": 1.5, "max": 2.5},
                "kitchen_renovation": {"min": 3.0, "max": 5.0}  # Most complex
            },
            "variability_factor": 0.2  # ±20% random variation
        }
        
        # Scenario complexity factors for adaptive processing
        self.scenario_complexity = {
            "travel_planning": 0.8,
            "cooking_dinner": 0.4,
            "software_project": 0.9,
            "conference_planning": 0.6,
            "kitchen_renovation": 1.0  # Most complex
        }
        
        # Tracking
        self.retry_log = []
        self.network_log = []
        self.efficiency_log = []
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging for network and retry operations."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def calculate_dynamic_improvement_score(self, accuracy: float, efficiency: float, 
                                          iterations: int, scenario_complexity: float) -> float:
        """Calculate dynamic improvement score based on multiple factors."""
        # Base score from accuracy (0-0.4)
        accuracy_score = min(0.4, (100 - accuracy) / 100 * 0.4)
        
        # Efficiency score (0-0.3)
        efficiency_score = min(0.3, efficiency / 10.0 * 0.3)
        
        # Iteration efficiency (0-0.2)
        iteration_score = max(0, 0.2 - (iterations - 1) * 0.05)
        
        # Complexity bonus (0-0.1)
        complexity_bonus = scenario_complexity * 0.1
        
        total_score = accuracy_score + efficiency_score + iteration_score + complexity_bonus
        return min(1.0, max(0.0, total_score))
        
    def simulate_realistic_mock_latency(self, scenario: str) -> float:
        """Simulate realistic latency for mock providers based on scenario complexity."""
        if not self.mock_config["enabled"]:
            return 0.0
            
        scenario_config = self.mock_config["scenario_latencies"].get(
            scenario, {"min": 1.0, "max": 3.0}
        )
        
        base_latency = random.uniform(scenario_config["min"], scenario_config["max"])
        
        # Add variability
        variability = self.mock_config["variability_factor"]
        variation = random.uniform(-variability, variability)
        
        final_latency = base_latency * (1 + variation)
        return max(0.1, final_latency)  # Minimum 0.1s
        
    def analyze_retry_trigger(self, error_message: str, call_duration: float) -> str:
        """Analyze and categorize retry triggers for optimization."""
        error_lower = error_message.lower()
        
        if call_duration > self.retry_config["timeout_threshold"]:
            return "timeout_exceeded"
        elif "429" in error_message or "too many requests" in error_lower:
            return "rate_limited"
        elif "quota" in error_lower or "limit" in error_lower:
            return "quota_exceeded"
        elif "timeout" in error_lower or "504" in error_message:
            return "network_timeout"
        elif "500" in error_message or "internal server" in error_lower:
            return "server_error"
        elif "401" in error_message or "unauthorized" in error_lower:
            return "auth_error"
        else:
            return "unknown_error"
            
    def calculate_exponential_backoff(self, attempt: int, base_delay: float = None) -> float:
        """Calculate exponential backoff delay with jitter."""
        if base_delay is None:
            base_delay = self.retry_config["base_delay"]
            
        delay = min(
            self.retry_config["max_delay"],
            base_delay * (self.retry_config["exponential_base"] ** attempt)
        )
        
        # Add jitter (±25%)
        jitter = random.uniform(-0.25, 0.25)
        return delay * (1 + jitter)
        
    def optimized_api_call_with_retry(self, func, provider: str, scenario: str) -> Dict[str, Any]:
        """Optimized API call with intelligent retry logic and comprehensive logging."""
        start_time = time.perf_counter()
        total_retry_time = 0.0
        retry_reasons = []
        
        # Check quota before making call
        quota_ok, quota_msg = self.check_quota_status(provider)
        if not quota_ok:
            self.log_network_event(provider, "quota_exceeded", 0.0, quota_msg)
            raise Exception(f"Quota exceeded for {provider}: {quota_msg}")
        
        self.log_network_event(provider, "api_call_start", 0.0, f"Starting optimized API call for {scenario}")
        
        for attempt in range(self.retry_config["max_retries"] + 1):
            try:
                # Simulate realistic mock latency
                if provider == "mock" and self.mock_config["enabled"]:
                    mock_delay = self.simulate_realistic_mock_latency(scenario)
                    if mock_delay > 0:
                        time.sleep(mock_delay)
                        self.log_network_event(provider, "mock_latency_simulation", 
                                             mock_delay * 1000, f"Simulated {mock_delay:.3f}s realistic delay")
                
                call_start = time.perf_counter()
                result = func()
                call_duration = time.perf_counter() - call_start
                
                # Increment quota usage on successful call
                self.increment_quota_usage(provider)
                
                # Log successful call
                self.log_network_event(provider, "success", call_duration * 1000, 
                                     f"Call completed in {call_duration:.3f}s")
                
                # Check if call exceeded timeout threshold (but still succeeded)
                if call_duration > self.retry_config["timeout_threshold"]:
                    if attempt < self.retry_config["max_retries"]:
                        retry_reason = "timeout_exceeded"
                        retry_reasons.append(retry_reason)
                        
                        delay = self.calculate_exponential_backoff(attempt)
                        
                        retry_info = {
                            "timestamp": datetime.now().isoformat(),
                            "provider": provider,
                            "scenario": scenario,
                            "attempt": attempt + 1,
                            "call_duration": call_duration,
                            "threshold": self.retry_config["timeout_threshold"],
                            "retry_delay": delay,
                            "reason": retry_reason
                        }
                        self.retry_log.append(retry_info)
                        
                        self.log_network_event(provider, "retry_timeout", call_duration * 1000,
                                             f"Call took {call_duration:.3f}s (>{self.retry_config['timeout_threshold']}s), retrying in {delay:.1f}s")
                        
                        retry_start = time.perf_counter()
                        time.sleep(delay)
                        total_retry_time += (time.perf_counter() - retry_start)
                        continue
                
                # Successful call
                total_time = time.perf_counter() - start_time
                
                return {
                    "result": result,
                    "call_duration": call_duration,
                    "total_duration": total_time,
                    "retry_duration": total_retry_time,
                    "attempts": attempt + 1,
                    "success": True,
                    "retry_reasons": retry_reasons
                }
                
            except Exception as e:
                retry_reason = self.analyze_retry_trigger(str(e), 0.0)
                retry_reasons.append(retry_reason)
                
                self.log_network_event(provider, retry_reason, 0.0, str(e))
                
                if attempt < self.retry_config["max_retries"]:
                    delay = self.calculate_exponential_backoff(attempt)
                    
                    retry_info = {
                        "timestamp": datetime.now().isoformat(),
                        "provider": provider,
                        "scenario": scenario,
                        "attempt": attempt + 1,
                        "error": str(e),
                        "error_type": retry_reason,
                        "retry_delay": delay,
                        "reason": "api_error"
                    }
                    self.retry_log.append(retry_info)
                    
                    retry_start = time.perf_counter()
                    time.sleep(delay)
                    total_retry_time += (time.perf_counter() - retry_start)
                else:
                    # Max retries exceeded
                    total_time = time.perf_counter() - start_time
                    return {
                        "result": None,
                        "call_duration": 0.0,
                        "total_duration": total_time,
                        "retry_duration": total_retry_time,
                        "attempts": attempt + 1,
                        "success": False,
                        "error": str(e),
                        "error_type": retry_reason,
                        "retry_reasons": retry_reasons
                    }
        
    def check_quota_status(self, provider: str) -> Tuple[bool, str]:
        """Check if provider is within quota limits."""
        if provider not in self.quota_limits:
            return True, "Unknown provider"
            
        quota_info = self.quota_limits[provider]
        usage_percentage = quota_info["current_usage"] / quota_info["daily_limit"]
        
        if usage_percentage >= 1.0:
            return False, f"Quota exceeded: {quota_info['current_usage']}/{quota_info['daily_limit']} calls"
        elif usage_percentage >= quota_info["warning_threshold"]:
            warning_msg = f"Quota warning: {usage_percentage:.1%} used ({quota_info['current_usage']}/{quota_info['daily_limit']} calls)"
            return True, warning_msg
        
        return True, f"Quota OK: {usage_percentage:.1%} used"
        
    def increment_quota_usage(self, provider: str):
        """Increment quota usage for a provider."""
        if provider in self.quota_limits:
            self.quota_limits[provider]["current_usage"] += 1
            
    def log_network_event(self, provider: str, event_type: str, latency_ms: float, details: str):
        """Log network events with comprehensive details."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "event_type": event_type,
            "latency_ms": latency_ms,
            "details": details
        }
        self.network_log.append(event)
        
    def run_parallel_tests(self, test_configs: Dict, scenarios: Dict, max_workers: int = 3) -> List[Dict]:
        """Run tests in parallel for improved efficiency."""
        test_results = []
        
        # Create test tasks
        test_tasks = []
        for config_name, config in test_configs.items():
            for scenario_name in scenarios.keys():
                test_tasks.append({
                    "config_name": config_name,
                    "config": config,
                    "scenario_name": scenario_name,
                    "scenario_data": scenarios[scenario_name]['problem']
                })
        
        print(f"🚀 Running {len(test_tasks)} tests with {max_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.run_single_test, task): task 
                for task in test_tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    test_results.append(result)
                    
                    status = "✅" if result["success"] else "❌"
                    print(f"   {status} {task['config_name']} on {task['scenario_name']}")
                    
                except Exception as e:
                    print(f"   ❌ {task['config_name']} on {task['scenario_name']}: {str(e)}")
                    test_results.append({
                        "test_id": f"{task['config_name']}_{task['scenario_name']}",
                        "config_name": task['config_name'],
                        "scenario_name": task['scenario_name'],
                        "provider": task['config']['provider'],
                        "success": False,
                        "error": str(e),
                        "error_type": "execution_error"
                    })
        
        return test_results
        
    def run_single_test(self, task: Dict) -> Dict:
        """Run a single test with enhanced metrics."""
        config_name = task["config_name"]
        config = task["config"]
        scenario_name = task["scenario_name"]
        scenario_data = task["scenario_data"]
        
        provider_name = config["provider"]
        provider_key = provider_name.lower()
        
        # Check quota
        quota_ok, quota_msg = self.check_quota_status(provider_key)
        if not quota_ok:
            return {
                "test_id": f"{config_name}_{scenario_name}",
                "config_name": config_name,
                "scenario_name": scenario_name,
                "provider": provider_name,
                "success": False,
                "error": quota_msg,
                "error_type": "quota_exceeded"
            }
        
        try:
            # Initialize refinement engine
            model_name = config.get("model", None)
            engine = create_refinement_engine(
                provider=provider_name,
                model=model_name,
                max_iterations=config["max_iterations"]
            )
            engine.quality_threshold = config["threshold"]
            
            # Measure wall clock time
            wall_start = time.perf_counter()
            
            # Enhanced API call with optimized retry logic
            def run_refinement():
                return engine.refine_plan(scenario_data)
            
            api_result = self.optimized_api_call_with_retry(
                run_refinement, provider_key, scenario_name
            )
            
            wall_end = time.perf_counter()
            wall_clock_time = wall_end - wall_start
            
            if api_result["success"]:
                result = api_result["result"]
                
                # Calculate enhanced timing metrics
                framework_time = getattr(result, 'total_time', 0.0)
                api_time = getattr(result, 'timing_breakdown', {}).get('total_api_time', 0.0)
                overhead_time = framework_time - api_time
                retry_time = api_result["retry_duration"]
                
                # Calculate retry-adjusted accuracy
                adjusted_wall_time = wall_clock_time - retry_time
                if adjusted_wall_time > 0:
                    time_accuracy = abs(framework_time - adjusted_wall_time) / adjusted_wall_time * 100
                else:
                    time_accuracy = 0.0
                
                # Calculate efficiency metric
                efficiency = framework_time / wall_clock_time if wall_clock_time > 0 else 1.0
                
                # Calculate dynamic improvement score
                scenario_complexity = self.scenario_complexity.get(scenario_name, 0.5)
                iterations = getattr(result, 'iterations', 0)
                improvement_score = self.calculate_dynamic_improvement_score(
                    time_accuracy, efficiency, iterations, scenario_complexity
                )
                
                # Log efficiency metrics
                self.efficiency_log.append({
                    "provider": provider_key,
                    "scenario": scenario_name,
                    "efficiency": efficiency,
                    "wall_time": wall_clock_time,
                    "framework_time": framework_time,
                    "retry_time": retry_time,
                    "improvement_score": improvement_score
                })
                
                return {
                    "test_id": f"{config_name}_{scenario_name}",
                    "config_name": config_name,
                    "scenario_name": scenario_name,
                    "provider": provider_name,
                    "success": True,
                    "framework_time": framework_time,
                    "wall_clock_time": wall_clock_time,
                    "adjusted_wall_time": adjusted_wall_time,
                    "retry_time": retry_time,
                    "time_accuracy": time_accuracy,
                    "accuracy_display": f"{time_accuracy:.2f}%",
                    "efficiency": efficiency,
                    "total_actual_time": wall_clock_time,
                    "total_api_time": api_time,
                    "framework_overhead_time": overhead_time,
                    "improvement_score": improvement_score,
                    "converged": getattr(result, 'converged', False),
                    "iterations": iterations,
                    "retry_attempts": api_result["attempts"],
                    "retry_reasons": api_result.get("retry_reasons", []),
                    "quota_usage": self.quota_limits[provider_key]["current_usage"],
                    "scenario_complexity": scenario_complexity
                }
            else:
                return {
                    "test_id": f"{config_name}_{scenario_name}",
                    "config_name": config_name,
                    "scenario_name": scenario_name,
                    "provider": provider_name,
                    "success": False,
                    "error": api_result.get("error", "Unknown error"),
                    "error_type": api_result.get("error_type", "unknown"),
                    "retry_attempts": api_result["attempts"],
                    "retry_time": api_result["retry_duration"],
                    "retry_reasons": api_result.get("retry_reasons", [])
                }
                
        except Exception as e:
            return {
                "test_id": f"{config_name}_{scenario_name}",
                "config_name": config_name,
                "scenario_name": scenario_name,
                "provider": provider_name,
                "success": False,
                "error": str(e),
                "error_type": "execution_error",
                "retry_attempts": 1
            }
    
    def run_comprehensive_test(self, enable_parallel: bool = True, enable_mock_latency: bool = True) -> Dict[str, Any]:
        """Run comprehensive framework test with all enhancements."""
        print("🚀 Starting Enhanced SRLP Framework v2.2 Comprehensive Test")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print("🔧 Enhancements: Optimized Retry Logic, Provider Efficiency, Realistic Mock Latency")
        print("=" * 80)
        
        # Enable mock latency simulation
        self.mock_config["enabled"] = enable_mock_latency
        if enable_mock_latency:
            print("🎭 Realistic mock latency simulation enabled (scenario-specific delays)")
        
        # Get test scenarios
        scenario_list = get_all_test_scenarios()
        scenarios = {scenario['name']: scenario for scenario in scenario_list}
        
        print(f"📋 Loaded {len(scenarios)} test scenarios")
        for scenario_name in scenarios.keys():
            complexity = self.scenario_complexity.get(scenario_name, 0.5)
            print(f"   • {scenario_name} (complexity: {complexity:.1f})")
        
        # Define enhanced test configurations
        test_configs = {
            "Mock-Realistic": {"provider": "mock", "max_iterations": 2, "threshold": 0.9},
            "Mock-Extended": {"provider": "mock", "max_iterations": 3, "threshold": 0.7},
            "Gemini-Optimized": {"provider": "gemini", "model": "gemini-1.5-flash", "max_iterations": 2, "threshold": 0.85},
            "Gemini-Adaptive": {"provider": "gemini", "model": "gemini-1.5-flash", "max_iterations": 3, "threshold": 0.75},
            "OpenAI-Efficient": {"provider": "openai", "model": "gpt-4o-mini", "max_iterations": 2, "threshold": 0.8},
            "OpenAI-Robust": {"provider": "openai", "model": "gpt-4o-mini", "max_iterations": 3, "threshold": 0.75},
            "Claude-Fast": {"provider": "claude", "model": "claude-3-haiku-20240307", "max_iterations": 2, "threshold": 0.85},
            "Claude-Thorough": {"provider": "claude", "model": "claude-3-haiku-20240307", "max_iterations": 3, "threshold": 0.7},
        }
        
        print(f"⚙️  Testing {len(test_configs)} optimized configurations")
        
        start_time = time.perf_counter()
        
        # Run tests (parallel or sequential)
        if enable_parallel:
            test_results = self.run_parallel_tests(test_configs, scenarios, max_workers=3)
        else:
            test_results = []
            total_tests = len(test_configs) * len(scenarios)
            current_test = 0
            
            for config_name, config in test_configs.items():
                for scenario_name in scenarios.keys():
                    current_test += 1
                    print(f"\n[{current_test}/{total_tests}] Testing {config_name} on {scenario_name}")
                    
                    task = {
                        "config_name": config_name,
                        "config": config,
                        "scenario_name": scenario_name,
                        "scenario_data": scenarios[scenario_name]['problem']
                    }
                    
                    result = self.run_single_test(task)
                    test_results.append(result)
                    
                    status = "✅" if result["success"] else "❌"
                    if result["success"]:
                        print(f"   {status} Success: {result.get('accuracy_display', 'N/A')} accuracy, {result.get('improvement_score', 0.0):.3f} improvement")
                    else:
                        print(f"   {status} Failed: {result.get('error', 'Unknown error')}")
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        
        # Store results
        self.results["test_results"] = test_results
        
        # Calculate final statistics
        successful_tests = sum(1 for result in test_results if result["success"])
        failed_tests = len(test_results) - successful_tests
        success_rate = (successful_tests / len(test_results) * 100) if test_results else 0
        
        # Update metadata
        self.results["metadata"].update({
            "total_tests": len(test_results),
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "total_duration": total_duration,
            "scenarios_tested": list(set(result["scenario_name"] for result in test_results)),
            "configurations_tested": list(set(result["config_name"] for result in test_results)),
            "providers_tested": list(set(result["provider"] for result in test_results))
        })
        
        # Calculate enhanced statistics
        self.calculate_enhanced_statistics()
        
        # Generate visualization data
        visualization_data = self.prepare_visualization_data()
        
        # Save results
        self.save_enhanced_results(visualization_data)
        
        # Generate visualizations
        self.generate_visualizations(visualization_data)
        
        # Print summary
        self.print_enhanced_summary()
        
        return self.results
    
    def calculate_enhanced_statistics(self):
        """Calculate comprehensive provider and scenario statistics with efficiency metrics."""
        # Provider statistics
        provider_data = {}
        scenario_data = {}
        retry_reasons = {}
        
        for result in self.results["test_results"]:
            if not result["success"]:
                continue
                
            provider = result["provider"]
            scenario = result["scenario_name"]
            
            # Provider stats
            if provider not in provider_data:
                provider_data[provider] = {
                    "tests": 0,
                    "scenarios": set(),
                    "total_time": 0.0,
                    "total_wall_time": 0.0,
                    "total_retry_time": 0.0,
                    "total_improvement": 0.0,
                    "accuracy_errors": [],
                    "efficiency_scores": [],
                    "retry_attempts": 0
                }
            
            provider_data[provider]["tests"] += 1
            provider_data[provider]["scenarios"].add(scenario)
            provider_data[provider]["total_time"] += result.get("framework_time", 0.0)
            provider_data[provider]["total_wall_time"] += result.get("wall_clock_time", 0.0)
            provider_data[provider]["total_retry_time"] += result.get("retry_time", 0.0)
            provider_data[provider]["total_improvement"] += result.get('improvement_score', 0.0)
            provider_data[provider]["accuracy_errors"].append(result.get("time_accuracy", 0.0))
            provider_data[provider]["efficiency_scores"].append(result.get("efficiency", 1.0))
            provider_data[provider]["retry_attempts"] += result.get("retry_attempts", 1) - 1
            
            # Track retry reasons
            for reason in result.get("retry_reasons", []):
                retry_reasons[reason] = retry_reasons.get(reason, 0) + 1
            
            # Scenario stats
            if scenario not in scenario_data:
                scenario_data[scenario] = {
                    "tests": 0,
                    "providers": set(),
                    "total_time": 0.0,
                    "total_improvement": 0.0,
                    "complexity": self.scenario_complexity.get(scenario, 0.5)
                }
            
            scenario_data[scenario]["tests"] += 1
            scenario_data[scenario]["providers"].add(provider)
            scenario_data[scenario]["total_time"] += result.get("framework_time", 0.0)
            scenario_data[scenario]["total_improvement"] += result.get('improvement_score', 0.0)
        
        # Convert to final format with efficiency metrics
        for provider, data in provider_data.items():
            avg_efficiency = sum(data["efficiency_scores"]) / len(data["efficiency_scores"]) if data["efficiency_scores"] else 1.0
            avg_wall_time = data["total_wall_time"] / data["tests"] if data["tests"] > 0 else 0.0
            
            self.results["provider_stats"][provider] = {
                "tests": data["tests"],
                "scenarios": len(data["scenarios"]),
                "avg_time": data["total_time"] / data["tests"] if data["tests"] > 0 else 0.0,
                "avg_wall_time": avg_wall_time,
                "avg_retry_time": data["total_retry_time"] / data["tests"] if data["tests"] > 0 else 0.0,
                "avg_improvement": data["total_improvement"] / data["tests"] if data["tests"] > 0 else 0.0,
                "avg_accuracy_error": sum(data["accuracy_errors"]) / len(data["accuracy_errors"]) if data["accuracy_errors"] else 0.0,
                "avg_efficiency": avg_efficiency,
                "total_retries": data["retry_attempts"],
                "scenarios_tested": list(data["scenarios"])
            }
        
        for scenario, data in scenario_data.items():
            self.results["scenario_stats"][scenario] = {
                "tests": data["tests"],
                "providers": len(data["providers"]),
                "avg_time": data["total_time"] / data["tests"] if data["tests"] > 0 else 0.0,
                "avg_improvement": data["total_improvement"] / data["tests"] if data["tests"] > 0 else 0.0,
                "complexity": data["complexity"]
            }
        
        # Enhanced retry statistics
        if self.retry_log:
            total_retries = len(self.retry_log)
            successful_retries = sum(1 for retry in self.retry_log if "success" in retry.get("reason", ""))
            avg_retry_duration = sum(retry.get("retry_delay", 0.0) for retry in self.retry_log) / total_retries
            
            self.results["retry_statistics"] = {
                "total_retries": total_retries,
                "retry_success_rate": (successful_retries / total_retries * 100) if total_retries > 0 else 0.0,
                "avg_retry_duration": avg_retry_duration,
                "retry_reasons": retry_reasons
            }
        
        # Efficiency metrics
        if self.efficiency_log:
            provider_efficiency = {}
            scenario_complexity_analysis = {}
            
            for entry in self.efficiency_log:
                provider = entry["provider"]
                scenario = entry["scenario"]
                
                if provider not in provider_efficiency:
                    provider_efficiency[provider] = []
                provider_efficiency[provider].append(entry["efficiency"])
                
                if scenario not in scenario_complexity_analysis:
                    scenario_complexity_analysis[scenario] = []
                scenario_complexity_analysis[scenario].append(entry["wall_time"])
            
            # Calculate efficiency metrics
            for provider, efficiencies in provider_efficiency.items():
                avg_efficiency = sum(efficiencies) / len(efficiencies)
                self.results["efficiency_metrics"]["provider_efficiency"][provider] = {
                    "avg_efficiency": avg_efficiency,
                    "efficiency_variance": sum((e - avg_efficiency) ** 2 for e in efficiencies) / len(efficiencies)
                }
            
            # Calculate scenario complexity analysis
            for scenario, times in scenario_complexity_analysis.items():
                avg_time = sum(times) / len(times)
                self.results["efficiency_metrics"]["scenario_complexity"][scenario] = {
                    "avg_execution_time": avg_time,
                    "complexity_factor": self.scenario_complexity.get(scenario, 0.5),
                    "efficiency_ratio": avg_time / self.scenario_complexity.get(scenario, 0.5)
                }
    
    def prepare_visualization_data(self) -> List[Dict]:
        """Prepare enhanced data for visualization."""
        visualization_data = []
        
        for result in self.results["test_results"]:
            if result["success"]:
                visualization_data.append({
                    "Provider": result["provider"],
                    "Scenario": result["scenario_name"],
                    "Config": result["config_name"],
                    "Wall_Clock_Time": result.get("wall_clock_time", 0.0),
                    "Framework_Time": result.get("framework_time", 0.0),
                    "Adjusted_Wall_Time": result.get("adjusted_wall_time", 0.0),
                    "Retry_Time": result.get("retry_time", 0.0),
                    "Improvement_Score": result.get("improvement_score", 0.0),
                    "Efficiency": result.get("efficiency", 1.0),
                    "Iterations": result.get("iterations", 0),
                    "Retry_Attempts": result.get("retry_attempts", 1),
                    "Scenario_Complexity": result.get("scenario_complexity", 0.5)
                })
        
        return visualization_data
    
    def save_enhanced_results(self, visualization_data: List[Dict]):
        """Save comprehensive results to multiple files."""
        # Save main results
        with open("enhanced_framework_results_v2.2.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save enhanced visualization data
        with open("enhanced_visualization_data_v2.2.csv", "w", newline="") as f:
            if visualization_data:
                writer = csv.DictWriter(f, fieldnames=visualization_data[0].keys())
                writer.writeheader()
                writer.writerows(visualization_data)
        
        # Save detailed retry log
        if self.retry_log:
            with open("enhanced_retry_log_v2.2.txt", "w") as f:
                f.write("Enhanced SRLP Framework v2.2 - Optimized Retry Log\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
                
                for retry in self.retry_log:
                    f.write(f"Timestamp: {retry['timestamp']}\n")
                    f.write(f"Provider: {retry['provider']}\n")
                    f.write(f"Scenario: {retry['scenario']}\n")
                    f.write(f"Attempt: {retry['attempt']}\n")
                    f.write(f"Reason: {retry['reason']}\n")
                    if 'call_duration' in retry:
                        f.write(f"Call Duration: {retry['call_duration']:.3f}s\n")
                    if 'error' in retry:
                        f.write(f"Error: {retry['error']}\n")
                    f.write(f"Retry Delay: {retry['retry_delay']:.1f}s\n")
                    f.write("-" * 40 + "\n")
        
        # Save enhanced network log
        with open("enhanced_network_log_v2.2.txt", "w") as f:
            f.write(f"Enhanced SRLP Framework v2.2 - Network Log\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n")
            
            for event in self.network_log:
                f.write(f"[{event['timestamp']}] {event['provider']} - {event['event_type']} - ")
                f.write(f"{event['latency_ms']:.2f}ms - {event['details']}\n")
    
    def generate_visualizations(self, visualization_data: List[Dict]):
        """Generate comprehensive visualizations using matplotlib."""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            
            if not visualization_data:
                print("⚠️  No visualization data available")
                return
            
            df = pd.DataFrame(visualization_data)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Enhanced SRLP Framework v2.2 - Performance Analysis', fontsize=16, fontweight='bold')
            
            # 1. Provider Performance Comparison
            provider_stats = df.groupby('Provider').agg({
                'Wall_Clock_Time': 'mean',
                'Retry_Time': 'mean',
                'Improvement_Score': 'mean',
                'Efficiency': 'mean'
            }).round(3)
            
            ax1 = axes[0, 0]
            provider_stats['Wall_Clock_Time'].plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('Average Wall Clock Time by Provider')
            ax1.set_ylabel('Time (seconds)')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Scenario Complexity Analysis
            scenario_stats = df.groupby('Scenario').agg({
                'Wall_Clock_Time': 'mean',
                'Scenario_Complexity': 'first',
                'Improvement_Score': 'mean'
            }).round(3)
            
            ax2 = axes[0, 1]
            scatter = ax2.scatter(scenario_stats['Scenario_Complexity'], 
                                scenario_stats['Wall_Clock_Time'],
                                s=scenario_stats['Improvement_Score']*500,
                                alpha=0.6, c=range(len(scenario_stats)), cmap='viridis')
            ax2.set_title('Scenario Complexity vs Execution Time')
            ax2.set_xlabel('Complexity Factor')
            ax2.set_ylabel('Average Time (seconds)')
            
            # Add scenario labels
            for i, (scenario, row) in enumerate(scenario_stats.iterrows()):
                ax2.annotate(scenario.replace('_', '\n'), 
                           (row['Scenario_Complexity'], row['Wall_Clock_Time']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # 3. Retry Analysis
            retry_stats = df.groupby('Provider')['Retry_Time'].sum()
            
            ax3 = axes[1, 0]
            retry_stats.plot(kind='pie', ax=ax3, autopct='%1.1f%%')
            ax3.set_title('Total Retry Time Distribution by Provider')
            ax3.set_ylabel('')
            
            # 4. Efficiency vs Improvement Score
            ax4 = axes[1, 1]
            for provider in df['Provider'].unique():
                provider_data = df[df['Provider'] == provider]
                ax4.scatter(provider_data['Efficiency'], provider_data['Improvement_Score'], 
                          label=provider, alpha=0.7, s=60)
            
            ax4.set_title('Efficiency vs Improvement Score')
            ax4.set_xlabel('Efficiency Ratio')
            ax4.set_ylabel('Improvement Score')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('enhanced_framework_analysis_v2.2.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate summary statistics chart
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Provider comparison heatmap
            metrics = ['Wall_Clock_Time', 'Retry_Time', 'Improvement_Score', 'Efficiency']
            heatmap_data = df.groupby('Provider')[metrics].mean()
            
            # Normalize data for heatmap
            normalized_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
            
            im = ax.imshow(normalized_data.T, cmap='RdYlGn', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(range(len(heatmap_data.index)))
            ax.set_yticks(range(len(metrics)))
            ax.set_xticklabels(heatmap_data.index)
            ax.set_yticklabels(metrics)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Normalized Performance (0=worst, 1=best)')
            
            # Add text annotations
            for i in range(len(metrics)):
                for j in range(len(heatmap_data.index)):
                    text = ax.text(j, i, f'{heatmap_data.iloc[j, i]:.3f}',
                                 ha="center", va="center", color="black", fontweight='bold')
            
            ax.set_title('Provider Performance Heatmap - Enhanced Framework v2.2')
            plt.tight_layout()
            plt.savefig('provider_performance_heatmap_v2.2.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Generated visualizations:")
            print("   • enhanced_framework_analysis_v2.2.png - Comprehensive performance analysis")
            print("   • provider_performance_heatmap_v2.2.png - Provider comparison heatmap")
            
        except ImportError:
            print("⚠️  Matplotlib not available - skipping visualization generation")
            print("   Install with: pip install matplotlib pandas")
        except Exception as e:
            print(f"⚠️  Error generating visualizations: {e}")
    
    def print_enhanced_summary(self):
        """Print comprehensive test summary with enhanced metrics."""
        print("\n" + "=" * 80)
        print("🎯 ENHANCED SRLP FRAMEWORK v2.2 - OPTIMIZED PERFORMANCE RESULTS")
        print("=" * 80)
        
        metadata = self.results["metadata"]
        print(f"📊 Overall Results:")
        print(f"   • Success Rate: {metadata['success_rate']:.1f}% ({metadata['successful_tests']}/{metadata['total_tests']} tests)")
        print(f"   • Total Duration: {metadata['total_duration']:.2f} seconds")
        print(f"   • Framework Version: {metadata['framework_version']}")
        print(f"   • Scenarios Tested: {len(metadata['scenarios_tested'])}/{len(metadata['scenarios_tested'])}")
        
        print(f"\n📈 Provider Performance (Optimized):")
        print(f"{'Provider':<15} {'Tests':<6} {'Avg Time':<10} {'Retry Time':<11} {'Efficiency':<11} {'Improvement':<12}")
        print("-" * 80)
        
        for provider, stats in self.results["provider_stats"].items():
            print(f"{provider:<15} {stats['tests']:<6} {stats['avg_time']:<10.3f} "
                  f"{stats['avg_retry_time']:<11.3f} {stats['avg_efficiency']:<11.3f} {stats['avg_improvement']:<12.3f}")
        
        print(f"\n🎯 Scenario Analysis (Complexity-Aware):")
        print(f"{'Scenario':<20} {'Tests':<6} {'Complexity':<11} {'Avg Time':<10} {'Improvement':<12}")
        print("-" * 75)
        
        for scenario, stats in self.results["scenario_stats"].items():
            scenario_short = scenario.replace("_", " ").title()[:18]
            print(f"{scenario_short:<20} {stats['tests']:<6} {stats['complexity']:<11.1f} "
                  f"{stats['avg_time']:<10.3f} {stats['avg_improvement']:<12.3f}")
        
        print(f"\n🔄 Enhanced Retry Statistics:")
        retry_stats = self.results["retry_statistics"]
        if retry_stats["total_retries"] > 0:
            print(f"   • Total Retries: {retry_stats['total_retries']} (max 2 per test)")
            print(f"   • Success Rate: {retry_stats['retry_success_rate']:.1f}%")
            print(f"   • Avg Retry Duration: {retry_stats['avg_retry_duration']:.2f}s")
            
            if retry_stats.get("retry_reasons"):
                print(f"   • Retry Reasons:")
                for reason, count in retry_stats["retry_reasons"].items():
                    print(f"     - {reason}: {count} times")
        else:
            print(f"   • No retries needed - excellent performance!")
        
        print(f"\n⚡ Efficiency Metrics:")
        if "provider_efficiency" in self.results["efficiency_metrics"]:
            for provider, metrics in self.results["efficiency_metrics"]["provider_efficiency"].items():
                print(f"   • {provider}: {metrics['avg_efficiency']:.3f} avg efficiency")
        
        print(f"\n📁 Generated Files:")
        print(f"   • enhanced_framework_results_v2.2.json - Complete optimized results")
        print(f"   • enhanced_visualization_data_v2.2.csv - Enhanced chart data")
        print(f"   • enhanced_retry_log_v2.2.txt - Detailed retry analysis")
        print(f"   • enhanced_network_log_v2.2.txt - Network performance tracking")
        print(f"   • enhanced_framework_analysis_v2.2.png - Performance visualizations")
        print(f"   • provider_performance_heatmap_v2.2.png - Provider comparison")
        
        print("\n🚀 Framework Status: ENHANCED v2.2 - OPTIMIZED & EFFICIENT")
        print("🔧 Key Improvements: Capped Retries, Realistic Mock Latency, Dynamic Scoring")
        print("=" * 80)

def main():
    """Main execution function."""
    framework = EnhancedSRLPFrameworkV22()
    
    print("🎯 Enhanced SRLP Framework v2.2 - Optimization Demo")
    print("🔧 Implementing actionable suggestions for performance enhancement")
    print("\nEnhancements:")
    print("   ✅ Optimized retry logic with exponential backoff (max 2 attempts)")
    print("   ✅ Realistic mock provider latency simulation (1-5s)")
    print("   ✅ Dynamic improvement scoring based on accuracy & efficiency")
    print("   ✅ Scenario-specific complexity tuning")
    print("   ✅ Parallel processing for improved efficiency")
    print("   ✅ Auto-generated performance visualizations")
    print("   ✅ Comprehensive bottleneck analysis")
    
    # Run comprehensive test with all enhancements
    results = framework.run_comprehensive_test(
        enable_parallel=True,
        enable_mock_latency=True
    )
    
    print("\n🎉 Enhanced SRLP Framework v2.2 testing completed successfully!")
    print("📊 Check the generated files for detailed analysis and visualizations.")
    
    return results

if __name__ == "__main__":
    main()