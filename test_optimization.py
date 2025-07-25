"""
Test script for optimized NL2SQL system
Tests performance improvements and ensures no regression
"""

import asyncio
import time
import json
from typing import Dict, Any, List
from datetime import datetime
import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from main import NL2SQLMultiAgentSystem
from services.system_integrator import system_integrator
from services.performance_monitor import perf_monitor


class OptimizationTestSuite:
    """Test suite for optimization validation"""
    
    def __init__(self):
        self.test_queries = [
            "Show top 3 customers by revenue with their details in March 2025",
            "Analyze revenue by region and show which region performs best in 2025",
            "What about their contact information?",
            "List all products with inventory below 100 units",
            "Show quarterly sales performance for 2024",
            "Which customers have not placed orders in the last 6 months?",
            "Compare sales performance between regions in Q1 2024",
            "Show product categories with highest profit margins"
        ]
        
        self.test_results = {
            "legacy_results": [],
            "optimized_results": [],
            "performance_comparison": {},
            "regression_check": True,
            "optimization_benefits": {}
        }
    
    async def run_full_test_suite(self):
        """Run comprehensive test suite"""
        print("🧪 Starting Optimization Test Suite")
        print("=" * 50)
        
        # Initialize system
        system = await self._initialize_system()
        if not system:
            print("❌ Failed to initialize system")
            return False
        
        # Step 1: Enable migration mode for comparison
        print("\n📊 Step 1: Testing with Migration Mode (Both Legacy and Optimized)")
        system_integrator.enable_migration_mode()
        
        # Run tests in migration mode
        migration_results = await self._run_test_queries(system, "migration")
        
        # Step 2: Test optimized only
        print("\n🚀 Step 2: Testing Optimized Only")
        system_integrator.disable_migration_mode()
        
        optimized_results = await self._run_test_queries(system, "optimized")
        
        # Step 3: Performance analysis
        print("\n📈 Step 3: Performance Analysis")
        await self._analyze_performance()
        
        # Step 4: Regression check
        print("\n🔍 Step 4: Regression Check")
        regression_passed = await self._check_regression(migration_results, optimized_results)
        
        # Step 5: Generate report
        print("\n📋 Step 5: Generating Test Report")
        await self._generate_test_report()
        
        # Cleanup
        await system.close()
        
        print(f"\n{'✅' if regression_passed else '❌'} Test Suite Complete - {'PASSED' if regression_passed else 'FAILED'}")
        return regression_passed
    
    async def _initialize_system(self) -> NL2SQLMultiAgentSystem:
        """Initialize and integrate the system"""
        try:
            print("🔄 Initializing NL2SQL system...")
            system = await NL2SQLMultiAgentSystem.create_and_initialize()
            
            print("🔄 Integrating optimized components...")
            integration_success = await system_integrator.integrate_with_existing_system(system)
            
            if not integration_success:
                print("❌ Failed to integrate optimized components")
                return None
            
            print("✅ System initialized and integrated successfully")
            return system
            
        except Exception as e:
            print(f"❌ System initialization failed: {str(e)}")
            return None
    
    async def _run_test_queries(self, system: NL2SQLMultiAgentSystem, mode: str) -> List[Dict[str, Any]]:
        """Run test queries in specified mode"""
        results = []
        session_id = f"test_session_{mode}_{int(time.time())}"
        
        print(f"Running {len(self.test_queries)} test queries in {mode} mode...")
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"  {i}/{len(self.test_queries)}: {query[:50]}...")
            
            start_time = time.time()
            
            try:
                # Use previous responses as context for follow-up queries
                context = ""
                if results and "what about" in query.lower():
                    context = results[-1].get("response", "")[:500]  # Limit context size
                
                result = await system.orchestrator_agent.process(
                    user_input=query,
                    session_id=session_id,
                    conversation_context=context
                )
                
                execution_time = time.time() - start_time
                
                result_data = {
                    "query": query,
                    "execution_time": execution_time,
                    "success": result.get("success", False),
                    "response": result.get("response", "")[:200],  # Limit for readability
                    "sql_query": result.get("sql_query", ""),
                    "data_count": len(result.get("data", [])),
                    "orchestrator_type": result.get("orchestrator_type", "unknown"),
                    "workflow_type": result.get("workflow_type", "unknown"),
                    "cache_hit": result.get("cache_hit", False),
                    "error": result.get("error") if not result.get("success") else None
                }
                
                results.append(result_data)
                
                status = "✅" if result.get("success") else "❌"
                print(f"    {status} {execution_time:.2f}s - {result.get('orchestrator_type', 'unknown')}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"    ❌ {execution_time:.2f}s - Error: {str(e)[:100]}")
                
                results.append({
                    "query": query,
                    "execution_time": execution_time,
                    "success": False,
                    "error": str(e),
                    "orchestrator_type": "error"
                })
            
            # Small delay between queries
            await asyncio.sleep(1)
        
        return results
    
    async def _analyze_performance(self):
        """Analyze performance metrics"""
        print("Analyzing performance metrics...")
        
        # Get system performance comparison
        comparison = system_integrator.get_performance_comparison()
        self.test_results["performance_comparison"] = comparison
        
        # Get detailed performance metrics
        detailed_metrics = perf_monitor.get_summary()
        self.test_results["detailed_metrics"] = detailed_metrics
        
        # Print performance summary
        print(f"  Legacy executions: {comparison['legacy_executions']}")
        print(f"  Optimized executions: {comparison['optimized_executions']}")
        print(f"  Legacy avg time: {comparison['legacy_avg_time']}s")
        print(f"  Optimized avg time: {comparison['optimized_avg_time']}s")
        print(f"  Performance improvement: {comparison['performance_improvement_percent']}%")
        
        # Analyze optimization benefits
        benefits = {
            "time_improvement": comparison['performance_improvement_percent'],
            "parallel_workflow_capability": True,
            "caching_enabled": True,
            "performance_monitoring": True,
            "agent_pooling": True
        }
        
        self.test_results["optimization_benefits"] = benefits
    
    async def _check_regression(self, migration_results: List[Dict], optimized_results: List[Dict]) -> bool:
        """Check for regression in functionality"""
        print("Checking for regression...")
        
        regression_issues = []
        
        # Check success rates
        migration_success_rate = sum(1 for r in migration_results if r["success"]) / len(migration_results)
        optimized_success_rate = sum(1 for r in optimized_results if r["success"]) / len(optimized_results)
        
        if optimized_success_rate < migration_success_rate * 0.9:  # Allow 10% tolerance
            regression_issues.append(f"Success rate decreased: {optimized_success_rate:.1%} vs {migration_success_rate:.1%}")
        
        # Check that SQL queries are still generated
        migration_sql_count = sum(1 for r in migration_results if r.get("sql_query"))
        optimized_sql_count = sum(1 for r in optimized_results if r.get("sql_query"))
        
        if optimized_sql_count < migration_sql_count * 0.9:
            regression_issues.append(f"SQL generation decreased: {optimized_sql_count} vs {migration_sql_count}")
        
        # Check response quality (basic check - responses should have reasonable length)
        migration_avg_response_len = sum(len(r.get("response", "")) for r in migration_results if r["success"]) / max(1, sum(1 for r in migration_results if r["success"]))
        optimized_avg_response_len = sum(len(r.get("response", "")) for r in optimized_results if r["success"]) / max(1, sum(1 for r in optimized_results if r["success"]))
        
        if optimized_avg_response_len < migration_avg_response_len * 0.5:  # Allow significant variance
            regression_issues.append(f"Response quality potentially decreased: avg {optimized_avg_response_len:.0f} vs {migration_avg_response_len:.0f} chars")
        
        # Store regression results
        self.test_results["regression_check"] = len(regression_issues) == 0
        self.test_results["regression_issues"] = regression_issues
        
        if regression_issues:
            print("  ❌ Regression issues found:")
            for issue in regression_issues:
                print(f"    - {issue}")
        else:
            print("  ✅ No regression detected")
        
        return len(regression_issues) == 0
    
    async def _generate_test_report(self):
        """Generate comprehensive test report"""
        report_path = f"test_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Add timestamp and summary
        self.test_results["timestamp"] = datetime.now().isoformat()
        self.test_results["test_summary"] = {
            "total_queries": len(self.test_queries),
            "regression_passed": self.test_results["regression_check"],
            "performance_improvement": self.test_results["performance_comparison"].get("performance_improvement_percent", 0),
            "optimization_successful": (
                self.test_results["regression_check"] and 
                self.test_results["performance_comparison"].get("performance_improvement_percent", 0) > 0
            )
        }
        
        # Save report
        try:
            with open(report_path, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"  📋 Test report saved to: {report_path}")
        except Exception as e:
            print(f"  ⚠️ Failed to save test report: {str(e)}")
        
        # Print summary
        print("\n📊 Test Summary:")
        print(f"  Total queries tested: {self.test_results['test_summary']['total_queries']}")
        print(f"  Regression check: {'✅ PASSED' if self.test_results['regression_check'] else '❌ FAILED'}")
        print(f"  Performance improvement: {self.test_results['test_summary']['performance_improvement']:.1f}%")
        print(f"  Overall optimization: {'✅ SUCCESSFUL' if self.test_results['test_summary']['optimization_successful'] else '❌ NEEDS WORK'}")


async def run_quick_test():
    """Run a quick test with a few queries"""
    print("🧪 Running Quick Optimization Test")
    
    test_suite = OptimizationTestSuite()
    # Use only first 3 queries for quick test
    test_suite.test_queries = test_suite.test_queries[:3]
    
    success = await test_suite.run_full_test_suite()
    return success


async def run_full_test():
    """Run full test suite"""
    test_suite = OptimizationTestSuite()
    success = await test_suite.run_full_test_suite()
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test optimized NL2SQL system")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer queries")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    
    args = parser.parse_args()
    
    if args.quick:
        success = asyncio.run(run_quick_test())
    elif args.full:
        success = asyncio.run(run_full_test())
    else:
        print("Please specify --quick or --full")
        success = False
    
    sys.exit(0 if success else 1)
