#!/usr/bin/env python3
"""
Performance Optimization Test - Fixed Version
Tests current system performance and provides optimization recommendations
"""

import asyncio
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import NL2SQLMultiAgentSystem

async def test_current_performance():
    """Test current system performance and identify optimization opportunities"""
    
    print("🚀 Current System Performance Analysis")
    print("=" * 60)
    
    # Initialize system
    print("\n📦 Initializing system...")
    system = NL2SQLMultiAgentSystem()
    await system.initialize()
    
    # Test queries for performance analysis
    test_queries = [
        "Show me the top 5 customers by revenue in May 2025",
        "What are the best selling products in terms of volume? in May 2025", 
        "Analyze revenue by region and show which region performs best in 2025",
        "Show top 3 customers by revenue with their details in March 2025",
        "Show the top performing distribution centers (CEDIs) by total sales in 2025",
    ]
    
    # Performance test
    print("\n⏱️ Performance Test...")
    execution_times = []
    session_id = f"perf_test_{int(time.time())}"
    
    for i, query in enumerate(test_queries, 1):
        print(f"  Query {i}/5: {query[:50]}...")
        
        start_time = time.time()
        try:
            result = await system.orchestrator_agent.process({
                "question": query,
                "session_id": session_id,
                "user_id": f"perf_test_user",
                "execute": True,
                "limit": 10,
                "include_summary": True
            })
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            success = result.get('success', False)
            print(f"    📊 {execution_time:.2f}s - {'Success' if success else 'Failed'}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            print(f"    ❌ {execution_time:.2f}s - Error: {str(e)[:50]}...")
    
    avg_time = sum(execution_times) / len(execution_times)
    min_time = min(execution_times)
    max_time = max(execution_times)
    total_time = sum(execution_times)
    
    # Performance analysis
    print(f"\n📈 Performance Analysis:")
    print(f"  Average Time:  {avg_time:.2f}s")
    print(f"  Minimum Time:  {min_time:.2f}s")
    print(f"  Maximum Time:  {max_time:.2f}s")
    print(f"  Total Time:    {total_time:.2f}s")
    print(f"  Queries/min:   {60 / avg_time:.1f}")
    
    # System analysis
    print(f"\n🔍 System Analysis:")
    print(f"  Orchestrator Type: {type(system.orchestrator_agent).__name__}")
    print(f"  Memory Service:    {'Available' if system.memory_service else 'Not Available'}")
    print(f"  Schema Service:    {'Available' if system.schema_service else 'Not Available'}")
    print(f"  MCP Plugin:        {'Available' if system.mcp_plugin else 'Not Available'}")
    
    # Check for optimization components
    optimization_components = {
        "OptimizedOrchestratorAgent": False,
        "PerformanceMonitor": hasattr(system, 'perf_monitor'),
        "AgentInterface": False,
        "ParallelExecution": False
    }
    
    try:
        from src.agents.optimized_orchestrator_agent import OptimizedOrchestratorAgent
        optimization_components["OptimizedOrchestratorAgent"] = True
    except ImportError:
        pass
    
    try:
        from src.agents.agent_interface import AgentMessage
        optimization_components["AgentInterface"] = True
    except ImportError:
        pass
    
    # Check if current orchestrator has parallel execution
    if hasattr(system.orchestrator_agent, 'enable_parallel_execution'):
        optimization_components["ParallelExecution"] = True
    
    print(f"\n🔧 Optimization Components Status:")
    for component, available in optimization_components.items():
        status = "✅ Available" if available else "❌ Missing"
        print(f"  {component}: {status}")
    
    # Performance recommendations
    print(f"\n💡 Performance Recommendations:")
    
    if avg_time > 30:
        print("  🔴 CRITICAL: Response time too high (>30s)")
        print("    Priority Actions:")
        print("    1. Implement OptimizedOrchestratorAgent")
        print("    2. Enable parallel execution for context & schema analysis")
        print("    3. Implement query result caching")
        print("    4. Optimize database connection pooling")
    elif avg_time > 15:
        print("  🟡 WARNING: Response time high (>15s)")
        print("    Recommended Actions:")
        print("    1. Deploy optimized orchestrator")
        print("    2. Implement intelligent caching")
        print("    3. Add query optimization")
    elif avg_time > 5:
        print("  🟢 GOOD: Response time acceptable (5-15s)")
        print("    Enhancement Actions:")
        print("    1. Fine-tune existing optimizations")
        print("    2. Monitor and maintain performance")
        print("    3. Implement result streaming for large datasets")
    else:
        print("  🟢 EXCELLENT: Response time optimal (<5s)")
        print("    Maintenance Actions:")
        print("    1. Maintain current performance levels")
        print("    2. Focus on feature enhancements")
        print("    3. Implement advanced monitoring")
    
    # Next steps based on current state
    print(f"\n🎯 Immediate Next Steps:")
    
    if optimization_components["OptimizedOrchestratorAgent"]:
        print("  ✅ OptimizedOrchestratorAgent available")
        print("  📝 Next: Run test_optimization_comparison.py to compare performance")
        print("  🔧 Action: Integrate optimized orchestrator into main system")
    else:
        print("  ❌ OptimizedOrchestratorAgent not found")
        print("  📝 Next: Implement optimized orchestrator following the improvement plan")
        print("  🔧 Action: Create OptimizedOrchestratorAgent class")
    
    if optimization_components["AgentInterface"]:
        print("  ✅ Unified agent interface available")
        print("  🔧 Action: Ensure all agents use the unified interface")
    else:
        print("  ❌ Unified agent interface missing")
        print("  � Action: Implement agent communication interface")
    
    # Specific optimization targets
    print(f"\n🎯 Optimization Targets:")
    if avg_time > 10:
        target_improvement = 50  # 50% improvement
        target_time = avg_time * 0.5
        print(f"  Target Time: {target_time:.2f}s ({target_improvement}% improvement)")
        print(f"  Time to Save: {avg_time - target_time:.2f}s per query")
        print(f"  Daily Time Savings: {(avg_time - target_time) * 100:.0f}s for 100 queries")
    
    # Cleanup
    await system.close()
    
    print(f"\n🎉 Performance analysis completed!")
    return {
        "avg_time": avg_time,
        "total_time": total_time,
        "optimization_available": any(optimization_components.values()),
        "needs_optimization": avg_time > 15,
        "success": True
    }

if __name__ == "__main__":
    try:
        result = asyncio.run(test_current_performance())
        if result["needs_optimization"]:
            print(f"📝 Recommendation: Run test_optimization_comparison.py for detailed optimization testing")
        else:
            print(f"✅ System performance is adequate")
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        sys.exit(1)
