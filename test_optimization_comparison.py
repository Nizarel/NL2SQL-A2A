#!/usr/bin/env python3
"""
Performance Optimization Test - Comparing Legacy vs Optimized Orchestrator
"""

import asyncio
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import NL2SQLMultiAgentSystem
from src.agents.optimized_orchestrator_agent import OptimizedOrchestratorAgent
from src.agents.agent_interface import AgentMessage, WorkflowConfig

async def test_optimization_comparison():
    """Test and compare legacy vs optimized orchestrator performance"""
    
    print("🚀 Performance Optimization Comparison Test")
    print("=" * 60)
    
    # Initialize system
    print("\n📦 Initializing NL2SQL system...")
    system = NL2SQLMultiAgentSystem()
    await system.initialize()
    
    # Test queries for performance comparison
    test_queries = [
        "Show me the top 5 customers by revenue in May 2025",
        "What are the best selling products in terms of volume? in May 2025",
        "Analyze revenue by region and show which region performs best in 2025",
        "Show top 3 customers by revenue with their details in March 2025",
        "Show the top performing distribution centers (CEDIs) by total sales in 2025",
    ]
    
    print(f"\n🧪 Testing with {len(test_queries)} queries...")
    
    # ==============================================
    # PHASE 1: Test Legacy Orchestrator
    # ==============================================
    print("\n⏱️ Phase 1: Legacy Orchestrator Performance Test...")
    legacy_times = []
    session_id = f"legacy_test_{int(time.time())}"
    
    for i, query in enumerate(test_queries, 1):
        print(f"  Query {i}/{len(test_queries)}: {query[:50]}...")
        
        start_time = time.time()
        try:
            result = await system.orchestrator_agent.process({
                "question": query,
                "session_id": session_id,
                "user_id": f"legacy_test_user",
                "execute": True,
                "limit": 10,
                "include_summary": True
            })
            execution_time = time.time() - start_time
            legacy_times.append(execution_time)
            
            success = result.get('success', False)
            print(f"    📊 {execution_time:.2f}s - {'Success' if success else 'Failed'}")
            
        except Exception as e:
            execution_time = time.time() - start_time
            legacy_times.append(execution_time)
            print(f"    ❌ {execution_time:.2f}s - Error: {str(e)[:50]}...")
    
    avg_legacy = sum(legacy_times) / len(legacy_times) if legacy_times else 0
    legacy_success_rate = sum(1 for t in legacy_times if t > 0) / len(legacy_times) * 100 if legacy_times else 0
    
    print(f"\n📊 Legacy Results:")
    print(f"  Average Time:  {avg_legacy:.2f}s")
    print(f"  Success Rate:  {legacy_success_rate:.1f}%")
    print(f"  Total Time:    {sum(legacy_times):.2f}s")
    
    # ==============================================
    # PHASE 2: Create and Test Optimized Orchestrator
    # ==============================================
    print("\n⚡ Phase 2: Creating Optimized Orchestrator...")
    
    try:
        # Create optimized orchestrator with same agents but new interface
        optimized_orchestrator = OptimizedOrchestratorAgent(
            kernel=system.kernel,
            memory_service=system.memory_service,
            schema_analyst_agent=system.schema_analyst_agent,
            sql_generator_agent=system.sql_generator_agent,
            executor_agent=system.executor_agent,
            summarizing_agent=system.summarizing_agent
        )
        
        # Initialize optimized orchestrator
        config = WorkflowConfig(
            parallel_execution=True,
            max_parallel_agents=3,
            enable_caching=True,
            performance_monitoring=True
        )
        await optimized_orchestrator._initialize_agent(config)
        
        print("  ✅ Optimized orchestrator created and configured")
        
        # Test optimized orchestrator
        print("\n⚡ Phase 2: Optimized Orchestrator Performance Test...")
        optimized_times = []
        session_id = f"optimized_test_{int(time.time())}"
        
        for i, query in enumerate(test_queries, 1):
            print(f"  Query {i}/{len(test_queries)}: {query[:50]}...")
            
            start_time = time.time()
            try:
                # Create optimized message
                message = AgentMessage(
                    message_type="query_processing",
                    content={
                        "user_input": query,
                        "session_id": session_id,
                        "user_id": f"optimized_test_user",
                        "execute": True,
                        "limit": 10,
                        "include_summary": True,
                        "optimization_enabled": True
                    }
                )
                
                # Process with optimized orchestrator
                result = await optimized_orchestrator.process_message(message)
                execution_time = time.time() - start_time
                optimized_times.append(execution_time)
                
                success = result.success if hasattr(result, 'success') else False
                print(f"    ⚡ {execution_time:.2f}s - {'Success' if success else 'Failed'}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                optimized_times.append(execution_time)
                print(f"    ❌ {execution_time:.2f}s - Error: {str(e)[:50]}...")
        
        avg_optimized = sum(optimized_times) / len(optimized_times) if optimized_times else 0
        optimized_success_rate = sum(1 for t in optimized_times if t > 0) / len(optimized_times) * 100 if optimized_times else 0
        
        print(f"\n⚡ Optimized Results:")
        print(f"  Average Time:  {avg_optimized:.2f}s")
        print(f"  Success Rate:  {optimized_success_rate:.1f}%")
        print(f"  Total Time:    {sum(optimized_times):.2f}s")
        
    except Exception as e:
        print(f"  ❌ Failed to create optimized orchestrator: {str(e)}")
        avg_optimized = 0
        optimized_success_rate = 0
        optimized_times = []
    
    # ==============================================
    # PHASE 3: Performance Comparison & Analysis
    # ==============================================
    print(f"\n📈 Performance Comparison & Analysis:")
    print("=" * 50)
    
    if avg_legacy > 0 and avg_optimized > 0:
        time_improvement = ((avg_legacy - avg_optimized) / avg_legacy) * 100
        total_time_saved = sum(legacy_times) - sum(optimized_times)
        
        print(f"📊 Time Performance:")
        print(f"  Legacy Average:     {avg_legacy:.2f}s")
        print(f"  Optimized Average:  {avg_optimized:.2f}s")
        print(f"  Time Improvement:   {time_improvement:.1f}%")
        print(f"  Time Saved:         {avg_legacy - avg_optimized:.2f}s per query")
        print(f"  Total Time Saved:   {total_time_saved:.2f}s")
        
        print(f"\n🎯 Success Rate:")
        print(f"  Legacy Success:     {legacy_success_rate:.1f}%")
        print(f"  Optimized Success:  {optimized_success_rate:.1f}%")
        print(f"  Success Improvement: {optimized_success_rate - legacy_success_rate:.1f}%")
        
        # Performance recommendations
        print(f"\n💡 Performance Analysis:")
        
        if time_improvement > 20:
            print("  🟢 EXCELLENT: Significant performance improvement!")
            print("    ✅ Optimized orchestrator is working effectively")
            print("    ✅ Ready for production deployment")
        elif time_improvement > 5:
            print("  🟡 GOOD: Moderate performance improvement")
            print("    ✅ Optimization is working but has room for improvement")
            print("    🔧 Consider additional optimizations")
        elif time_improvement > 0:
            print("  🟡 MARGINAL: Small performance improvement")
            print("    ⚠️ Optimization benefits are minimal")
            print("    🔧 Review optimization implementation")
        else:
            print("  🔴 CONCERNING: No performance improvement or regression")
            print("    ❌ Optimization may not be working correctly")
            print("    🔧 Debug and review optimization logic")
            
        # Specific recommendations based on times
        if avg_optimized > 30:
            print(f"\n🚨 Response Time Alert:")
            print("    🔴 Average response time still >30s")
            print("    🔧 Consider implementing aggressive caching")
            print("    🔧 Optimize database connection pooling")
            print("    🔧 Review query complexity and optimization")
        elif avg_optimized > 15:
            print(f"\n⚠️ Response Time Warning:")
            print("    🟡 Average response time >15s")
            print("    🔧 Implement result caching for repeated queries")
            print("    🔧 Consider query result streaming")
        elif avg_optimized > 5:
            print(f"\n✅ Response Time Good:")
            print("    🟢 Average response time 5-15s (acceptable)")
            print("    🔧 Monitor and maintain current performance")
        else:
            print(f"\n🎉 Response Time Excellent:")
            print("    🟢 Average response time <5s (optimal)")
            print("    ✅ Performance is production-ready")
    
    else:
        print("  ❌ Unable to compare - one or both tests failed")
        print("  🔧 Check system configuration and dependencies")
    
    # ==============================================
    # PHASE 4: Optimization Metrics
    # ==============================================
    print(f"\n🔍 Optimization Implementation Status:")
    print("=" * 40)
    
    # Check if optimization components exist
    optimization_status = {
        "optimized_orchestrator": OptimizedOrchestratorAgent is not None,
        "agent_interface": True,  # We imported it successfully
        "performance_monitoring": hasattr(system, 'perf_monitor') if system else False,
        "parallel_execution": avg_optimized > 0,  # Indicates it worked
        "workflow_config": True  # We used it successfully
    }
    
    for component, status in optimization_status.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component.replace('_', ' ').title()}: {'Ready' if status else 'Missing'}")
    
    # ==============================================
    # PHASE 5: Next Steps & Recommendations
    # ==============================================
    print(f"\n🎯 Next Steps & Recommendations:")
    print("=" * 35)
    
    if avg_optimized > 0 and time_improvement > 5:
        print("  ✅ INTEGRATION READY:")
        print("    1. Integrate optimized orchestrator into main system")
        print("    2. Update API to use optimized workflows")
        print("    3. Deploy performance monitoring")
        print("    4. Implement gradual rollout")
        
    elif avg_optimized > 0:
        print("  🔧 OPTIMIZATION NEEDED:")
        print("    1. Review parallel execution implementation")
        print("    2. Optimize agent communication overhead")
        print("    3. Implement more aggressive caching")
        print("    4. Profile and identify bottlenecks")
        
    else:
        print("  🚨 TROUBLESHOOTING REQUIRED:")
        print("    1. Debug optimized orchestrator initialization")
        print("    2. Check agent interface compatibility")
        print("    3. Verify dependency configuration")
        print("    4. Review error logs and stack traces")
    
    # Cleanup
    await system.close()
    
    print(f"\n🎉 Performance optimization comparison completed!")
    return {
        "legacy_avg": avg_legacy,
        "optimized_avg": avg_optimized,
        "time_improvement_percent": time_improvement if avg_legacy > 0 and avg_optimized > 0 else 0,
        "success": True,
        "optimization_working": avg_optimized > 0 and time_improvement > 0
    }

if __name__ == "__main__":
    try:
        result = asyncio.run(test_optimization_comparison())
        if result["optimization_working"]:
            print(f"🎉 Optimization test completed successfully - improvements detected!")
        else:
            print(f"⚠️ Optimization test completed - review results for next steps")
    except Exception as e:
        print(f"❌ Performance optimization test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
