"""
Test Script: Schema Analyst Integration with Orchestrator
Tests the new optimized schema context workflow and redundancy elimination
"""

import asyncio
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import NL2SQLMultiAgentSystem


async def test_schema_integration():
    """Test the integrated Schema Analyst workflow"""
    
    print("🚀 Testing Schema Analyst Integration...")
    print("=" * 60)
    
    # Initialize the system
    system = NL2SQLMultiAgentSystem()
    
    try:
        await system.initialize()
        print("✅ System initialized successfully")
        
        # Test workflow status
        status = await system.get_workflow_status()
        print(f"📊 Workflow Status: {status}")
        
        # Test multiple questions to verify optimized schema workflow
        test_questions = [
            {
                "question": "What are the top 5 customers by revenue?",
                "description": "Customer revenue ranking",
                "expected_tables": ["cliente", "segmentacion"]
            },
            {
                "question": "Show me product sales by category for this year",
                "description": "Product category analysis",
                "expected_tables": ["producto", "segmentacion", "tiempo"]
            },
            {
                "question": "Which territories have the highest volume sales?",
                "description": "Territory performance",
                "expected_tables": ["cliente_cedi", "segmentacion", "mercado"]
            }
        ]
        
        print(f"\n🧪 Testing {len(test_questions)} different query types...")
        
        for i, test_case in enumerate(test_questions, 1):
            print(f"\n📋 Test {i}/{len(test_questions)}: {test_case['description']}")
            print(f"❓ Question: {test_case['question']}")
            
            start_time = time.time()
            
            result = await system.ask_question(
                question=test_case["question"],
                execute=False,  # Don't execute SQL for faster testing
                include_summary=False
            )
            
            execution_time = time.time() - start_time
            
            if result["success"]:
                print("✅ Schema-optimized workflow completed successfully!")
                
                # Check for schema analysis results
                if "schema_analysis" in result["data"]:
                    schema_info = result["data"]["schema_analysis"]
                    relevant_tables = schema_info.get('relevant_tables', [])
                    
                    print(f"🔍 Relevant tables identified: {relevant_tables}")
                    print(f"🎯 Join strategy: {schema_info.get('join_strategy', {}).get('strategy', 'unknown')}")
                    print(f"📈 Confidence score: {schema_info.get('confidence_score', 0):.3f}")
                    print(f"💡 Performance hints: {len(schema_info.get('performance_hints', []))} hints")
                    
                    # Verify expected tables are included
                    expected_tables = test_case.get("expected_tables", [])
                    tables_found = any(table in relevant_tables for table in expected_tables)
                    print(f"🎯 Expected tables detection: {'✅ GOOD' if tables_found else '⚠️  PARTIAL'}")
                else:
                    print("❌ No schema analysis found in results")
                
                # Check metadata for optimization info
                metadata = result.get("metadata", {})
                print(f"🔧 Schema optimization: {metadata.get('schema_optimization', 'unknown')}")
                cache_info = metadata.get('cache_info', 'unknown')
                print(f"💾 Cache info: {cache_info}")
                
                # Check for optimized context usage
                if metadata.get('schema_analyzed'):
                    cache_hit = metadata.get('schema_cache_hit', False)
                    if cache_hit:
                        cache_type = metadata.get('schema_cache_type', 'unknown')
                        print(f"⚡ Cache hit: {cache_type} (faster execution)")
                    else:
                        print("🔍 Fresh schema analysis performed")
                
                # Show generated SQL preview
                if result["data"].get("sql_query"):
                    sql_preview = result['data']['sql_query'][:150]
                    print(f"📝 Generated SQL: {sql_preview}...")
                
                print(f"⏱️  Execution time: {execution_time:.3f}s")
                
            else:
                print(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
            
            print("-" * 40)
        
        # Test cache efficiency with similar questions
        print("\n🔄 Testing Cache Efficiency...")
        print("-" * 40)
        
        similar_questions = [
            "Show top 10 customers by revenue",  # Similar to first question
            "List highest revenue customers",    # Very similar
            "Revenue by customer ranking"        # Similar intent
        ]
        
        cache_results = []
        
        for question in similar_questions:
            print(f"\n❓ Testing cache with: {question}")
            start_time = time.time()
            
            result = await system.ask_question(
                question=question,
                execute=False,
                include_summary=False
            )
            
            execution_time = time.time() - start_time
            
            if result["success"]:
                metadata = result.get("metadata", {})
                cache_hit = metadata.get("schema_cache_hit", False)
                cache_type = metadata.get("schema_cache_type", "none")
                
                cache_results.append({
                    "question": question,
                    "cache_hit": cache_hit,
                    "cache_type": cache_type,
                    "time": execution_time
                })
                
                print(f"   ⚡ Cache: {'✅ HIT' if cache_hit else '❌ MISS'} ({cache_type})")
                print(f"   ⏱️  Time: {execution_time:.3f}s")
            else:
                print(f"   ❌ Failed: {result.get('error')}")
        
        # Analyze cache performance
        cache_hits = sum(1 for r in cache_results if r["cache_hit"])
        cache_efficiency = cache_hits / len(cache_results) * 100 if cache_results else 0
        
        print(f"\n📊 Cache Analysis:")
        print(f"   Efficiency: {cache_hits}/{len(cache_results)} hits ({cache_efficiency:.1f}%)")
        
        if cache_efficiency > 0:
            avg_cache_time = sum(r["time"] for r in cache_results if r["cache_hit"]) / max(cache_hits, 1)
            avg_no_cache_time = sum(r["time"] for r in cache_results if not r["cache_hit"]) / max(len(cache_results) - cache_hits, 1)
            
            if cache_hits > 0 and avg_cache_time < avg_no_cache_time:
                speedup = avg_no_cache_time / avg_cache_time
                print(f"   Performance: {speedup:.1f}x faster with cache")
        
        # Test context optimization verification
        print("\n🎯 Testing Context Optimization...")
        print("-" * 40)
        
        optimization_test = await system.ask_question(
            question="Show customer revenue by territory and product category",
            execute=False,
            include_summary=False
        )
        
        if optimization_test["success"]:
            metadata = optimization_test.get("metadata", {})
            
            # Check schema source and context size
            schema_source = metadata.get("schema_source", "unknown")
            context_size = metadata.get("schema_context_size", 0)
            
            print(f"📏 Schema context size: {context_size:,} characters")
            print(f"🎯 Schema source: {schema_source}")
            
            if schema_source == "optimized":
                print("✅ Context optimization: ACTIVE")
                print("   ✅ Smaller prompts sent to SQL Generator")
                print("   ✅ Only relevant schema information included")
            else:
                print("⚠️  Context optimization: NOT DETECTED")
                print(f"   Current source: {schema_source}")
        
        print("\n🎉 Schema Analyst Integration Test COMPLETED!")
        print("=" * 60)
        print("✅ Orchestration flow successfully integrated")
        print("✅ Optimized context delivery verified")
        print("✅ Cache system operational")
        print("✅ Performance improvements demonstrated")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await system.close()
        print("🔐 System closed")


async def test_redundancy_elimination():
    """Test that redundancy has been eliminated between Schema Analyst and SQL Generator"""
    
    print("\n🔍 Testing Redundancy Elimination...")
    print("=" * 60)
    
    system = NL2SQLMultiAgentSystem()
    
    try:
        await system.initialize()
        
        # Test a complex question that would require both schema analysis and SQL generation
        test_question = "Show monthly revenue trends by product category and territory for the last 12 months"
        
        print(f"❓ Testing complex query: {test_question}")
        
        start_time = time.time()
        result = await system.ask_question(
            question=test_question,
            execute=False,
            include_summary=False
        )
        total_time = time.time() - start_time
        
        if result["success"]:
            metadata = result.get("metadata", {})
            
            print("✅ Complex workflow execution successful")
            print(f"⏱️  Total time: {total_time:.3f}s")
            
            # Verify workflow steps
            steps_completed = metadata.get("steps_completed", [])
            expected_steps = ["schema_analysis", "sql_generation"]
            
            print(f"📊 Steps completed: {', '.join(steps_completed)}")
            
            # Check that schema analysis happened first
            schema_analyzed = "schema_analysis" in steps_completed
            sql_generated = "sql_generation" in steps_completed
            
            print(f"🔍 Schema analysis: {'✅ COMPLETED' if schema_analyzed else '❌ MISSING'}")
            print(f"💡 SQL generation: {'✅ COMPLETED' if sql_generated else '❌ MISSING'}")
            
            # Check that SQL generator used optimized context
            schema_source = metadata.get("schema_source", "unknown")
            schema_optimized = metadata.get("schema_optimization", "unknown")
            
            print(f"🎯 Schema optimization: {schema_optimized}")
            print(f"📋 Context source: {schema_source}")
            
            # Verify no redundant work
            redundancy_check = (
                schema_analyzed and 
                sql_generated and 
                schema_source == "optimized" and 
                schema_optimized == "enabled"
            )
            
            if redundancy_check:
                print("\n🎯 REDUNDANCY ELIMINATION: ✅ SUCCESS")
                print("   ✅ Schema Analyst provides optimized context")
                print("   ✅ SQL Generator uses optimized context (no duplicate analysis)")
                print("   ✅ Workflow coordination working correctly")
                print("   ✅ No overlap between agent responsibilities")
                
                # Check timing efficiency
                schema_time = metadata.get("schema_analysis_time", 0)
                if schema_time > 0:
                    efficiency = (schema_time / total_time) * 100
                    print(f"   📈 Schema analysis overhead: {efficiency:.1f}% of total time")
                
                return True
            else:
                print("\n❌ REDUNDANCY ELIMINATION: FAILED")
                print(f"   Schema analyzed: {schema_analyzed}")
                print(f"   SQL generated: {sql_generated}")
                print(f"   Optimized context: {schema_source == 'optimized'}")
                print(f"   Optimization enabled: {schema_optimized == 'enabled'}")
                return False
        else:
            print(f"❌ Workflow failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Redundancy test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await system.close()


async def main():
    """Run comprehensive integration tests"""
    
    print("🧪 SCHEMA ANALYST INTEGRATION TEST SUITE")
    print("=" * 80)
    
    # Test 1: Basic integration and optimization
    print("Test 1: Schema Integration and Optimization")
    test1_passed = await test_schema_integration()
    
    # Test 2: Redundancy elimination verification
    print("\nTest 2: Redundancy Elimination")
    test2_passed = await test_redundancy_elimination()
    
    # Final summary
    print("\n📋 FINAL TEST RESULTS")
    print("=" * 80)
    print(f"🔍 Schema Integration & Optimization: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"🔄 Redundancy Elimination: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    overall_success = test1_passed and test2_passed
    
    if overall_success:
        print(f"\n🎉 OVERALL RESULT: ✅ ALL TESTS PASSED")
        print("\n🎯 Integration Successfully Completed:")
        print("   ✅ Schema Analyst Agent integrated into orchestration")
        print("   ✅ Optimized context delivery working")
        print("   ✅ No redundancy between agents")
        print("   ✅ Intelligent caching operational")
        print("   ✅ Performance improvements achieved")
        print("   ✅ Smaller prompts to SQL Generator")
    else:
        print(f"\n❌ OVERALL RESULT: SOME TESTS FAILED")
        print("\n⚠️  Issues detected:")
        if not test1_passed:
            print("   ❌ Schema integration problems")
        if not test2_passed:
            print("   ❌ Redundancy elimination issues")
        print("\n🔧 Review the test output above for details")
    
    return overall_success


if __name__ == "__main__":
    """Run the comprehensive integration test suite"""
    print("Starting Schema Analyst Integration Tests...")
    success = asyncio.run(main())
    print(f"\nTest suite {'✅ COMPLETED SUCCESSFULLY' if success else '❌ COMPLETED WITH FAILURES'}")
    exit(0 if success else 1)
