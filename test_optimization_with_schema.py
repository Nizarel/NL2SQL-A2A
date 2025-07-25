"""
Test with proper schema-compliant queries
"""

import requests
import json
import time

def test_schema_compliant_queries():
    """Test queries that follow the schema requirements"""
    
    base_url = "http://localhost:8000"
    
    print("🎯 Testing Schema-Compliant Queries")
    print("=" * 40)
    
    # Test queries that should work with proper schema
    test_queries = [
        {
            "name": "Simple table info",
            "question": "Show me information about dev.customers table",
            "timeout": 20
        },
        {
            "name": "Basic count query", 
            "question": "How many records are in dev.customers?",
            "timeout": 15
        },
        {
            "name": "Schema exploration",
            "question": "What tables are available in the dev schema?",
            "timeout": 15
        }
    ]
    
    for i, query_test in enumerate(test_queries, 1):
        print(f"\n📋 Test {i}: {query_test['name']}")
        print(f"   Question: {query_test['question']}")
        
        try:
            query_data = {
                "question": query_test['question'],
                "user_id": "test_user",
                "session_id": f"test_session_{i}"
            }
            
            start_time = time.time()
            response = requests.post(
                f"{base_url}/orchestrator/query", 
                json=query_data,
                timeout=query_test['timeout']
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success: {result.get('success', False)}")
                
                data = result.get('data', {})
                exec_time = data.get('execution_time', end_time - start_time)
                print(f"   ⏱️ Execution time: {exec_time:.2f}s")
                
                # Check for optimization metadata
                if 'orchestrator_type' in data:
                    print(f"   🤖 Orchestrator: {data['orchestrator_type']}")
                if 'workflow_type' in data:
                    print(f"   🔄 Workflow: {data['workflow_type']}")
                if 'agent_calls' in data:
                    print(f"   📞 Agent calls: {data['agent_calls']}")
                    
            else:
                print(f"   ❌ Failed - Status: {response.status_code}")
                if response.text:
                    error_text = response.text[:150]
                    print(f"   Error: {error_text}...")
                    
        except requests.exceptions.Timeout:
            print(f"   ⏱️ Timeout after {query_test['timeout']}s")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def check_performance_after_queries():
    """Check performance metrics after running queries"""
    
    base_url = "http://localhost:8000"
    
    print("\n📊 Performance After Queries:")
    try:
        response = requests.get(f"{base_url}/performance/summary", timeout=5)
        if response.status_code == 200:
            data = response.json()
            summary_data = data.get('data', {})
            print(f"   Total operations: {summary_data.get('total_operations', 0)}")
            print(f"   Average time: {summary_data.get('average_execution_time', 0):.3f}s")
            if summary_data.get('total_operations', 0) > 0:
                print("   ✅ Performance tracking working!")
            else:
                print("   ℹ️ No operations tracked yet")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

if __name__ == "__main__":
    test_schema_compliant_queries()
    check_performance_after_queries()
    
    print("\n🎉 Schema-Compliant Test Complete!")
    print("✅ Testing optimization with proper queries")
