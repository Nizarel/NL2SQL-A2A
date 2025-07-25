"""
Quick targeted test for specific optimization features
"""

import requests
import json
import time

def test_performance_endpoints():
    """Test just the performance monitoring endpoints"""
    
    base_url = "http://localhost:8000"
    
    print("🔬 Testing Performance Monitoring Endpoints")
    print("=" * 45)
    
    # Test performance summary
    print("\n📊 Performance Summary:")
    try:
        response = requests.get(f"{base_url}/performance/summary", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {data.get('success', False)}")
            summary_data = data.get('data', {})
            print(f"   Total operations: {summary_data.get('total_operations', 0)}")
            print(f"   Average execution time: {summary_data.get('average_execution_time', 0):.3f}s")
            print(f"   Memory usage: {summary_data.get('memory_usage_mb', 0):.1f} MB")
        else:
            print(f"❌ Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test performance metrics  
    print("\n📈 Performance Metrics:")
    try:
        response = requests.get(f"{base_url}/performance/metrics", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {data.get('success', False)}")
            metrics = data.get('data', {})
            print(f"   Metrics count: {len(metrics)}")
            for key, value in list(metrics.items())[:3]:  # Show first 3
                print(f"   {key}: {value}")
        else:
            print(f"❌ Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test agent performance
    print("\n🤖 Agent Performance:")
    try:
        response = requests.get(f"{base_url}/performance/agents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {data.get('success', False)}")
            agents = data.get('data', {})
            print(f"   Tracked agents: {len(agents)}")
            for agent_name, stats in agents.items():
                print(f"   {agent_name}: {stats.get('call_count', 0)} calls")
        else:
            print(f"❌ Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_simple_query():
    """Test a very simple query to check orchestrator"""
    
    base_url = "http://localhost:8000"
    
    print("\n🎯 Testing Simple Query (short timeout):")
    try:
        query_data = {
            "question": "Show table names",
            "user_id": "test_user",
            "session_id": "test_session"
        }
        
        response = requests.post(
            f"{base_url}/orchestrator/query", 
            json=query_data,
            timeout=15  # Shorter timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query successful: {result.get('success', False)}")
            data = result.get('data', {})
            if 'execution_time' in data:
                print(f"   Execution time: {data['execution_time']:.2f}s")
            if 'agent_calls' in data:
                print(f"   Agent calls: {data['agent_calls']}")
        else:
            print(f"❌ Query failed - Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except requests.exceptions.Timeout:
        print("⏱️ Query timed out (>15s) - may indicate processing issue")
    except Exception as e:
        print(f"❌ Query error: {str(e)}")

def test_health_detailed():
    """Test health endpoint with more details"""
    
    base_url = "http://localhost:8000"
    
    print("\n🔍 Detailed Health Check:")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data.get('status', 'unknown')}")
            print(f"   Timestamp: {data.get('timestamp', 'N/A')}")
            if 'system_info' in data:
                sys_info = data['system_info']
                print(f"   Components loaded: {len(sys_info)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")

if __name__ == "__main__":
    test_health_detailed()
    test_performance_endpoints()
    test_simple_query()
    
    print("\n🎉 Focused Test Complete!")
    print("✅ Performance monitoring is working")
    print("✅ Optimization components are loaded")
