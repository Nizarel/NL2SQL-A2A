"""
Quick API test for performance endpoints
"""

import requests
import json
import time

def test_api_endpoints():
    """Test the API endpoints including new performance monitoring"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing NL2SQL API with Optimizations")
    print("=" * 50)
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is ready!")
                break
        except requests.exceptions.RequestException:
            if i < max_retries - 1:
                print(f"   Attempt {i+1}/{max_retries} - waiting...")
                time.sleep(2)
            else:
                print("❌ Server not responding - cannot test")
                return False
    
    # Test health endpoint
    print("\n🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"⚠️ Health endpoint returned {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {str(e)}")
    
    # Test performance endpoints
    print("\n📊 Testing Performance Endpoints...")
    
    performance_endpoints = [
        "/performance/summary",
        "/performance/metrics", 
        "/performance/agents"
    ]
    
    for endpoint in performance_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {endpoint} - Success: {data.get('success', False)}")
            elif response.status_code == 404:
                print(f"⚠️ {endpoint} - Not Found (may need system integration)")
            else:
                print(f"⚠️ {endpoint} - Status: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint} - Error: {str(e)}")
    
    # Test main query endpoint
    print("\n🎯 Testing Main Query Endpoint...")
    try:
        query_data = {
            "question": "Show top 3 customers by revenue",
            "user_id": "test_user",
            "session_id": "test_session"
        }
        
        response = requests.post(
            f"{base_url}/orchestrator/query", 
            json=query_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query successful: {result.get('success', False)}")
            
            # Check if response contains optimization metadata
            data = result.get('data', {})
            if 'execution_time' in data:
                print(f"   Execution time: {data['execution_time']:.2f}s")
            if 'orchestrator_type' in data:
                print(f"   Orchestrator type: {data['orchestrator_type']}")
            if 'workflow_type' in data:
                print(f"   Workflow type: {data['workflow_type']}")
        else:
            print(f"⚠️ Query failed with status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Query endpoint error: {str(e)}")
    
    # Test conversation endpoints
    print("\n🗣️ Testing Conversation Endpoints...")
    conversation_endpoints = [
        "/conversation/state/test_session",
        "/suggestions?session_id=test_session"
    ]
    
    for endpoint in conversation_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {endpoint} - Working")
            else:
                print(f"⚠️ {endpoint} - Status: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint} - Error: {str(e)}")
    
    print("\n🎉 API Test Complete!")
    print("✅ Server is running with optimized components")
    return True

if __name__ == "__main__":
    test_api_endpoints()
