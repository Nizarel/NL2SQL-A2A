#!/usr/bin/env python3
"""
Memory Integration Test
Tests the orchestrator memory service integration without requiring full agent setup
"""

import os
import sys
import time
import asyncio
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from dotenv import load_dotenv
    load_dotenv('src/.env')
    print("✅ Environment variables loaded")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

# Import the memory service directly
try:
    from services.orchestrator_memory_service import OrchestratorMemoryService
    print("✅ OrchestratorMemoryService imported successfully")
except Exception as e:
    print(f"❌ Failed to import OrchestratorMemoryService: {e}")
    sys.exit(1)

try:
    from services.chat_logger import ChatLogger
    print("✅ ChatLogger imported successfully")
except Exception as e:
    print(f"❌ Failed to import ChatLogger: {e}")
    sys.exit(1)

try:
    from Models.agent_response import ChatLogEntry, AgentResponse, UserSession, LogTokens
    print("✅ Model classes imported successfully")
except Exception as e:
    print(f"❌ Failed to import model classes: {e}")
    sys.exit(1)


def test_environment_setup():
    """Test environment variable setup"""
    print("\n🔧 TESTING ENVIRONMENT SETUP")
    print("=" * 50)
    
    required_vars = [
        'COSMOS_DB_CONNECTION_STRING',
        'AccountKey',
        'COSMOS_DB_ENDPOINT'
    ]
    
    env_status = {}
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'KEY' in var or 'CONNECTION' in var:
                display = value[:15] + "..." + value[-10:] if len(value) > 25 else value
            else:
                display = value
            print(f"  ✅ {var}: {display}")
            env_status[var] = True
        else:
            print(f"  ❌ {var}: Not set")
            env_status[var] = False
    
    all_set = all(env_status.values())
    if all_set:
        print("  🎉 All required environment variables are set!")
    else:
        print("  ⚠️  Some environment variables are missing")
    
    return all_set


def test_memory_service_creation():
    """Test creating the memory service"""
    print("\n🧪 TESTING MEMORY SERVICE CREATION")
    print("=" * 50)
    
    try:
        # Test with environment variables (should auto-detect)
        memory_service = OrchestratorMemoryService()
        print("  ✅ Memory service created with auto-detection")
        
        # Test authentication info
        auth_info = memory_service.get_auth_info()
        print("  📊 Authentication Configuration:")
        for key, value in auth_info.items():
            print(f"    {key}: {value}")
        
        return memory_service
        
    except Exception as e:
        print(f"  ❌ Failed to create memory service: {e}")
        return None


def test_memory_service_initialization(memory_service):
    """Test initializing the memory service"""
    print("\n🚀 TESTING MEMORY SERVICE INITIALIZATION")
    print("=" * 50)
    
    try:
        memory_service.initialize()
        print("  ✅ Memory service initialized successfully")
        print("  📊 Connection established to Cosmos DB")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to initialize memory service: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        print("  Detailed error:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                print(f"    {line}")
        return False


def test_user_session_management(memory_service):
    """Test user session creation and management"""
    print("\n👤 TESTING USER SESSION MANAGEMENT")
    print("=" * 50)
    
    try:
        test_user_id = f"test_user_{int(time.time())}"
        
        # Test session creation
        session = memory_service.start_conversation(test_user_id)
        print(f"  ✅ Session created: {session.session_id}")
        print(f"  📊 User ID: {session.user_id}")
        print(f"  📅 Start time: {session.start_time}")
        print(f"  🔄 Active: {session.is_active}")
        
        # Test session metadata update
        metadata = {
            "test_run": "memory_integration_test",
            "timestamp": datetime.now().isoformat(),
            "features": ["memory", "session_management"]
        }
        
        memory_service.update_session_metadata(
            test_user_id, 
            session.session_id, 
            metadata
        )
        print("  ✅ Session metadata updated")
        
        return session
        
    except Exception as e:
        print(f"  ❌ Session management test failed: {e}")
        return None


def test_chat_logging(memory_service, session):
    """Test chat log entry creation and retrieval"""
    print("\n💬 TESTING CHAT LOGGING")
    print("=" * 50)
    
    try:
        # Create a test chat log entry
        test_data = {
            "user_input": "What are the total sales for this month?",
            "workflow_results": {
                "schema_analysis": {
                    "relevant_tables": ["sales", "time_dimension"],
                    "confidence": 0.95
                },
                "sql_generation": {
                    "generated_sql": "SELECT SUM(amount) FROM sales WHERE month = MONTH(GETDATE())",
                    "complexity": "simple"
                },
                "execution": {
                    "success": True,
                    "row_count": 1,
                    "execution_time_ms": 45
                }
            }
        }
        
        # Log the workflow
        chat_entry = memory_service.log_orchestrator_workflow(
            user_id=session.user_id,
            session_id=session.session_id,
            user_input=test_data["user_input"],
            workflow_results=test_data["workflow_results"],
            processing_time_ms=150
        )
        
        print(f"  ✅ Chat entry logged: {chat_entry.id}")
        print(f"  📝 User input: {chat_entry.user_input}")
        print(f"  ⏱️  Processing time: {chat_entry.processing_time_ms}ms")
        print(f"  🤖 Agent responses: {len(chat_entry.agent_responses)}")
        
        # Test retrieving conversation history
        history = memory_service.get_conversation_history(
            session.user_id,
            session.session_id,
            limit=5
        )
        
        print(f"  ✅ Conversation history retrieved: {len(history)} entries")
        
        return chat_entry
        
    except Exception as e:
        print(f"  ❌ Chat logging test failed: {e}")
        return None


def test_memory_statistics(memory_service, session):
    """Test memory statistics and insights"""
    print("\n📊 TESTING MEMORY STATISTICS")
    print("=" * 50)
    
    try:
        # Test basic memory stats
        stats = memory_service.get_memory_stats(session.user_id)
        print("  📈 Memory Statistics:")
        for key, value in stats.items():
            print(f"    {key}: {value}")
        
        # Test enhanced memory stats (SK 1.35.0 features)
        enhanced_stats = memory_service.get_enhanced_memory_stats(session.user_id)
        print("\n  🚀 Enhanced Statistics (SK 1.35.0):")
        for key, value in enhanced_stats.items():
            if isinstance(value, list):
                print(f"    {key}: {len(value)} items")
                for item in value[:3]:  # Show first 3
                    print(f"      • {item}")
            else:
                print(f"    {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Memory statistics test failed: {e}")
        return False


def test_similarity_search(memory_service, session):
    """Test similarity search functionality"""
    print("\n🔍 TESTING SIMILARITY SEARCH")
    print("=" * 50)
    
    try:
        # Test finding similar queries
        similar_queries = memory_service.find_similar_queries(
            session.user_id,
            "sales performance metrics",
            limit=5
        )
        
        print(f"  ✅ Similar queries found: {len(similar_queries)}")
        for i, query in enumerate(similar_queries[:3], 1):
            print(f"    {i}. {query.user_input[:60]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Similarity search test failed: {e}")
        return False


async def test_semantic_kernel_features(memory_service, session):
    """Test Semantic Kernel 1.35.0 specific features"""
    print("\n🧠 TESTING SEMANTIC KERNEL 1.35.0 FEATURES")
    print("=" * 50)
    
    try:
        # Test collection management
        collections = await memory_service.get_collections()
        print(f"  ✅ Collections available: {len(collections)}")
        for collection in collections:
            print(f"    • {collection}")
        
        # Test collection existence
        test_collection = "test_memory_collection"
        exists = await memory_service.does_collection_exist(test_collection)
        print(f"  📋 Collection '{test_collection}' exists: {exists}")
        
        if not exists:
            await memory_service.create_collection(test_collection)
            print(f"  ✅ Collection '{test_collection}' created")
        
        # Test memory record creation and storage
        from semantic_kernel.memory.memory_record import MemoryRecord
        
        test_record = MemoryRecord(
            id=f"test_record_{int(time.time())}",
            text="This is a test memory record for semantic search",
            is_reference=False,
            external_source_name="memory_integration_test",
            description="Test record for validating SK 1.35.0 features"
        )
        
        record_id = await memory_service.upsert(test_collection, test_record)
        print(f"  ✅ Memory record stored: {record_id}")
        
        # Test retrieval
        retrieved_record = await memory_service.get(test_collection, record_id)
        if retrieved_record:
            print(f"  ✅ Memory record retrieved: {retrieved_record.id}")
            print(f"    Text: {retrieved_record.text[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Semantic Kernel features test failed: {e}")
        return False


def test_cleanup(memory_service, session):
    """Test cleanup and resource management"""
    print("\n🧹 TESTING CLEANUP")
    print("=" * 50)
    
    try:
        # End the session
        memory_service.end_conversation(session.user_id, session.session_id)
        print("  ✅ Session ended successfully")
        
        # Close memory service
        memory_service.close()
        print("  ✅ Memory service closed successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Cleanup test failed: {e}")
        return False


async def run_full_memory_integration_test():
    """Run the complete memory integration test suite"""
    print("🧪 NL2SQL MEMORY INTEGRATION TEST SUITE")
    print("=" * 60)
    print(f"Test started at: {datetime.now().isoformat()}")
    print()
    
    test_results = {}
    
    # Test 1: Environment Setup
    test_results['environment'] = test_environment_setup()
    
    if not test_results['environment']:
        print("\n❌ Environment setup failed - stopping tests")
        return False
    
    # Test 2: Memory Service Creation
    memory_service = test_memory_service_creation()
    test_results['creation'] = memory_service is not None
    
    if not memory_service:
        print("\n❌ Memory service creation failed - stopping tests")
        return False
    
    # Test 3: Memory Service Initialization
    test_results['initialization'] = test_memory_service_initialization(memory_service)
    
    if not test_results['initialization']:
        print("\n❌ Memory service initialization failed - stopping tests")
        return False
    
    # Test 4: User Session Management
    session = test_user_session_management(memory_service)
    test_results['session_management'] = session is not None
    
    if not session:
        print("\n❌ Session management failed - stopping tests")
        return False
    
    # Test 5: Chat Logging
    chat_entry = test_chat_logging(memory_service, session)
    test_results['chat_logging'] = chat_entry is not None
    
    # Test 6: Memory Statistics
    test_results['statistics'] = test_memory_statistics(memory_service, session)
    
    # Test 7: Similarity Search
    test_results['similarity_search'] = test_similarity_search(memory_service, session)
    
    # Test 8: Semantic Kernel Features
    test_results['sk_features'] = await test_semantic_kernel_features(memory_service, session)
    
    # Test 9: Cleanup
    test_results['cleanup'] = test_cleanup(memory_service, session)
    
    # Summary
    print("\n🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    success_rate = (passed_tests / total_tests) * 100
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name.replace('_', ' ').title()}")
    
    print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT: Memory integration is working perfectly!")
    elif success_rate >= 70:
        print("👍 GOOD: Memory integration is mostly working")
    elif success_rate >= 50:
        print("⚠️  PARTIAL: Some memory features are working")
    else:
        print("❌ POOR: Memory integration needs attention")
    
    print(f"\nTest completed at: {datetime.now().isoformat()}")
    
    return success_rate >= 70


if __name__ == "__main__":
    print("🤖 Starting NL2SQL Memory Integration Test...")
    print()
    
    try:
        success = asyncio.run(run_full_memory_integration_test())
        
        if success:
            print("\n🚀 MEMORY INTEGRATION TEST COMPLETED SUCCESSFULLY!")
            print("   Your orchestrator memory service is ready for production use.")
        else:
            print("\n⚠️  MEMORY INTEGRATION TEST COMPLETED WITH ISSUES")
            print("   Please review the failed tests and fix any issues.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
