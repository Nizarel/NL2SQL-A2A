"""
Test script for optimized components with absolute imports
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

def test_imports():
    """Test all optimized component imports"""
    try:
        print("🧪 Testing optimized components...")
        
        # Test agent interface
        from agents.agent_interface import AgentMessage, WorkflowConfig, WorkflowStage
        print("✅ Agent interface imported")
        
        # Test performance monitor
        from services.performance_monitor import PerformanceMonitor, perf_monitor
        print("✅ Performance monitor imported")
        
        # Create test instances
        config = WorkflowConfig(
            parallel_execution=True,
            max_parallel_agents=3,
            enable_caching=True
        )
        print("✅ WorkflowConfig created")
        
        message = AgentMessage(
            message_type="test",
            content={"test": "data"}
        )
        print("✅ AgentMessage created")
        
        # Test performance monitor
        monitor = PerformanceMonitor()
        print("✅ Performance monitor created")
        
        print("\n🎉 All optimized components are working!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        from services.performance_monitor import perf_monitor
        
        # Test performance tracking
        @perf_monitor.track_sync("test_function")
        def test_function():
            return "test_result"
        
        result = test_function()
        print(f"✅ Performance tracking test: {result}")
        
        # Get metrics
        metrics = perf_monitor.get_summary()
        print(f"✅ Performance metrics: {metrics.get('total_operations', 0)} operations")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Optimization Component Tests")
    print("=" * 50)
    
    # Test imports
    import_success = test_imports()
    
    if import_success:
        # Test functionality
        func_success = test_basic_functionality()
        
        if func_success:
            print("\n✅ All tests passed! Components are ready for integration.")
        else:
            print("\n⚠️ Import tests passed but functionality tests failed.")
    else:
        print("\n❌ Import tests failed. Please check component setup.")
    
    print("\n📋 Next Steps:")
    print("1. Run 'python test_quick_optimization.py' to test system integration")
    print("2. Use the performance endpoints in the API to monitor improvements")
    print("3. Review the conversation features to ensure no regression")
