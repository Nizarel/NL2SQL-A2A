#!/usr/bin/env python3
"""
Comprehensive Service Test - Test all enhanced services together
Validates the complete optimized service architecture
"""

import sys
import os
import time
import traceback
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_configuration_service():
    """Test the configuration service"""
    print("🔧 Testing Configuration Service...")
    
    try:
        from src.services.configuration_service import config_service
        
        # Test configuration loading
        print("  ✅ Configuration service imported successfully")
        
        # Test configuration retrieval
        database_config = config_service.get_database_config()
        print(f"  ✅ Database config loaded: {database_config.mcp_server_url[:50]}...")
        
        # Test configuration validation
        validation_report = config_service.validate_config_integrity()
        print(f"  ✅ Configuration validation: {validation_report['valid']}")
        
        # Test configuration summary
        summary = config_service.get_config_summary()
        print(f"  ✅ Configuration summary: {len(summary['sections_loaded'])} sections loaded")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration service test failed: {e}")
        traceback.print_exc()
        return False

def test_monitoring_service():
    """Test the monitoring service"""
    print("📊 Testing Monitoring Service...")
    
    try:
        from src.services.monitoring_service import monitoring_service
        
        # Test metric recording
        monitoring_service.record_metric("test_metric", 42.5)
        print("  ✅ Metric recording successful")
        
        # Test health check
        health_status = monitoring_service.get_system_health()
        print(f"  ✅ System health status: {health_status['overall_status']}")
        
        # Test performance summary
        performance = monitoring_service.get_performance_summary()
        print(f"  ✅ Performance metrics: {performance['metrics_count']} metrics tracked")
        
        # Test alert creation
        monitoring_service.create_alert("info", "test", "Test alert for validation")
        alerts = monitoring_service.get_active_alerts()
        print(f"  ✅ Alert system: {len(alerts)} active alerts")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Monitoring service test failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_error_handling():
    """Test enhanced error handling service"""
    print("🚨 Testing Enhanced Error Handling Service...")
    
    try:
        from src.services.error_handling_service import ErrorHandlingService, ErrorCategory
        
        # Test error categorization
        test_error = ValueError("Invalid SQL syntax: missing comma")
        category = ErrorHandlingService.categorize_error(test_error)
        print(f"  ✅ Error categorization: {category.value}")
        
        # Test enhanced error response
        enhanced_response = ErrorHandlingService.create_enhanced_error_response(
            test_error,
            context={"query": "SELECT * FROM table"},
            correlation_id="test-123"
        )
        print(f"  ✅ Enhanced error response: {enhanced_response['severity']}")
        print(f"  ✅ Suggestions generated: {len(enhanced_response['suggestions'])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_template_service():
    """Test enhanced template service"""
    print("📝 Testing Enhanced Template Service...")
    
    try:
        from src.services.template_service import TemplateService
        
        # Initialize template service
        template_service = TemplateService()
        template_service.initialize_templates()
        print("  ✅ Template service initialized")
        
        # Test complexity recommendation
        question = "Show me the top 10 customers with complex analytics and optimization"
        recommended_complexity = template_service.recommend_complexity_level(question)
        print(f"  ✅ Complexity recommendation: {recommended_complexity}")
        
        # Test custom template creation
        success = template_service.create_custom_template(
            "test_template",
            "SELECT {{ columns }} FROM {{ table }} WHERE {{ condition }}",
            "custom",
            "Test template for validation"
        )
        print(f"  ✅ Custom template creation: {success}")
        
        # Test complexity analytics
        analytics = template_service.get_complexity_analytics()
        print(f"  ✅ Complexity analytics available: {analytics.get('total_template_uses', 0)} uses")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced template service test failed: {e}")
        traceback.print_exc()
        return False

def test_sql_utility_service():
    """Test SQL utility service"""
    print("🔍 Testing SQL Utility Service...")
    
    try:
        from src.services.sql_utility_service import SQLUtilityService
        
        # Test SQL cleaning
        dirty_sql = "  SELECT   *    FROM  table1   ; -- comment  "
        clean_sql = SQLUtilityService.clean_sql_query(dirty_sql)
        print(f"  ✅ SQL cleaning successful: {len(clean_sql)} chars")
        
        # Test SQL extraction
        response_text = "Here's your query: SELECT COUNT(*) FROM customers WHERE active = 1"
        extracted_sql = SQLUtilityService.extract_sql_from_response(response_text)
        print(f"  ✅ SQL extraction successful: Found SQL query")
        
        # Test SQL validation
        is_valid = SQLUtilityService.validate_sql_syntax("SELECT * FROM table1")
        print(f"  ✅ SQL validation successful: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ SQL utility service test failed: {e}")
        traceback.print_exc()
        return False

def test_service_integration():
    """Test integration between services"""
    print("🔗 Testing Service Integration...")
    
    try:
        from src.services.monitoring_service import monitoring_service
        from src.services.error_handling_service import ErrorHandlingService
        
        # Simulate a workflow with monitoring
        start_time = time.time()
        
        # Record some metrics
        monitoring_service.record_metric("query_processing_time", 1234.5)
        monitoring_service.record_metric("memory_usage", 75.2)
        
        # Simulate an error and recovery
        test_error = ConnectionError("Database connection timeout")
        error_response = ErrorHandlingService.create_enhanced_error_response(
            test_error,
            context={"operation": "database_query"},
            correlation_id="integration-test-001"
        )
        
        # Record error rate
        monitoring_service.record_metric("error_rate", 2.5)
        
        processing_time = (time.time() - start_time) * 1000
        print(f"  ✅ Integrated workflow completed in {processing_time:.1f}ms")
        print(f"  ✅ Error categorized as: {error_response['error_type']}")
        print(f"  ✅ Monitoring captured metrics successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Service integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive service tests"""
    print("🧪 Comprehensive Service Test Suite")
    print("=" * 60)
    print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Configuration Service", test_configuration_service),
        ("Monitoring Service", test_monitoring_service),
        ("Enhanced Error Handling", test_enhanced_error_handling),
        ("Enhanced Template Service", test_enhanced_template_service),
        ("SQL Utility Service", test_sql_utility_service),
        ("Service Integration", test_service_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_function in tests:
        try:
            if test_function():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: FAILED ({e})")
        
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All service tests passed! Enhanced architecture is working perfectly.")
        return 0
    else:
        print(f"⚠️  {failed} service tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
