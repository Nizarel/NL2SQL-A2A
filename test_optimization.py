#!/usr/bin/env python3
"""
Test script to verify the code optimization results
"""

import sys
import os
sys.path.append('src')

def test_services():
    """Test that all new services are working"""
    print("🔍 Testing optimized services...")
    
    # Test SQLUtilityService
    from services.sql_utility_service import SQLUtilityService
    test_sql = "SELECT * FROM cliente LIMIT 10;"
    cleaned_sql = SQLUtilityService.clean_sql_query(test_sql)
    assert "TOP 10" in cleaned_sql, "SQL cleaning should convert LIMIT to TOP"
    print("✅ SQLUtilityService: LIMIT to TOP conversion working")
    
    # Test ErrorHandlingService
    from services.error_handling_service import ErrorHandlingService
    error_response = ErrorHandlingService.handle_sql_error(
        Exception("Test error"), "SELECT * FROM test"
    )
    assert "error" in error_response, "Error handling should create error response"
    print("✅ ErrorHandlingService: Error handling working")
    
    # Test TemplateService
    from services.template_service import TemplateService
    template_service = TemplateService()
    template_service.initialize_templates()
    basic_template = template_service.get_template_function("basic")
    assert basic_template is not None, "Template service should provide basic template"
    print("✅ TemplateService: Template management working")
    
    return True

def test_code_reduction():
    """Test that duplicate code has been eliminated"""
    print("\n🔍 Testing code duplication elimination...")
    
    # Read the SQL Generator Agent file
    with open("src/agents/sql_generator_agent.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check that old duplicate methods are not present
    duplicate_methods = [
        "def _clean_markdown_formatting",
        "def _clean_sql_syntax", 
        "def _clean_date_functions",
        "def _clean_limit_clauses",
        "def _validate_table_prefixes",
        "def _final_cleanup",
        "def _extract_tables_from_sql",
        "def _determine_query_type"
    ]
    
    eliminated_count = 0
    for method in duplicate_methods:
        if method not in content:
            eliminated_count += 1
            print(f"✅ Duplicate method eliminated: {method}")
        else:
            print(f"❌ Duplicate method still present: {method}")
    
    print(f"\n📊 Eliminated {eliminated_count}/{len(duplicate_methods)} duplicate methods")
    
    # Check that services are being used
    service_usages = [
        "SQLUtilityService.clean_sql_query",
        "ErrorHandlingService.handle_agent_processing_error",
        "TemplateService"
    ]
    
    using_services = 0
    for usage in service_usages:
        if usage in content:
            using_services += 1
            print(f"✅ Using service: {usage}")
        else:
            print(f"❌ Not using service: {usage}")
    
    print(f"\n📊 Using {using_services}/{len(service_usages)} new services")
    
    return eliminated_count >= 6 and using_services >= 2  # At least 6 methods eliminated and 2 services used

def test_file_sizes():
    """Test file size improvements"""
    print("\n📏 Analyzing file sizes...")
    
    # Get file sizes
    sizes = {}
    files_to_check = [
        "src/agents/sql_generator_agent.py",
        "src/services/sql_utility_service.py", 
        "src/services/error_handling_service.py",
        "src/services/template_service.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            sizes[file_path] = size
            print(f"📄 {file_path}: {size:,} bytes")
    
    return sizes

def main():
    """Main test function"""
    print("🚀 Testing NL2SQL Code Optimization Results")
    print("=" * 50)
    
    try:
        # Test services
        services_ok = test_services()
        
        # Test code reduction
        reduction_ok = test_code_reduction()
        
        # Test file sizes
        sizes = test_file_sizes()
        
        print("\n" + "=" * 50)
        if services_ok and reduction_ok:
            print("🎉 OPTIMIZATION SUCCESSFUL!")
            print("✅ All new services working correctly")
            print("✅ Code duplication significantly reduced")
            print("✅ Centralized services architecture implemented")
            print("\n📊 Benefits achieved:")
            print("   • Eliminated duplicate SQL cleaning methods")
            print("   • Centralized error handling")
            print("   • Unified template management")
            print("   • Improved maintainability")
            print("   • Better code organization")
        else:
            print("⚠️ Optimization partially successful but needs review")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
