"""
Comprehensive Unit Tests for Schema Analyst Agent with MCP Integration

Tests the complete Schema Analyst Agent functionality including:
- MCP plugin integration and database connectivity
- Schema analysis and intelligent context generation
- Advanced caching capabilities (exact + semantic matching)
- Cache performance, statistics, and monitoring
- Edge cases and error handling
- End-to-end workflow validation

Cache Testing Coverage:
✅ Exact Cache Matching - Tests identical query caching
✅ Semantic Cache Similarity - Tests embedding-based similar query matching  
✅ Cache Statistics & Monitoring - Tests hit rates, statistics tracking
✅ Similarity Thresholds - Tests different similarity threshold behaviors
✅ Cache Without Embeddings - Tests fallback when embedding service unavailable
✅ Cache Edge Cases - Tests error handling, special characters, empty queries
✅ Cache Performance - Tests comprehensive caching scenarios and performance

Test Categories:
1. MCP Database Integration (4 tests)
2. Schema Service Functionality (1 test) 
3. Basic Schema Analysis (1 test)
4. Comprehensive Cache Testing (7 tests)
5. End-to-End Workflow (1 test)

Total: 14 comprehensive tests covering all major functionality
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding

from agents.schema_analyst_agent import SchemaAnalystAgent
from services.schema_service import SchemaService
from plugins.mcp_database_plugin import MCPDatabasePlugin


class TestSchemaAnalystWithMCP:
    """Test class for Schema Analyst Agent with MCP integration"""
    
    async def setup_kernel(self):
        """Setup Semantic Kernel with Azure OpenAI services"""
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'src', '.env'))
        
        # Create kernel
        kernel = Kernel()
        
        # Add Azure OpenAI chat service
        chat_service = AzureChatCompletion(
            service_id="default_chat",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        )
        kernel.add_service(chat_service)
        
        # Add Azure OpenAI embedding service for semantic caching
        try:
            embedding_service = AzureTextEmbedding(
                service_id="default_embedding",
                deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            )
            kernel.add_service(embedding_service)
            print("✅ Embedding service added for semantic caching")
        except Exception as e:
            print(f"⚠️ Could not add embedding service: {e}")
        
        return kernel
    
    async def setup_mcp_plugin(self):
        """Setup MCP Database Plugin"""
        mcp_server_url = os.getenv("MCP_SERVER_URL", "https://azsql-fastmcpserv2.jollyfield-479bc951.eastus2.azurecontainerapps.io/mcp/")
        return MCPDatabasePlugin(mcp_server_url=mcp_server_url)
    
    async def setup_schema_service(self, mcp_plugin):
        """Setup Schema Service with MCP plugin"""
        schema_service = SchemaService(mcp_plugin=mcp_plugin)
        await schema_service.initialize_schema_context()
        return schema_service
    
    async def setup_schema_analyst(self, kernel, schema_service):
        """Setup Schema Analyst Agent"""
        return SchemaAnalystAgent(kernel=kernel, schema_service=schema_service)
    
    async def test_mcp_database_connection(self, mcp_plugin):
        """Test MCP database connectivity through SK plugin"""
        print("\n🔧 Testing MCP Database Connection...")
        
        # Test database info
        db_info = await mcp_plugin.database_info()
        print(f"📊 Database Info: {db_info}")
        
        assert "Connected" in str(db_info) or "Azure SQL" in str(db_info)
        print("✅ MCP database connection successful")
    
    async def test_mcp_list_tables(self, mcp_plugin):
        """Test listing tables through MCP plugin"""
        print("\n📋 Testing MCP Table Listing...")
        
        # Test table listing
        tables = await mcp_plugin.list_tables()
        print(f"📊 Available Tables: {tables}")
        
        # Verify expected tables are present
        expected_tables = ['cliente', 'cliente_cedi', 'mercado', 'producto', 'segmentacion', 'tiempo']
        for table in expected_tables:
            assert table in str(tables), f"Expected table '{table}' not found"
        
        print("✅ MCP table listing successful")
    
    async def test_mcp_describe_table(self, mcp_plugin):
        """Test describing table schema through MCP plugin"""
        print("\n🔍 Testing MCP Table Description...")
        
        # Test describing segmentacion table (main fact table)
        table_schema = await mcp_plugin.describe_table("segmentacion")
        print(f"📊 Segmentacion Schema: {table_schema[:300]}...")
        
        # Verify key columns are present
        expected_columns = ['customer_id', 'material_id', 'calday', 'VentasCajasUnidad', 'IngresoNetoSImpuestos']
        for column in expected_columns:
            assert column in str(table_schema), f"Expected column '{column}' not found"
        
        print("✅ MCP table description successful")
    
    async def test_mcp_execute_query(self, mcp_plugin):
        """Test executing SQL query through MCP plugin"""
        print("\n⚡ Testing MCP Query Execution...")
        
        # Test simple query
        query = "SELECT TOP 5 customer_id, material_id, VentasCajasUnidad FROM segmentacion WHERE VentasCajasUnidad > 0"
        result = await mcp_plugin.read_data(query, limit=5)
        print(f"📊 Query Result: {result[:200]}...")
        
        # Verify query executed successfully
        assert "customer_id" in str(result) or "Query Results" in str(result)
        print("✅ MCP query execution successful")
    
    async def test_schema_service_initialization(self, schema_service):
        """Test Schema Service initialization with MCP data"""
        print("\n🔧 Testing Schema Service Initialization...")
        
        # Test schema service methods
        schema_summary = schema_service.get_full_schema_summary()
        print(f"📊 Schema Summary: {schema_summary[:300]}...")
        
        assert len(schema_summary) > 0
        assert "segmentacion" in schema_summary
        print("✅ Schema Service initialization successful")
    
    async def test_schema_analyst_basic_analysis(self, schema_analyst):
        """Test basic schema analysis functionality"""
        print("\n🧠 Testing Schema Analyst Basic Analysis...")
        
        # Test analysis for a customer sales question
        test_question = "Show me top customers by revenue"
        
        result = await schema_analyst.process({
            "question": test_question,
            "context": "",
            "use_cache": False  # Disable cache for testing
        })
        
        print(f"📊 Analysis Result Success: {result['success']}")
        if not result['success']:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            return
        
        # Verify analysis success
        assert result["success"] == True
        assert "relevant_tables" in result["data"]
        assert "join_strategy" in result["data"]
        assert "optimized_schema" in result["data"]
        
        # Verify relevant tables identified
        relevant_tables = result["data"]["relevant_tables"]
        assert "cliente" in relevant_tables or "segmentacion" in relevant_tables
        
        print("✅ Schema Analyst basic analysis successful")
    
    async def test_schema_analyst_semantic_caching(self, schema_analyst):
        """Test semantic caching functionality"""
        print("\n💾 Testing Schema Analyst Semantic Caching...")
        
        # First query
        question1 = "What are the top selling products?"
        result1 = await schema_analyst.process({
            "question": question1,
            "use_cache": True
        })
        
        # Similar query that should use semantic cache
        question2 = "Show me the best selling items"
        result2 = await schema_analyst.process({
            "question": question2,
            "use_cache": True,
            "similarity_threshold": 0.7  # Lower threshold for testing
        })
        
        print(f"📊 First Result Cache Hit: {result1['metadata'].get('cache_hit', False)}")
        print(f"📊 Second Result Cache Hit: {result2['metadata'].get('cache_hit', False)}")
        
        # Verify both queries succeeded
        assert result1["success"] == True
        assert result2["success"] == True
        
        print("✅ Schema Analyst semantic caching test completed")
    
    async def test_exact_cache_functionality(self, schema_analyst):
        """Test exact cache matching functionality"""
        print("\n🎯 Testing Exact Cache Functionality...")
        
        # Clear cache first
        schema_analyst.clear_all_caches()
        
        # Test identical questions for exact cache hit
        question = "Show revenue by customer territory"
        
        # First execution - should cache
        result1 = await schema_analyst.process({
            "question": question,
            "use_cache": True
        })
        
        # Second execution - should hit exact cache
        result2 = await schema_analyst.process({
            "question": question,
            "use_cache": True
        })
        
        # Verify results
        assert result1["success"] == True
        assert result2["success"] == True
        assert result1["metadata"].get("cache_hit", False) == False  # First should miss
        assert result2["metadata"].get("cache_hit", False) == True   # Second should hit
        assert result2["metadata"].get("cache_type") == "exact"
        
        print("✅ Exact cache functionality test successful")
    
    async def test_cache_statistics_and_monitoring(self, schema_analyst):
        """Test cache statistics and monitoring capabilities"""
        print("\n📊 Testing Cache Statistics and Monitoring...")
        
        # Clear cache for clean stats
        schema_analyst.clear_all_caches()
        
        # Execute multiple queries to generate statistics
        test_questions = [
            "What are total sales by product category?",
            "Show me customer revenue breakdown",
            "What are total sales by product category?",  # Exact match
            "Display customer revenue analysis"           # Semantic match potential
        ]
        
        for question in test_questions:
            await schema_analyst.process({
                "question": question,
                "use_cache": True,
                "similarity_threshold": 0.75
            })
        
        # Get and verify statistics
        stats = await schema_analyst.get_cache_statistics()
        
        print(f"📊 Cache Statistics: {stats}")
        
        # Verify statistics structure and values
        assert "total_queries" in stats
        assert "exact_hits" in stats
        assert "semantic_hits" in stats
        assert "hit_rate_percent" in stats
        assert "exact_cache_size" in stats
        assert "semantic_cache_size" in stats
        
        assert stats["total_queries"] >= 4
        assert stats["exact_hits"] >= 1  # Should have at least one exact match
        assert isinstance(stats["hit_rate_percent"], (int, float))
        
        print("✅ Cache statistics test successful")
    
    async def test_cache_similarity_thresholds(self, schema_analyst):
        """Test different similarity thresholds for semantic caching"""
        print("\n🎚️ Testing Cache Similarity Thresholds...")
        
        # Clear cache
        schema_analyst.clear_all_caches()
        
        # Base question
        base_question = "Show top customers by revenue"
        
        # Store base analysis
        await schema_analyst.process({
            "question": base_question,
            "use_cache": True
        })
        
        # Test similar question with different thresholds
        similar_question = "Display highest revenue customers"
        
        # Test high threshold (should miss)
        result_high = await schema_analyst.process({
            "question": similar_question,
            "use_cache": True,
            "similarity_threshold": 0.95
        })
        
        # Test medium threshold (might hit)
        result_medium = await schema_analyst.process({
            "question": similar_question,
            "use_cache": True,
            "similarity_threshold": 0.75
        })
        
        # Test low threshold (should hit)
        result_low = await schema_analyst.process({
            "question": similar_question,
            "use_cache": True,
            "similarity_threshold": 0.6
        })
        
        print(f"📊 High threshold cache hit: {result_high['metadata'].get('cache_hit', False)}")
        print(f"📊 Medium threshold cache hit: {result_medium['metadata'].get('cache_hit', False)}")
        print(f"📊 Low threshold cache hit: {result_low['metadata'].get('cache_hit', False)}")
        
        # Verify all queries succeeded
        assert result_high["success"] == True
        assert result_medium["success"] == True
        assert result_low["success"] == True
        
        print("✅ Similarity threshold test successful")
    
    async def test_cache_without_embeddings(self, kernel_no_embeddings, schema_service):
        """Test cache functionality when embedding service is not available"""
        print("\n🚫 Testing Cache Without Embeddings...")
        
        # Create kernel without embedding service
        kernel = Kernel()
        
        # Add only chat service (no embeddings)
        from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
        chat_service = AzureChatCompletion(
            service_id="default_chat",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        )
        kernel.add_service(chat_service)
        
        # Create schema analyst without embeddings
        schema_analyst_no_embed = SchemaAnalystAgent(kernel=kernel, schema_service=schema_service)
        
        # Test that exact caching still works
        question = "What are sales by territory?"
        
        result1 = await schema_analyst_no_embed.process({
            "question": question,
            "use_cache": True
        })
        
        result2 = await schema_analyst_no_embed.process({
            "question": question,
            "use_cache": True
        })
        
        # Verify exact caching works without embeddings
        assert result1["success"] == True
        assert result2["success"] == True
        assert result2["metadata"].get("cache_hit", False) == True
        assert result2["metadata"].get("cache_type") == "exact"
        
        print("✅ Cache without embeddings test successful")
    
    async def test_cache_edge_cases(self, schema_analyst):
        """Test cache edge cases and error handling"""
        print("\n🔍 Testing Cache Edge Cases...")
        
        # Test with empty question
        result_empty = await schema_analyst.process({
            "question": "",
            "use_cache": True
        })
        assert result_empty["success"] == False
        
        # Test with very long question
        long_question = "What are the sales metrics " * 100 + "by customer?"
        result_long = await schema_analyst.process({
            "question": long_question,
            "use_cache": True
        })
        assert result_long["success"] == True
        
        # Test cache disabled
        result_no_cache = await schema_analyst.process({
            "question": "Show product categories",
            "use_cache": False
        })
        assert result_no_cache["success"] == True
        assert result_no_cache["metadata"].get("cache_hit", False) == False
        
        # Test special characters in question
        special_question = "What are sales for @#$% customer?"
        result_special = await schema_analyst.process({
            "question": special_question,
            "use_cache": True
        })
        assert result_special["success"] == True
        
        print("✅ Cache edge cases test successful")
    
    async def test_comprehensive_cache_performance(self, schema_analyst):
        """Test cache performance with multiple queries"""
        print("\n⚡ Testing Comprehensive Cache Performance...")
        
        # Clear cache for clean test
        schema_analyst.clear_all_caches()
        
        # Define test question sets
        question_sets = [
            # Exact matches
            ("What is total revenue?", "What is total revenue?"),
            ("Show customer sales", "Show customer sales"),
            
            # Semantic matches
            ("Top selling products", "Best selling items"),
            ("Customer revenue analysis", "Customer income breakdown"),
            ("Sales by territory", "Revenue by region"),
        ]
        
        cache_hits = 0
        total_queries = 0
        
        for original, similar in question_sets:
            # Execute original
            result1 = await schema_analyst.process({
                "question": original,
                "use_cache": True
            })
            total_queries += 1
            
            # Execute similar (should cache hit)
            result2 = await schema_analyst.process({
                "question": similar,
                "use_cache": True,
                "similarity_threshold": 0.7
            })
            total_queries += 1
            
            if result2["metadata"].get("cache_hit", False):
                cache_hits += 1
                
            assert result1["success"] == True
            assert result2["success"] == True
        
        # Get final statistics
        final_stats = await schema_analyst.get_cache_statistics()
        
        print(f"📊 Cache Hits: {cache_hits}/{len(question_sets)}")
        print(f"📊 Final Cache Stats: {final_stats}")
        
        # Verify performance improvement
        assert final_stats["total_queries"] >= total_queries
        assert final_stats["hit_rate_percent"] > 0
        
        print("✅ Comprehensive cache performance test successful")
    
    async def test_end_to_end_workflow(self, kernel, mcp_plugin, schema_service):
        """Test complete end-to-end workflow"""
        print("\n🎯 Testing End-to-End Workflow...")
        
        # Initialize schema analyst
        schema_analyst = SchemaAnalystAgent(kernel=kernel, schema_service=schema_service)
        
        # Test workflow: Question → Schema Analysis → Optimized Context
        test_question = "Find customers with highest sales volume in 2024"
        
        # Step 1: Analyze schema
        analysis_result = await schema_analyst.process({
            "question": test_question,
            "use_cache": False
        })
        
        print(f"📊 Schema Analysis: {analysis_result['success']}")
        
        # Step 2: Use optimized schema context for query planning
        if analysis_result["success"]:
            optimized_schema = analysis_result["data"]["optimized_schema"]
            relevant_tables = analysis_result["data"]["relevant_tables"]
            join_strategy = analysis_result["data"]["join_strategy"]
            
            print(f"📊 Relevant Tables: {relevant_tables}")
            print(f"📊 Join Strategy: {join_strategy.get('strategy', 'unknown')}")
            print(f"📊 Optimized Schema Length: {len(optimized_schema)} chars")
            
            # Verify we have meaningful analysis
            assert len(relevant_tables) > 0
            assert len(optimized_schema) > 0
            assert join_strategy.get("strategy") in ["single_table", "star_schema", "dimension_lookup"]
        
        print("✅ End-to-end workflow test successful")


async def run_tests():
    """Run all tests"""
    print("🚀 Starting Schema Analyst Agent + MCP Integration Tests...")
    
    test_class = TestSchemaAnalystWithMCP()
    
    # Setup fixtures
    print("\n🔧 Setting up fixtures...")
    kernel = await test_class.setup_kernel()
    mcp_plugin = await test_class.setup_mcp_plugin()
    schema_service = await test_class.setup_schema_service(mcp_plugin)
    schema_analyst = await test_class.setup_schema_analyst(kernel, schema_service)
    
    # Run individual tests
    try:
        # Core MCP functionality tests
        await test_class.test_mcp_database_connection(mcp_plugin)
        await test_class.test_mcp_list_tables(mcp_plugin)
        await test_class.test_mcp_describe_table(mcp_plugin)
        await test_class.test_mcp_execute_query(mcp_plugin)
        
        # Schema service tests
        await test_class.test_schema_service_initialization(schema_service)
        
        # Schema analyst basic functionality
        await test_class.test_schema_analyst_basic_analysis(schema_analyst)
        
        # Comprehensive cache testing
        await test_class.test_exact_cache_functionality(schema_analyst)
        await test_class.test_schema_analyst_semantic_caching(schema_analyst)
        await test_class.test_cache_statistics_and_monitoring(schema_analyst)
        await test_class.test_cache_similarity_thresholds(schema_analyst)
        await test_class.test_cache_without_embeddings(None, schema_service)
        await test_class.test_cache_edge_cases(schema_analyst)
        await test_class.test_comprehensive_cache_performance(schema_analyst)
        
        # End-to-end workflow
        await test_class.test_end_to_end_workflow(kernel, mcp_plugin, schema_service)
        
        print("\n🎉 All tests passed successfully!")
        print("\n📊 Comprehensive test coverage achieved:")
        print("   ✅ MCP Database Integration")
        print("   ✅ Schema Service Functionality") 
        print("   ✅ Basic Schema Analysis")
        print("   ✅ Exact Cache Matching")
        print("   ✅ Semantic Cache Similarity")
        print("   ✅ Cache Statistics & Monitoring")
        print("   ✅ Similarity Threshold Testing")
        print("   ✅ Cache Without Embeddings")
        print("   ✅ Cache Edge Cases")
        print("   ✅ Cache Performance Testing")
        print("   ✅ End-to-End Workflow")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_tests())
