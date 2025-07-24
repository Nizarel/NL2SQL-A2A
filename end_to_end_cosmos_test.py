#!/usr/bin/env python3
"""
End-to-End Cosmos DB Conversation Logging Test
Validates that conversation logs are properly stored with formatted results and summary data
"""

import asyncio
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import NL2SQLMultiAgentSystem
from services.cosmos_db_service import CosmosDbService
from services.orchestrator_memory_service import OrchestratorMemoryService


class CosmosTestResult:
    """Track test results for each question"""
    def __init__(self, question: str):
        self.question = question
        self.success = False
        self.execution_time = 0
        self.has_formatted_results = False
        self.formatted_results_rows = 0
        self.has_summary = False
        self.insights_count = 0
        self.recommendations_count = 0
        self.conversation_log_id = None
        self.error = None


class EndToEndCosmosTest:
    """End-to-end test for Cosmos DB conversation logging"""
    
    def __init__(self):
        self.nl2sql_system = None
        self.cosmos_service = None
        self.memory_service = None
        self.test_results = []
        self.session_id = f"e2e_test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.user_id = "e2e_test_user"
        
        # Test questions covering various business scenarios
        self.test_questions = [
            "Show me the top 5 customers by revenue in May 2025",
            "What are the best selling products in terms of volume? in May 2025",
            "Analyze revenue by region and show which region performs best in 2025",
            "Show top 3 customers by revenue with their details in March 2025",
            "Show the top performing distribution centers (CEDIs) by total sales in 2025",
            "Analyze Norte region profit performance by showing the top products for CEDIs in that region in 2025",
            "Generate a query to find customers who haven't made purchases in the last 6 months",
            "Which products have declining sales trends and in which regions in May 2025"
        ]
    
    async def initialize(self):
        """Initialize the NL2SQL system and memory service"""
        print("🔧 Initializing End-to-End Test Environment...")
        print("=" * 70)
        
        # Load environment
        load_dotenv(os.path.join("src", ".env"))
        
        # Initialize NL2SQL system
        print("🤖 Initializing NL2SQL Multi-Agent System...")
        self.nl2sql_system = NL2SQLMultiAgentSystem()
        await self.nl2sql_system.initialize()
        
        # Set up Cosmos DB and memory service
        print("🗄️ Setting up Cosmos DB and memory service...")
        self.cosmos_service = CosmosDbService(
            endpoint="https://cosmos-acrasalesanalytics2.documents.azure.com:443/",
            database_name="sales_analytics",
            chat_container_name="nl2sql_chatlogs",
            cache_container_name="nl2sql_cache"
        )
        await self.cosmos_service.initialize()
        
        self.memory_service = OrchestratorMemoryService(self.cosmos_service)
        self.nl2sql_system.orchestrator_agent.set_memory_service(self.memory_service)
        
        print("✅ Test environment initialized successfully!")
        print()
    
    async def test_single_question(self, question: str, question_num: int) -> CosmosTestResult:
        """Test a single question and validate Cosmos DB storage"""
        result = CosmosTestResult(question)
        
        print(f"📋 Question {question_num}: {question}")
        print("-" * 70)
        
        try:
            start_time = time.time()
            
            # Process the question
            print("🔄 Processing...")
            orchestrator_result = await self.nl2sql_system.orchestrator_agent.process({
                "question": question,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "execute": True,
                "limit": 100,
                "include_summary": True,
                "enable_conversation_logging": True
            })
            
            result.execution_time = round(time.time() - start_time, 2)
            result.success = orchestrator_result.get("success", False)
            
            if result.success:
                print(f"✅ Query executed successfully in {result.execution_time}s")
                
                # Check formatted results
                data = orchestrator_result.get("data", {})
                formatted_results = data.get("formatted_results", {})
                
                if formatted_results and formatted_results.get("rows"):
                    result.has_formatted_results = True
                    result.formatted_results_rows = len(formatted_results["rows"])
                    print(f"📊 Formatted Results: ✅ {result.formatted_results_rows} rows")
                else:
                    print("📊 Formatted Results: ❌ Missing")
                
                # Check summary
                summary = data.get("summary", {})
                if summary:
                    result.has_summary = True
                    result.insights_count = len(summary.get("key_insights", []))
                    result.recommendations_count = len(summary.get("recommendations", []))
                    print(f"📋 Summary: ✅ Present")
                    print(f"💡 Insights: {result.insights_count}")
                    print(f"🎯 Recommendations: {result.recommendations_count}")
                else:
                    print("📋 Summary: ❌ Missing")
                
                # Validate Cosmos DB storage
                print("🔍 Validating Cosmos DB storage...")
                await self._validate_cosmos_storage(result)
                
            else:
                result.error = orchestrator_result.get("error", "Unknown error")
                print(f"❌ Query failed: {result.error}")
        
        except Exception as e:
            result.error = str(e)
            print(f"❌ Test failed: {result.error}")
        
        print()
        return result
    
    async def _validate_cosmos_storage(self, result: CosmosTestResult):
        """Validate that conversation log was properly stored in Cosmos DB"""
        try:
            # Get the latest conversation log for this session
            conversations = await self.memory_service.get_user_conversation_history(
                user_id=self.user_id,
                session_id=self.session_id,
                limit=1
            )
            
            if conversations:
                conv = conversations[0]
                result.conversation_log_id = conv.id
                
                # Validate formatted results
                if conv.formatted_results and conv.formatted_results.rows:
                    stored_rows = len(conv.formatted_results.rows)
                    if stored_rows == result.formatted_results_rows:
                        print(f"✅ Cosmos Storage - Formatted Results: {stored_rows} rows")
                    else:
                        print(f"⚠️ Cosmos Storage - Row count mismatch: expected {result.formatted_results_rows}, got {stored_rows}")
                else:
                    print("❌ Cosmos Storage - No formatted results found")
                    result.has_formatted_results = False
                
                # Validate agent response
                if conv.agent_response:
                    stored_insights = len(conv.agent_response.key_insights or [])
                    stored_recommendations = len(conv.agent_response.recommendations or [])
                    
                    print(f"✅ Cosmos Storage - Agent Response: Present")
                    print(f"  📝 Executive Summary: {'✅' if conv.agent_response.executive_summary else '❌'}")
                    print(f"  💡 Insights: {stored_insights}")
                    print(f"  🎯 Recommendations: {stored_recommendations}")
                    
                    # Update counts with actual stored values
                    if stored_insights != result.insights_count:
                        print(f"⚠️ Insight count mismatch: expected {result.insights_count}, stored {stored_insights}")
                    if stored_recommendations != result.recommendations_count:
                        print(f"⚠️ Recommendation count mismatch: expected {result.recommendations_count}, stored {stored_recommendations}")
                else:
                    print("❌ Cosmos Storage - No agent response found")
                    result.has_summary = False
                
                # Validate performance data
                if conv.performance:
                    print(f"✅ Cosmos Storage - Performance: {conv.performance.processing_time_ms}ms")
                else:
                    print("❌ Cosmos Storage - No performance data")
                
            else:
                print("❌ Cosmos Storage - No conversation log found")
                result.conversation_log_id = None
                result.has_formatted_results = False
                result.has_summary = False
        
        except Exception as e:
            print(f"❌ Cosmos validation failed: {str(e)}")
            result.error = f"Cosmos validation error: {str(e)}"
    
    async def run_all_tests(self):
        """Run all test questions and generate summary report"""
        print(f"🚀 Starting End-to-End Cosmos DB Test")
        print(f"👤 User: {self.user_id}")
        print(f"📅 Session: {self.session_id}")
        print(f"📊 Total Questions: {len(self.test_questions)}")
        print("=" * 70)
        print()
        
        # Run each test question
        for i, question in enumerate(self.test_questions, 1):
            result = await self.test_single_question(question, i)
            self.test_results.append(result)
            
            # Brief pause between tests
            if i < len(self.test_questions):
                print("⏳ Pausing 2 seconds before next test...")
                await asyncio.sleep(2)
                print()
        
        # Generate summary report
        await self._generate_summary_report()
    
    async def _generate_summary_report(self):
        """Generate comprehensive test summary report"""
        print("📊 END-TO-END TEST SUMMARY REPORT")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        successful_queries = sum(1 for r in self.test_results if r.success)
        with_formatted_results = sum(1 for r in self.test_results if r.has_formatted_results)
        with_summary = sum(1 for r in self.test_results if r.has_summary)
        with_cosmos_logs = sum(1 for r in self.test_results if r.conversation_log_id)
        
        print(f"📈 Overall Statistics:")
        print(f"  🎯 Total Tests: {total_tests}")
        print(f"  ✅ Successful Queries: {successful_queries}/{total_tests} ({successful_queries/total_tests*100:.1f}%)")
        print(f"  📊 With Formatted Results: {with_formatted_results}/{total_tests} ({with_formatted_results/total_tests*100:.1f}%)")
        print(f"  📋 With Summary Data: {with_summary}/{total_tests} ({with_summary/total_tests*100:.1f}%)")
        print(f"  🗄️ Stored in Cosmos DB: {with_cosmos_logs}/{total_tests} ({with_cosmos_logs/total_tests*100:.1f}%)")
        print()
        
        # Performance metrics
        avg_time = sum(r.execution_time for r in self.test_results) / total_tests
        total_rows = sum(r.formatted_results_rows for r in self.test_results)
        total_insights = sum(r.insights_count for r in self.test_results)
        total_recommendations = sum(r.recommendations_count for r in self.test_results)
        
        print(f"⚡ Performance Metrics:")
        print(f"  ⏱️ Average Execution Time: {avg_time:.2f}s")
        print(f"  📊 Total Data Rows Retrieved: {total_rows}")
        print(f"  💡 Total Insights Generated: {total_insights}")
        print(f"  🎯 Total Recommendations: {total_recommendations}")
        print()
        
        # Detailed results table
        print(f"📋 Detailed Results:")
        print(f"{'#':<3} {'Success':<8} {'Rows':<6} {'Insights':<9} {'Recs':<5} {'Time':<7} {'Cosmos':<8}")
        print("-" * 70)
        
        for i, result in enumerate(self.test_results, 1):
            success = "✅" if result.success else "❌"
            cosmos = "✅" if result.conversation_log_id else "❌"
            
            print(f"{i:<3} {success:<8} {result.formatted_results_rows:<6} "
                  f"{result.insights_count:<9} {result.recommendations_count:<5} "
                  f"{result.execution_time:<7.2f} {cosmos:<8}")
        
        print()
        
        # Failed tests details
        failed_tests = [r for r in self.test_results if not r.success or not r.conversation_log_id]
        if failed_tests:
            print(f"❌ Failed Tests ({len(failed_tests)}):")
            for i, result in enumerate(failed_tests, 1):
                print(f"  {i}. {result.question[:50]}...")
                if result.error:
                    print(f"     Error: {result.error}")
                if not result.conversation_log_id:
                    print(f"     Issue: Cosmos DB storage failed")
            print()
        
        # Success summary
        if successful_queries == total_tests and with_cosmos_logs == total_tests:
            print("🎉 ALL TESTS PASSED! Cosmos DB conversation logging is working perfectly!")
        elif successful_queries == total_tests:
            print("⚠️ All queries succeeded but some Cosmos DB storage issues detected.")
        else:
            print(f"⚠️ {total_tests - successful_queries} query failures detected. Please review logs.")
        
        print()
        print(f"🔗 Session ID for detailed investigation: {self.session_id}")
        print(f"👤 User ID: {self.user_id}")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.nl2sql_system:
            await self.nl2sql_system.close()
        print("🔐 Test environment cleaned up")


async def main():
    """Main test execution"""
    test = EndToEndCosmosTest()
    
    try:
        await test.initialize()
        await test.run_all_tests()
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        await test.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
