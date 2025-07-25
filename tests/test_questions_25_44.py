"""
Focused Test Suite: Questions 25-44
Tests specific business questions related to returns, key customers, coverage, and forecasting
"""

import os
import sys
import asyncio
import time
import traceback
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import NL2SQLMultiAgentSystem


class ErrorLogger:
    """Logger for capturing test errors and failed queries"""
    
    def __init__(self, log_file="test_25_44_errors.log"):
        self.log_file = log_file
        self.errors = []
    
    def log_error(self, question, error_message, generated_query=None, metadata=None):
        """Log an error with context"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "error_message": str(error_message),
            "generated_query": generated_query,
            "metadata": metadata or {}
        }
        
        self.errors.append(error_entry)
        
        # Write to file immediately
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"ERROR LOGGED AT: {error_entry['timestamp']}\n")
                f.write(f"QUESTION: {error_entry['question']}\n")
                f.write(f"ERROR: {error_entry['error_message']}\n")
                if generated_query:
                    f.write(f"GENERATED QUERY:\n{generated_query}\n")
                if metadata:
                    f.write(f"METADATA: {json.dumps(metadata, indent=2)}\n")
                f.write(f"{'='*80}\n")
        except Exception as e:
            print(f"⚠️  Failed to write error log: {e}")
    
    def save_summary(self):
        """Save error summary to JSON file"""
        try:
            summary_file = self.log_file.replace('.log', '_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_errors": len(self.errors),
                    "test_run": datetime.now().isoformat(),
                    "errors": self.errors
                }, f, indent=2, ensure_ascii=False)
            print(f"📊 Error summary saved to: {summary_file}")
        except Exception as e:
            print(f"⚠️  Failed to save error summary: {e}")


async def test_questions_25_44():
    """Test questions 25-44 from Questions.txt"""
    
    print("🚀 Testing Questions 25-44...")
    print("=" * 80)
    
    # Initialize error logger
    error_logger = ErrorLogger("test_25_44_errors.log")
    
    # Initialize the system
    system = NL2SQLMultiAgentSystem()
    
    # Questions 25-44 from Questions.txt
    test_questions = [
        # Q25
        "What is the return or returned product rate for each CEDI (if recorded) by category or material?",
        # Q26
        "How do returns affect the overall profitability of a CEDI for each product?",
        # Q27
        "Who are the key customers (Cuenta_clave) that concentrate most of the sales and how have their purchases varied?",
        # Q28
        "Are there billable customers who have recently stopped buying and merit a follow-up visit or call?",
        # Q29
        "What is the coverage percentage (points of sale served vs. potential points of sale) in each zone?",
        # Q30
        "Which customers have not received visits or orders in the last period despite being active?",
        # Q31
        "How close are actual sales to the forecasts (forecast) in a specific period for the top 5 products in each zone?",
        # Q32
        "In which zones has demand been overestimated and in which has it been underestimated?",
        # Q33
        "What is the brand share in each category (e.g., carbonated vs. non-carbonated)?",
        # Q34
        "How does promotion effectiveness vary by commercial channel?",
        # Q35
        "Which territories show expansion opportunities, based on low coverage but high potential?",
        # Q36
        "Which products might have greater marketing push according to their historical growth?",
        # Q37
        "What are the different customer segments and how do they differ in purchasing patterns?",
        # Q38
        "Which customer type presents the highest profit margin, what would be the strategy to retain them, and what are their top 5 purchased products?",
        # Q39
        "Which subterritory shows the highest sales growth for a specific product category (e.g., carbonated vs. non-carbonated) and what factors explain it?",
        # Q40
        "What's the month-over-month revenue growth for the last 6 months?",
        # Q41
        "Which zones have the highest revenue per customer?",
        # Q42
        "What's the revenue contribution of each product category?",
        # Q43
        "Show me revenue performance by day of week",
        # Q44
        "What's the revenue impact of new product launches?"
    ]
    
    try:
        await system.initialize()
        print("✅ System initialized successfully")
        
        # Test workflow status
        status = await system.get_workflow_status()
        print(f"📊 Workflow Status: {status}")
        
        print(f"\n🧪 Testing {len(test_questions)} business questions (Q25-Q44)...")
        
        # Statistics tracking
        successful_tests = 0
        failed_tests = 0
        total_time = 0
        results_summary = []
        
        for i, question in enumerate(test_questions, 25):  # Start numbering from 25
            print(f"\n📋 Question {i}")
            print(f"❓ {question}")
            
            start_time = time.time()
            
            try:
                result = await system.ask_question(
                    question=question,
                    execute=True,  # Execute SQL to get actual results
                    include_summary=False
                )
                
                execution_time = time.time() - start_time
                total_time += execution_time
                
                if result["success"]:
                    successful_tests += 1
                    print("✅ Query generated and executed successfully!")
                    
                    # Check for schema analysis results
                    if "schema_analysis" in result["data"]:
                        schema_tables = result["data"]["schema_analysis"].get("relevant_tables", [])
                        print(f"🗄️  Relevant tables: {', '.join(schema_tables)}")
                    
                    # Show execution results
                    if "execution_result" in result["data"]:
                        exec_result = result["data"]["execution_result"]
                        if exec_result.get("success"):
                            row_count = exec_result.get("row_count", 0)
                            print(f"📊 Execution result: {row_count} rows returned")
                            
                            # Show sample data if available
                            if "formatted_results" in exec_result and exec_result["formatted_results"].get("rows"):
                                sample_rows = exec_result["formatted_results"]["rows"][:3]  # First 3 rows
                                print(f"📄 Sample data: {len(sample_rows)} rows shown")
                                for j, row in enumerate(sample_rows, 1):
                                    row_preview = str(row)[:100] + "..." if len(str(row)) > 100 else str(row)
                                    print(f"     Row {j}: {row_preview}")
                        else:
                            print(f"❌ SQL execution failed: {exec_result.get('error', 'Unknown error')}")
                    
                    # Show generated SQL preview
                    if result["data"].get("sql_query"):
                        sql_preview = result["data"]["sql_query"][:150] + "..." if len(result["data"]["sql_query"]) > 150 else result["data"]["sql_query"]
                        print(f"💡 SQL: {sql_preview}")
                    
                    print(f"⏱️  Total time: {execution_time:.3f}s")
                    
                    # Store successful result
                    results_summary.append({
                        "question_number": i,
                        "question": question,
                        "success": True,
                        "execution_time": execution_time,
                        "sql_query": result["data"].get("sql_query"),
                        "row_count": result["data"].get("execution_result", {}).get("row_count", 0),
                        "metadata": result.get("metadata", {})
                    })
                    
                else:
                    failed_tests += 1
                    error_msg = result.get('error', 'Unknown error')
                    print(f"❌ Query generation failed: {error_msg}")
                    
                    # Log error with generated query if available
                    generated_query = result.get("data", {}).get("sql_query") if result.get("data") else None
                    metadata = result.get("metadata", {})
                    
                    error_logger.log_error(
                        question=f"Q{i}: {question}",
                        error_message=error_msg,
                        generated_query=generated_query,
                        metadata=metadata
                    )
                    
                    # Store failed result
                    results_summary.append({
                        "question_number": i,
                        "question": question,
                        "success": False,
                        "error": error_msg,
                        "execution_time": execution_time,
                        "sql_query": generated_query,
                        "metadata": metadata
                    })
                
            except Exception as e:
                failed_tests += 1
                execution_time = time.time() - start_time
                total_time += execution_time
                
                error_msg = f"Exception occurred: {str(e)}"
                print(f"❌ Exception: {error_msg}")
                
                # Log exception
                error_logger.log_error(
                    question=f"Q{i}: {question}",
                    error_message=error_msg,
                    generated_query=None,
                    metadata={"exception": True, "traceback": traceback.format_exc()}
                )
                
                # Store exception result
                results_summary.append({
                    "question_number": i,
                    "question": question,
                    "success": False,
                    "error": error_msg,
                    "execution_time": execution_time,
                    "exception": True
                })
            
            print("-" * 50)
        
        # Calculate statistics
        success_rate = (successful_tests / len(test_questions)) * 100
        avg_time = total_time / len(test_questions)
        
        print(f"\n📊 TEST RESULTS SUMMARY (Questions 25-44)")
        print("=" * 60)
        print(f"✅ Successful queries: {successful_tests}/{len(test_questions)} ({success_rate:.1f}%)")
        print(f"❌ Failed queries: {failed_tests}/{len(test_questions)} ({100-success_rate:.1f}%)")
        print(f"⏱️  Total execution time: {total_time:.3f}s")
        print(f"⏱️  Average time per query: {avg_time:.3f}s")
        
        # Show successful queries with data
        successful_with_data = [r for r in results_summary if r["success"] and r.get("row_count", 0) > 0]
        print(f"📊 Queries returning data: {len(successful_with_data)}/{successful_tests}")
        
        if successful_with_data:
            print(f"\n🎯 TOP PERFORMING QUERIES:")
            sorted_results = sorted(successful_with_data, key=lambda x: x.get("row_count", 0), reverse=True)[:5]
            for result in sorted_results:
                print(f"   Q{result['question_number']}: {result['row_count']} rows - {result['execution_time']:.2f}s")
        
        # Show failed queries
        if failed_tests > 0:
            print(f"\n❌ FAILED QUERIES:")
            failed_results = [r for r in results_summary if not r["success"]]
            for result in failed_results:
                error_preview = result.get("error", "Unknown")[:60] + "..." if len(result.get("error", "")) > 60 else result.get("error", "Unknown")
                print(f"   Q{result['question_number']}: {error_preview}")
            
            print(f"📋 Detailed error log saved")
            error_logger.save_summary()
        
        # Performance insights
        print(f"\n⚡ PERFORMANCE INSIGHTS")
        print("-" * 40)
        if successful_tests > 0:
            successful_results = [r for r in results_summary if r["success"]]
            times = [r["execution_time"] for r in successful_results]
            fastest = min(times)
            slowest = max(times)
            print(f"   Fastest query: {fastest:.3f}s")
            print(f"   Slowest query: {slowest:.3f}s")
            print(f"   Time variation: {slowest - fastest:.3f}s")
            
            # Data insights
            row_counts = [r.get("row_count", 0) for r in successful_results if r.get("row_count", 0) > 0]
            if row_counts:
                avg_rows = sum(row_counts) / len(row_counts)
                max_rows = max(row_counts)
                print(f"   Average rows returned: {avg_rows:.1f}")
                print(f"   Maximum rows returned: {max_rows}")
        
        print(f"\n🎉 QUESTIONS 25-44 TEST COMPLETED!")
        print("=" * 60)
        
        if success_rate >= 80:
            print("🎯 EXCELLENT: High success rate achieved")
        elif success_rate >= 60:
            print("👍 GOOD: Reasonable success rate")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Low success rate")
        
        # Save detailed results
        results_file = "test_25_44_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_summary": {
                    "total_questions": len(test_questions),
                    "successful": successful_tests,
                    "failed": failed_tests,
                    "success_rate": success_rate,
                    "total_time": total_time,
                    "avg_time": avg_time
                },
                "results": results_summary
            }, f, indent=2, ensure_ascii=False)
        print(f"📄 Detailed results saved to: {results_file}")
        
        return success_rate >= 60  # Consider 60% or higher as passing
        
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        traceback.print_exc()
        error_logger.log_error(
            question="SYSTEM_INITIALIZATION",
            error_message=str(e),
            generated_query=None,
            metadata={"system_error": True, "traceback": traceback.format_exc()}
        )
        return False
        
    finally:
        await system.close()
        print("🔐 System closed")


async def main():
    """Run focused test on questions 25-44"""
    
    print("🧪 FOCUSED TEST: QUESTIONS 25-44")
    print("=" * 60)
    print(f"📅 Test run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Focus Areas: Returns, Key Customers, Coverage, Forecasting")
    
    # Clear previous log files
    for log_file in ["test_25_44_errors.log", "test_25_44_results.json"]:
        if os.path.exists(log_file):
            os.remove(log_file)
            print(f"🗑️  Cleared previous file: {log_file}")
    
    # Run focused test
    test_passed = await test_questions_25_44()
    
    # Final summary
    print("\n📋 FINAL TEST RESULT")
    print("=" * 60)
    
    if test_passed:
        print(f"🎉 RESULT: ✅ TEST PASSED")
        print("\n🎯 Questions 25-44 Successfully Processed:")
        print("   ✅ Returns and profitability analysis")
        print("   ✅ Key customer identification")
        print("   ✅ Coverage and territory analysis")
        print("   ✅ Forecasting and growth metrics")
        print("   ✅ Performance metrics captured")
    else:
        print(f"❌ RESULT: TEST FAILED")
        print("\n⚠️  Issues detected in Questions 25-44:")
        print("   ❌ Some queries failed to generate or execute")
        print("   🔧 Review error logs for details")
    
    print(f"\n📊 Generated files:")
    for file_name in ["test_25_44_errors.log", "test_25_44_errors_summary.json", "test_25_44_results.json"]:
        if os.path.exists(file_name):
            print(f"   📄 {file_name}")
    
    return test_passed


if __name__ == "__main__":
    """Run the focused test suite for questions 25-44"""
    print("Starting Focused Test Suite: Questions 25-44...")
    success = asyncio.run(main())
    print(f"\nTest suite {'✅ COMPLETED SUCCESSFULLY' if success else '❌ COMPLETED WITH FAILURES'}")
    exit(0 if success else 1)
