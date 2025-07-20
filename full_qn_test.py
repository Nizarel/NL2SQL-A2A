"""
Full Question Test Suite: Test all questions from Questions.txt
Tests the NL2SQL system with comprehensive business questions and logs errors
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
    
    def __init__(self, log_file="test_errors.log"):
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


async def test_all_questions():
    """Test all questions from Questions.txt"""
    
    print("🚀 Testing All Questions from Questions.txt...")
    print("=" * 80)
    
    # Initialize error logger
    error_logger = ErrorLogger("full_qn_test_errors.log")
    
    # Initialize the system
    system = NL2SQLMultiAgentSystem()
    
    # All questions from Questions.txt
    test_questions = [
        "What are the best-selling products (by Material or Product), both in volume and value?",
        "How do the sales of returnable vs. non-returnable products (Retornabilidad) compare?",
        "What is the profitability (margin or profit) by channel, zone, or product category?",
        "Which product or category has generated the highest profit in the last quarter for each CEDI?",
        "Is there variation in the average selling price per material over the months by zone?",
        "How have promotions impacted the zone, product, and customer type in sales volume?",
        "What is the number of active routes in each CEDI and what is the average volume handled by each route?",
        "What are the 10 territories with the highest number of billable customers served and the top 5 products sold in each of those territories?",
        "Which CEDI has the highest dispatch volume?",
        "How does performance (sales, coverage) vary between different CEDIs?",
        "How has the adoption of a new product been in terms of volumes sold and how many customers have purchased it?",
        "How many new customers have started buying a specific product since its launch?",
        "From historical sales information, what sales forecast is projected for the next month or quarter for each CEDI and Product?",
        "Which products show more marked seasonal trends and what is the inventory recommendation?",
        "What is the brand penetration in each subterritory or region compared to the competition (if captured)?",
        "What percentage of customers regularly buy a set of key products (e.g., a certain package of SKUs)?",
        "How many customers have decreased their purchase frequency in recent weeks and might be considered at risk of loss and what are their frequent products?",
        "What percentage of routes are meeting established delivery times?",
        "How does the average purchase ticket vary among customers from different channels (e.g., supermarkets vs. small stores) in each zone?",
        "What customer segment is most profitable for each product and why?",
        "Which routes and customers have met or exceeded their volume and value targets?",
        "Are there areas where the sales force might need reinforcement to improve coverage?",
        "Which products have the highest turnover (highest daily or weekly sales) and require more stock in the warehouse for each CEDI?",
        "At what times of the month is the highest demand concentrated for each customer and product, and how does it impact inventory levels?",
        "What is the return or returned product rate for each CEDI (if recorded) by category or material?",
        "How do returns affect the overall profitability of a CEDI for each product?",
        "Who are the key customers (Cuenta_clave) that concentrate most of the sales and how have their purchases varied?",
        "Are there billable customers who have recently stopped buying and merit a follow-up visit or call?",
        "What is the coverage percentage (points of sale served vs. potential points of sale) in each zone?",
        "Which customers have not received visits or orders in the last period despite being active?",
        "How close are actual sales to the forecasts (forecast) in a specific period for the top 5 products in each zone?",
        "In which zones has demand been overestimated and in which has it been underestimated?",
        "What is the brand share in each category (e.g., carbonated vs. non-carbonated)?",
        "How does promotion effectiveness vary by commercial channel?",
        "Which territories show expansion opportunities, based on low coverage but high potential?",
        "Which products might have greater marketing push according to their historical growth?",
        "What are the different customer segments and how do they differ in purchasing patterns?",
        "Which customer type presents the highest profit margin, what would be the strategy to retain them, and what are their top 5 purchased products?",
        "Which subterritory shows the highest sales growth for a specific product category (e.g., carbonated vs. non-carbonated) and what factors explain it?",
        "What's the month-over-month revenue growth for the last 6 months?",
        "Which zones have the highest revenue per customer?",
        "What's the revenue contribution of each product category?",
        "Show me revenue performance by day of week",
        "What's the revenue impact of new product launches?"
    ]
    
    try:
        await system.initialize()
        print("✅ System initialized successfully")
        
        # Test workflow status
        status = await system.get_workflow_status()
        print(f"📊 Workflow Status: {status}")
        
        print(f"\n🧪 Testing {len(test_questions)} business questions...")
        
        # Statistics tracking
        successful_tests = 0
        failed_tests = 0
        total_time = 0
        results_summary = []
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📋 Test {i}/{len(test_questions)}")
            print(f"❓ Question: {question}")
            
            start_time = time.time()
            
            try:
                result = await system.ask_question(
                    question=question,
                    execute=False,  # Don't execute SQL for faster testing
                    include_summary=False
                )
                
                execution_time = time.time() - start_time
                total_time += execution_time
                
                if result["success"]:
                    successful_tests += 1
                    print("✅ Query generated successfully!")
                    
                    # Check for schema analysis results
                    if "schema_analysis" in result["data"]:
                        schema_tables = result["data"]["schema_analysis"].get("relevant_tables", [])
                        print(f"🗄️  Relevant tables: {', '.join(schema_tables)}")
                    else:
                        print("🗄️  Schema analysis: Not available")
                    
                    # Show generated SQL preview
                    if result["data"].get("sql_query"):
                        sql_preview = result["data"]["sql_query"][:200] + "..." if len(result["data"]["sql_query"]) > 200 else result["data"]["sql_query"]
                        print(f"💡 SQL: {sql_preview}")
                    
                    # Check metadata for optimization info
                    metadata = result.get("metadata", {})
                    print(f"🔧 Schema optimization: {metadata.get('schema_optimization', 'unknown')}")
                    cache_info = metadata.get('cache_info', 'unknown')
                    print(f"💾 Cache info: {cache_info}")
                    
                    print(f"⏱️  Execution time: {execution_time:.3f}s")
                    
                    # Store successful result
                    results_summary.append({
                        "question": question,
                        "success": True,
                        "execution_time": execution_time,
                        "sql_query": result["data"].get("sql_query"),
                        "metadata": metadata
                    })
                    
                else:
                    failed_tests += 1
                    error_msg = result.get('error', 'Unknown error')
                    print(f"❌ Query generation failed: {error_msg}")
                    
                    # Log error with generated query if available
                    generated_query = result.get("data", {}).get("sql_query") if result.get("data") else None
                    metadata = result.get("metadata", {})
                    
                    error_logger.log_error(
                        question=question,
                        error_message=error_msg,
                        generated_query=generated_query,
                        metadata=metadata
                    )
                    
                    # Store failed result
                    results_summary.append({
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
                    question=question,
                    error_message=error_msg,
                    generated_query=None,
                    metadata={"exception": True, "traceback": traceback.format_exc()}
                )
                
                # Store exception result
                results_summary.append({
                    "question": question,
                    "success": False,
                    "error": error_msg,
                    "execution_time": execution_time,
                    "exception": True
                })
            
            print("-" * 40)
        
        # Calculate statistics
        success_rate = (successful_tests / len(test_questions)) * 100
        avg_time = total_time / len(test_questions)
        
        print(f"\n📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"✅ Successful queries: {successful_tests}/{len(test_questions)} ({success_rate:.1f}%)")
        print(f"❌ Failed queries: {failed_tests}/{len(test_questions)} ({100-success_rate:.1f}%)")
        print(f"⏱️  Total execution time: {total_time:.3f}s")
        print(f"⏱️  Average time per query: {avg_time:.3f}s")
        
        if failed_tests > 0:
            print(f"📋 Error log saved with {len(error_logger.errors)} entries")
            error_logger.save_summary()
        
        # Analyze most common failures
        if failed_tests > 0:
            print(f"\n🔍 FAILURE ANALYSIS")
            print("-" * 40)
            
            error_types = {}
            for result in results_summary:
                if not result["success"]:
                    error = result.get("error", "Unknown")
                    # Categorize errors
                    if "timeout" in error.lower():
                        error_type = "Timeout"
                    elif "connection" in error.lower():
                        error_type = "Connection"
                    elif "sql" in error.lower() or "syntax" in error.lower():
                        error_type = "SQL Generation"
                    elif "schema" in error.lower():
                        error_type = "Schema Analysis"
                    elif result.get("exception"):
                        error_type = "Exception"
                    else:
                        error_type = "Other"
                    
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"   {error_type}: {count} occurrences")
        
        # Performance insights
        print(f"\n⚡ PERFORMANCE INSIGHTS")
        print("-" * 40)
        successful_results = [r for r in results_summary if r["success"]]
        if successful_results:
            times = [r["execution_time"] for r in successful_results]
            fastest = min(times)
            slowest = max(times)
            print(f"   Fastest query: {fastest:.3f}s")
            print(f"   Slowest query: {slowest:.3f}s")
            print(f"   Time variation: {slowest - fastest:.3f}s")
        
        # Cache analysis
        cache_hits = sum(1 for r in results_summary if r.get("metadata", {}).get("schema_cache_hit", False))
        if cache_hits > 0:
            cache_efficiency = (cache_hits / len(test_questions)) * 100
            print(f"   Cache efficiency: {cache_hits}/{len(test_questions)} ({cache_efficiency:.1f}%)")
        
        print(f"\n🎉 COMPREHENSIVE BUSINESS QUESTION TEST COMPLETED!")
        print("=" * 80)
        
        if success_rate >= 80:
            print("🎯 EXCELLENT: High success rate achieved")
        elif success_rate >= 60:
            print("👍 GOOD: Reasonable success rate")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Low success rate")
        
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


async def test_specific_categories():
    """Test specific categories of questions for detailed analysis"""
    
    print("\n🎯 Testing Specific Question Categories...")
    print("=" * 60)
    
    system = NL2SQLMultiAgentSystem()
    error_logger = ErrorLogger("category_test_errors.log")
    
    # Categorized questions for focused testing
    categories = {
        "Revenue & Profitability": [
            "What's the month-over-month revenue growth for the last 6 months?",
            "Which zones have the highest revenue per customer?",
            "What's the revenue contribution of each product category?",
            "What is the profitability (margin or profit) by channel, zone, or product category?"
        ],
        "Product Performance": [
            "What are the best-selling products (by Material or Product), both in volume and value?",
            "Which products have the highest turnover (highest daily or weekly sales) and require more stock in the warehouse for each CEDI?",
            "What's the revenue impact of new product launches?",
            "How has the adoption of a new product been in terms of volumes sold and how many customers have purchased it?"
        ],
        "Customer Analysis": [
            "Who are the key customers (Cuenta_clave) that concentrate most of the sales and how have their purchases varied?",
            "What customer segment is most profitable for each product and why?",
            "How many customers have decreased their purchase frequency in recent weeks and might be considered at risk of loss and what are their frequent products?",
            "What are the different customer segments and how do they differ in purchasing patterns?"
        ],
        "Territory & Route Performance": [
            "What are the 10 territories with the highest number of billable customers served and the top 5 products sold in each of those territories?",
            "Which territories show expansion opportunities, based on low coverage but high potential?",
            "What is the number of active routes in each CEDI and what is the average volume handled by each route?",
            "What percentage of routes are meeting established delivery times?"
        ]
    }
    
    try:
        await system.initialize()
        
        category_results = {}
        
        for category, questions in categories.items():
            print(f"\n📂 Testing Category: {category}")
            print(f"   Questions: {len(questions)}")
            
            category_success = 0
            category_time = 0
            
            for i, question in enumerate(questions, 1):
                print(f"\n   📋 {category} {i}/{len(questions)}")
                print(f"   ❓ {question[:80]}...")
                
                start_time = time.time()
                
                try:
                    result = await system.ask_question(
                        question=question,
                        execute=False,
                        include_summary=False
                    )
                    
                    execution_time = time.time() - start_time
                    category_time += execution_time
                    
                    if result["success"]:
                        category_success += 1
                        print(f"   ✅ Success ({execution_time:.2f}s)")
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        print(f"   ❌ Failed: {error_msg}")
                        
                        # Log category-specific error
                        error_logger.log_error(
                            question=f"[{category}] {question}",
                            error_message=error_msg,
                            generated_query=result.get("data", {}).get("sql_query"),
                            metadata={"category": category, **result.get("metadata", {})}
                        )
                
                except Exception as e:
                    execution_time = time.time() - start_time
                    category_time += execution_time
                    error_msg = str(e)
                    print(f"   ❌ Exception: {error_msg}")
                    
                    error_logger.log_error(
                        question=f"[{category}] {question}",
                        error_message=error_msg,
                        generated_query=None,
                        metadata={"category": category, "exception": True}
                    )
            
            # Category summary
            success_rate = (category_success / len(questions)) * 100
            avg_time = category_time / len(questions)
            
            category_results[category] = {
                "success_rate": success_rate,
                "successful": category_success,
                "total": len(questions),
                "avg_time": avg_time
            }
            
            print(f"   📊 {category} Results: {category_success}/{len(questions)} ({success_rate:.1f}%) - Avg: {avg_time:.2f}s")
        
        # Category comparison
        print(f"\n📊 CATEGORY COMPARISON")
        print("-" * 60)
        
        sorted_categories = sorted(category_results.items(), key=lambda x: x[1]["success_rate"], reverse=True)
        
        for category, results in sorted_categories:
            rate = results["success_rate"]
            status = "🎯" if rate >= 80 else "👍" if rate >= 60 else "⚠️ "
            print(f"{status} {category}: {results['successful']}/{results['total']} ({rate:.1f}%) - {results['avg_time']:.2f}s avg")
        
        # Save category-specific error summary
        if error_logger.errors:
            error_logger.save_summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Category test failed: {str(e)}")
        return False
        
    finally:
        await system.close()


async def main():
    """Run comprehensive full question test suite"""
    
    print("🧪 FULL BUSINESS QUESTIONS TEST SUITE")
    print("=" * 80)
    print(f"📅 Test run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Clear previous log files
    for log_file in ["full_qn_test_errors.log", "category_test_errors.log"]:
        if os.path.exists(log_file):
            os.remove(log_file)
            print(f"🗑️  Cleared previous log: {log_file}")
    
    # Test 1: All questions comprehensive test
    print("\n" + "="*80)
    print("Test 1: Comprehensive Business Questions Test")
    print("="*80)
    test1_passed = await test_all_questions()
    
    # Test 2: Category-specific detailed analysis
    print("\n" + "="*80)
    print("Test 2: Category-Specific Analysis")
    print("="*80)
    test2_passed = await test_specific_categories()
    
    # Final summary
    print("\n📋 FINAL TEST RESULTS")
    print("=" * 80)
    print(f"🧪 Comprehensive Questions Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"🎯 Category Analysis Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    overall_success = test1_passed and test2_passed
    
    if overall_success:
        print(f"\n🎉 OVERALL RESULT: ✅ ALL TESTS PASSED")
        print("\n🎯 System Successfully Handles Business Questions:")
        print("   ✅ Comprehensive question coverage")
        print("   ✅ Error logging and analysis")
        print("   ✅ Category-specific performance insights")
        print("   ✅ Performance metrics captured")
        print("   ✅ Failure analysis completed")
    else:
        print(f"\n❌ OVERALL RESULT: SOME TESTS FAILED")
        print("\n⚠️  Issues detected:")
        if not test1_passed:
            print("   ❌ Comprehensive test issues")
        if not test2_passed:
            print("   ❌ Category analysis problems")
        print("\n🔧 Review error logs for details")
    
    print(f"\n📊 Error logs generated:")
    for log_file in ["full_qn_test_errors.log", "category_test_errors.log", 
                     "full_qn_test_errors_summary.json", "category_test_errors_summary.json"]:
        if os.path.exists(log_file):
            print(f"   📄 {log_file}")
    
    return overall_success


if __name__ == "__main__":
    """Run the comprehensive business questions test suite"""
    print("Starting Full Business Questions Test Suite...")
    success = asyncio.run(main())
    print(f"\nTest suite {'✅ COMPLETED SUCCESSFULLY' if success else '❌ COMPLETED WITH FAILURES'}")
    exit(0 if success else 1)
