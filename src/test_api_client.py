"""
Example API Client for testing the NL2SQL Multi-Agent API
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any

class NL2SQLAPIClient:
    """Simple client for the NL2SQL API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                return await response.json()
    
    async def orchestrator_query(self, question: str, execute: bool = True, 
                               limit: int = 100, include_summary: bool = True,
                               context: str = "") -> Dict[str, Any]:
        """Process a natural language query using the orchestrator"""
        payload = {
            "question": question,
            "execute": execute,
            "limit": limit,
            "include_summary": include_summary,
            "context": context
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/orchestrator/query",
                json=payload
            ) as response:
                return await response.json()
    
    async def generate_sql(self, question: str, context: str = "") -> Dict[str, Any]:
        """Generate SQL from natural language"""
        payload = {
            "question": question,
            "context": context
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/agents/sql-generator/generate",
                json=payload
            ) as response:
                return await response.json()
    
    async def execute_sql(self, sql_query: str, limit: int = 100, 
                         timeout: int = 30) -> Dict[str, Any]:
        """Execute SQL query"""
        payload = {
            "sql_query": sql_query,
            "limit": limit,
            "timeout": timeout
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/agents/executor/execute",
                json=payload
            ) as response:
                return await response.json()
    
    async def get_database_tables(self) -> Dict[str, Any]:
        """Get list of database tables"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/database/tables") as response:
                return await response.json()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/status") as response:
                return await response.json()


async def test_api():
    """Test the NL2SQL API"""
    client = NL2SQLAPIClient()
    
    print("🧪 Testing NL2SQL Multi-Agent API")
    print("=" * 50)
    
    try:
        # Test 1: Health check
        print("1. Health Check...")
        health = await client.health_check()
        print(f"   Status: {'✅ Healthy' if health.get('success') else '❌ Unhealthy'}")
        
        # Test 2: System status
        print("\n2. System Status...")
        status = await client.get_system_status()
        if status.get('success'):
            agents = status.get('data', {}).get('agents', {})
            print(f"   Orchestrator: {status.get('data', {}).get('orchestrator', 'unknown')}")
            for agent, state in agents.items():
                print(f"   {agent}: {state}")
        
        # Test 3: Database tables
        print("\n3. Database Tables...")
        tables = await client.get_database_tables()
        if tables.get('success'):
            print(f"   Tables info: {str(tables.get('data', {}))[:100]}...")
        
        # Test 4: SQL Generation only
        print("\n4. SQL Generation Test...")
        sql_result = await client.generate_sql("Show top 5 customers by revenue")
        if sql_result.get('success'):
            sql_query = sql_result.get('data', {}).get('sql_query', '')
            print(f"   Generated SQL: {sql_query[:100]}...")
            
            # Test 5: SQL Execution
            print("\n5. SQL Execution Test...")
            exec_result = await client.execute_sql(sql_query, limit=5)
            if exec_result.get('success'):
                row_count = exec_result.get('metadata', {}).get('row_count', 0)
                print(f"   Execution result: {row_count} rows returned")
            else:
                print(f"   Execution failed: {exec_result.get('error')}")
        else:
            print(f"   SQL generation failed: {sql_result.get('error')}")
        
        # Test 6: Full orchestrator workflow
        print("\n6. Full Orchestrator Workflow...")
        full_result = await client.orchestrator_query(
            question="What are the top 3 products by revenue?",
            limit=3
        )
        if full_result.get('success'):
            data = full_result.get('data', {})
            print(f"   SQL Generated: {'✅' if data.get('sql_query') else '❌'}")
            print(f"   Query Executed: {'✅' if data.get('executed') else '❌'}")
            print(f"   Summary Generated: {'✅' if data.get('summary') else '❌'}")
            
            if data.get('summary'):
                summary = data['summary'].get('executive_summary', '')
                print(f"   Executive Summary: {summary[:150]}...")
        else:
            print(f"   Orchestrator workflow failed: {full_result.get('error')}")
        
        print("\n🎉 API Testing Complete!")
        
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_api())
