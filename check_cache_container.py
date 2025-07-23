"""
Script to check cache entries in the nl2sql_cache container
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from services.cosmos_db_service import CosmosDbService


async def check_cache_container():
    """Check what's in the cache container"""
    print("🔍 Checking nl2sql_cache container...")
    
    cosmos_service = CosmosDbService(
        endpoint="https://cosmos-acrasalesanalytics2.documents.azure.com:443/",
        database_name="sales_analytics",
        chat_container_name="nl2sql_chatlogs",
        cache_container_name="nl2sql_cache"
    )
    
    await cosmos_service.initialize()
    
    try:
        # Query all items in the cache container
        cache_container = cosmos_service._cache_container
        
        query = "SELECT * FROM c"
        
        cache_items = []
        async for item in cache_container.query_items(query=query):
            cache_items.append(item)
        
        print(f"📊 Found {len(cache_items)} items in nl2sql_cache container")
        
        if cache_items:
            print("\n🗂️ Cache Items:")
            for i, item in enumerate(cache_items[:5], 1):  # Show first 5 items
                print(f"  {i}. ID: {item.get('id', 'N/A')}")
                print(f"     Type: {item.get('metadata', {}).get('type', 'N/A')}")
                print(f"     User ID: {item.get('metadata', {}).get('user_id', 'N/A')}")
                print(f"     Text: {item.get('text', 'N/A')[:100]}...")
                print(f"     Timestamp: {item.get('metadata', {}).get('timestamp', 'N/A')}")
                print()
        else:
            print("❌ No cache items found")
            
        # Try to query specific cache items
        workflow_query = "SELECT * FROM c WHERE c.metadata.type = 'workflow_result'"
        workflow_items = []
        async for item in cache_container.query_items(query=workflow_query):
            workflow_items.append(item)
            
        print(f"🔧 Found {len(workflow_items)} workflow_result cache items")
        
        if workflow_items:
            print("\n📋 Workflow Cache Items:")
            for i, item in enumerate(workflow_items, 1):
                metadata = item.get('metadata', {})
                print(f"  {i}. Workflow ID: {metadata.get('workflow_id', 'N/A')}")
                print(f"     SQL Query: {metadata.get('sql_query', 'N/A')}")
                print(f"     Success: {metadata.get('success', 'N/A')}")
                print(f"     Processing Time: {metadata.get('processing_time_ms', 'N/A')}ms")
                print()
                
    except Exception as e:
        print(f"❌ Error checking cache container: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await cosmos_service.close()


if __name__ == "__main__":
    asyncio.run(check_cache_container())
