#!/usr/bin/env python3
"""Debug script to check available regions and data"""

import asyncio
import os
from dotenv import load_dotenv
from src.plugins.mcp_database_plugin import MCPDatabasePlugin

async def debug_regions():
    """Check what regions are available in the database"""
    # Load environment
    env_path = os.path.join("/workspaces/NL2SQL/src/.env")
    print(f"🔍 Loading .env from: {env_path}")
    print(f"🔍 .env file exists: {os.path.exists(env_path)}")
    result = load_dotenv(env_path)
    print(f"🔍 load_dotenv result: {result}")
    
    # Initialize database plugin
    mcp_server_url = os.getenv("MCP_SERVER_URL", "https://azsql-fastmcpserv2.jollyfield-479bc951.eastus2.azurecontainerapps.io/mcp/")
    db_plugin = MCPDatabasePlugin(mcp_server_url=mcp_server_url)
    
    # Check available regions
    print("\n1. Checking available regions...")
    regions_query = "SELECT Region FROM cliente_cedi GROUP BY Region ORDER BY Region"
    result = await db_plugin.read_data(regions_query)
    print(f"Available regions: {result}")
    
    # Check table data counts (already working)
    print("\n2. Checking data counts...")
    tables = ['cliente', 'cliente_cedi', 'mercado', 'producto', 'segmentacion', 'tiempo']
    for table in tables:
        count_query = f"SELECT COUNT(*) as count FROM {table}"
        result = await db_plugin.read_data(count_query)
        print(f"Table {table}: {result}")
    
    # Check cliente_cedi structure
    print("\n3. Checking cliente_cedi structure...")
    structure_query = "SELECT TOP 5 * FROM cliente_cedi"
    result = await db_plugin.read_data(structure_query)
    print(f"Sample cliente_cedi data: {result}")
    
    # Check mercado structure  
    print("\n4. Checking mercado structure...")
    mercado_query = "SELECT TOP 5 * FROM mercado"
    result = await db_plugin.read_data(mercado_query)
    print(f"Sample mercado data: {result}")
    
    await db_plugin.close()

if __name__ == "__main__":
    asyncio.run(debug_regions())
