#!/usr/bin/env python3
"""
Debug script to examine actual database schema
"""
import sys
import os
import asyncio

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from plugins.mcp_database_plugin import MCPDatabasePlugin

async def examine_schema():
    """Examine the actual database schema"""
    print("🔍 Examining Database Schema...")
    
    # Initialize MCP Plugin
    mcp_server_url = "https://azsql-fastmcpserv2.jollyfield-479bc951.eastus2.azurecontainerapps.io/mcp/"
    mcp_plugin = MCPDatabasePlugin(mcp_server_url)
    
    async with mcp_plugin:
        # Get tables
        tables_result = await mcp_plugin._list_tables_raw()
        print(f"📋 Tables: {tables_result}")
        
        # Get schema for each table
        table_names = ['cliente', 'cliente_cedi', 'mercado', 'producto', 'segmentacion', 'tiempo']
        
        for table_name in table_names:
            print(f"\n🔍 Schema for {table_name}:")
            try:
                schema_info = await mcp_plugin._describe_table_raw(table_name)
                print(f"   {schema_info}")
            except Exception as e:
                print(f"   Error: {e}")

if __name__ == "__main__":
    asyncio.run(examine_schema())
