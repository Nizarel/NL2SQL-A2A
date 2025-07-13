"""
MCP Database Plugin for Semantic Kernel
Integrates MCP Database Query Tools with Semantic Kernel using fastmcp.client
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from semantic_kernel.functions import kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel
from pydantic import Field
from fastmcp import Client


class MCPDatabasePlugin(KernelBaseModel):
    """
    MCP Database Plugin that provides database operations through MCP server
    Uses fastmcp.Client for proper MCP protocol communication
    """
    
    mcp_server_url: str = Field(description="MCP server URL")
    _client: Optional[Client] = Field(default=None, exclude=True)
    
    def __init__(self, mcp_server_url: str):
        super().__init__(mcp_server_url=mcp_server_url)
        self._client = None
    
    async def _get_client(self) -> Client:
        """Get or create MCP client"""
        if self._client is None:
            self._client = Client(self.mcp_server_url)
            await self._client.connect()
        return self._client
    
    @kernel_function(
        description="List all available tables in the database",
        name="list_database_tables"
    )
    async def list_tables(self) -> str:
        """List all tables in the database using fastmcp client"""
        client = await self._get_client()
        result = await client.call_tool("list_tables")
        return str(result)
    
    @kernel_function(
        description="Get detailed schema information for a specific table",
        name="describe_table_schema"
    )
    async def describe_table(self, table_name: str) -> str:
        """Get table structure and schema information using fastmcp client"""
        client = await self._get_client()
        result = await client.call_tool("describe_table", {"table_name": table_name})
        return str(result)
    
    @kernel_function(
        description="Execute a SELECT query and return results (automatically adds dev. schema prefix)",
        name="execute_sql_query"
    )
    async def read_data(self, query: str, limit: int = 100) -> str:
        """Execute SELECT queries to read data from the database using fastmcp client"""
        # Add dev. schema prefix to table names if not present
        modified_query = self._add_schema_prefix(query)
        
        client = await self._get_client()
        result = await client.call_tool("read_data", {"query": modified_query, "limit": limit})
        return str(result)
    
    @kernel_function(
        description="Get comprehensive database information and connection status",
        name="get_database_info"
    )
    async def database_info(self) -> str:
        """Get database connection information using fastmcp client"""
        client = await self._get_client()
        result = await client.call_tool("database_info")
        return str(result)
    
    def _add_schema_prefix(self, query: str) -> str:
        """Add dev. schema prefix to table names in the query"""
        table_names = ["cliente", "cliente_cedi", "mercado", "producto", "segmentacion", "tiempo"]
        modified_query = query
        
        for table in table_names:
            # Replace table name with dev.table_name if not already prefixed
            if f"dev.{table}" not in modified_query.lower():
                # Use word boundaries to avoid partial matches
                import re
                pattern = r'\b' + re.escape(table) + r'\b'
                modified_query = re.sub(pattern, f"dev.{table}", modified_query, flags=re.IGNORECASE)
        
        return modified_query
    
    async def close(self):
        """Close MCP client connection"""
        if self._client:
            await self._client.disconnect()
            self._client = None
