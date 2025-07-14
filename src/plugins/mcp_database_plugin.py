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
    client: Optional[Client] = Field(default=None, exclude=True)
    
    def __init__(self, mcp_server_url: str):
        super().__init__(mcp_server_url=mcp_server_url)
        self.client = Client(mcp_server_url)
    
    async def _get_client(self) -> Client:
        """Get the MCP client"""
        return self.client
    
    @kernel_function(
        description="List all available tables in the database",
        name="list_database_tables"
    )
    async def list_tables(self) -> str:
        """List all tables in the database using fastmcp client"""
        async with self.client:
            result = await self.client.call_tool("list_tables")
            return str(result)
    
    @kernel_function(
        description="Get detailed schema information for a specific table",
        name="describe_table_schema"
    )
    async def describe_table(self, table_name: str) -> str:
        """Get table structure and schema information using fastmcp client"""
        async with self.client:
            result = await self.client.call_tool("describe_table", {"table_name": table_name})
            return str(result)
    
    @kernel_function(
        description="Execute a SELECT query and return results (automatically adds dev. schema prefix)",
        name="execute_sql_query"
    )
    async def read_data(self, query: str, limit: int = 100) -> str:
        """Execute SELECT queries to read data from the database using fastmcp client"""
        # Add dev. schema prefix to table names if not present
        modified_query = self._add_schema_prefix(query)
        
        async with self.client:
            result = await self.client.call_tool("read_data", {"query": modified_query, "limit": limit})
            return str(result)
    
    @kernel_function(
        description="Get comprehensive database information and connection status",
        name="get_database_info"
    )
    async def database_info(self) -> str:
        """Get database connection information using fastmcp client"""
        async with self.client:
            result = await self.client.call_tool("database_info")
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
        if self.client:
            # Client handles disconnection automatically
            self.client = None
    
    # Context manager methods for batch operations
    async def __aenter__(self):
        """Enter context manager for batch operations"""
        self._context = await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager for batch operations"""
        return await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    # Raw methods for use within context manager
    async def _list_tables_raw(self) -> str:
        """List tables - for use within context manager"""
        try:
            result = await self.client.call_tool("list_tables")
            return str(result)
        except Exception as e:
            return f"Error listing tables: {str(e)}"
    
    async def _describe_table_raw(self, table_name: str) -> str:
        """Describe table - for use within context manager"""
        try:
            result = await self.client.call_tool("describe_table", {"table_name": table_name})
            return str(result)
        except Exception as e:
            return f"Error describing table {table_name}: {str(e)}"
    
    async def _database_info_raw(self) -> str:
        """Get database info - for use within context manager"""
        try:
            result = await self.client.call_tool("database_info")
            return str(result)
        except Exception as e:
            return f"Error getting database info: {str(e)}"
