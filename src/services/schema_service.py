def create_mcp_schema_service(mcp_config: dict) -> 'GenericSchemaService':
    """Factory to create GenericSchemaService with MCPDatabasePlugin as data source"""
    from plugins.mcp_database_plugin import MCPDatabasePlugin
    mcp_plugin = MCPDatabasePlugin(**mcp_config)
    return GenericSchemaService(mcp_plugin, config=mcp_config)
"""
Generic Schema Service - Adaptable to any database through MCP or other providers
"""

from typing import Dict, List, Any, Optional, Protocol
from abc import abstractmethod
import re


class MCPPluginProtocol(Protocol):
    """Protocol for MCP-like plugins"""
    async def _list_tables_raw(self) -> str:
        ...
    
    async def _describe_table_raw(self, table_name: str) -> str:
        ...


class GenericSchemaService:
    """
    Generic schema service that can work with any database
    """
    
    def __init__(
        self, 
        data_source: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        self.data_source = data_source
        self.config = config or {}
        
        # Cache for schema information
        self._tables_cache: Optional[List[str]] = None
        self._schemas_cache: Dict[str, Any] = {}
        self._relationships_cache: Optional[Dict[str, Any]] = None
        
        # Configuration
        self.schema_prefix = self.config.get("schema_prefix", "")
        self.table_parser = self.config.get("table_parser", self._default_table_parser)
        self.relationship_detector = self.config.get("relationship_detector", self._default_relationship_detector)
    
    async def get_tables(self) -> List[str]:
        """Get list of table names"""
        if self._tables_cache is not None:
            return self._tables_cache
        
        # Get tables from data source
        if hasattr(self.data_source, '_list_tables_raw'):
            raw_result = await self.data_source._list_tables_raw()
            self._tables_cache = self.table_parser(raw_result)
        else:
            # Fallback for other data sources
            self._tables_cache = await self._fetch_tables_generic()
        
        return self._tables_cache
    
    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema for a specific table"""
        if table_name in self._schemas_cache:
            return self._schemas_cache[table_name]
        
        # Get schema from data source
        if hasattr(self.data_source, '_describe_table_raw'):
            raw_schema = await self.data_source._describe_table_raw(table_name)
            schema = self._parse_schema(raw_schema, table_name)
        else:
            schema = await self._fetch_schema_generic(table_name)
        
        self._schemas_cache[table_name] = schema
        return schema
    
    async def get_relationships(self) -> Dict[str, Any]:
        """Get table relationships"""
        if self._relationships_cache is not None:
            return self._relationships_cache
        
        # Detect relationships
        tables = await self.get_tables()
        all_schemas = {}
        
        for table in tables:
            all_schemas[table] = await self.get_table_schema(table)
        
        self._relationships_cache = self.relationship_detector(all_schemas)
        return self._relationships_cache
    
    def _default_table_parser(self, raw_result: str) -> List[str]:
        """Default parser for table names from raw string"""
        # Try multiple parsing strategies
        
        # Strategy 1: Look for list format
        if "[" in raw_result and "]" in raw_result:
            start = raw_result.find("[")
            end = raw_result.find("]")
            if start != -1 and end != -1:
                tables_str = raw_result[start+1:end]
                tables = [t.strip().strip("'\"") for t in tables_str.split(",")]
                return [t for t in tables if t]
        
        # Strategy 2: Look for comma-separated values
        if "," in raw_result:
            tables = [t.strip() for t in raw_result.split(",")]
            return [t for t in tables if t and not t.startswith("Available")]
        
        # Strategy 3: Line-by-line
        lines = raw_result.strip().split("\n")
        return [l.strip() for l in lines if l.strip() and not l.startswith("#")]
    
    def _default_relationship_detector(self, all_schemas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Default relationship detection based on column names"""
        relationships = {}
        
        for table_name, schema in all_schemas.items():
            relationships[table_name] = []
            
            # Look for foreign key patterns in column names
            columns = schema.get("columns", [])
            for column in columns:
                col_name = column.get("name", "").lower()
                
                # Common FK patterns
                if col_name.endswith("_id") or col_name.endswith("id"):
                    # Try to find matching table
                    potential_table = col_name.replace("_id", "").replace("id", "")
                    
                    for other_table in all_schemas:
                        if other_table.lower() == potential_table or potential_table in other_table.lower():
                            relationships[table_name].append({
                                "table": other_table,
                                "type": "many_to_one",
                                "key": column.get("name"),
                                "foreign_table": other_table,
                                "foreign_key": "id"  # Common assumption
                            })
        
        return relationships
    
    def _parse_schema(self, raw_schema: str, table_name: str) -> Dict[str, Any]:
        """Parse raw schema string into structured format"""
        schema = {
            "name": table_name,
            "columns": [],
            "raw": raw_schema
        }
        
        # Simple column extraction - override for specific formats
        lines = raw_schema.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and "(" in line:
                # Try to extract column info
                parts = line.split()
                if len(parts) >= 2:
                    schema["columns"].append({
                        "name": parts[0],
                        "type": parts[1].strip("(),"),
                        "full_definition": line
                    })
        
        return schema
    
    async def _fetch_tables_generic(self) -> List[str]:
        """Generic table fetching for non-MCP data sources"""
        # Override this method for specific data sources
        return []
    
    async def _fetch_schema_generic(self, table_name: str) -> Dict[str, Any]:
        """Generic schema fetching for non-MCP data sources"""
        # Override this method for specific data sources
        return {"name": table_name, "columns": []}
    
    def create_query_analyzer(self) -> 'QueryAnalyzer':
        """Create a query analyzer for this schema"""
        return QueryAnalyzer(self)


class QueryAnalyzer:
    """Analyzes queries against the schema"""
    
    def __init__(self, schema_service: GenericSchemaService):
        self.schema_service = schema_service
    
    def identify_relevant_tables(self, query: str, context: str = "") -> List[str]:
        """Identify tables relevant to a query"""
        query_lower = query.lower()
        context_lower = context.lower()
        combined = f"{query_lower} {context_lower}"
        
        relevant_tables = []
        
        # Simple keyword matching - extend as needed
        tables = self.schema_service._tables_cache or []
        for table in tables:
            if table.lower() in combined:
                relevant_tables.append(table)
        
        return relevant_tables
    
    def extract_metrics(self, query: str) -> List[str]:
        """Extract potential metrics from query"""
        metrics = []
        query_lower = query.lower()
        
        # Common aggregate patterns
        aggregates = ["sum", "count", "avg", "average", "max", "min", "total"]
        for agg in aggregates:
            if agg in query_lower:
                metrics.append(agg)
        
        return metrics