"""
Schema Service for NL2SQL Agent
Manages database schema context and provides schema information for SQL generation
"""

from typing import Dict, List, Any, Optional
import json
import re


class SchemaService:
    """
    Service that manages database schema context for SQL generation
    """
    
    def __init__(self, mcp_plugin):
        self.mcp_plugin = mcp_plugin
        self.schema_context = {}
        self.tables_info = {}
        self.relationships = {}
    
    async def initialize_schema_context(self) -> Dict[str, Any]:
        """
        Initialize schema context by gathering all table information
        """
        # Get list of tables
        tables_result = await self.mcp_plugin.list_tables()
        print(f"Tables found: {tables_result}")
        
        # Parse table names
        table_names = self._parse_table_names(tables_result)
        
        # Get schema for each table
        for table_name in table_names:
            print(f"Getting schema for table: {table_name}")
            schema_info = await self.mcp_plugin.describe_table(table_name)
            self.tables_info[table_name] = schema_info
        
        # Build schema context
        self.schema_context = {
            "database_name": "Business Analytics Database",
            "schema_name": "dev",
            "tables": self.tables_info,
            "relationships": self._build_relationships(),
            "business_context": self._get_business_context()
        }
        
        return self.schema_context
    
    def _parse_table_names(self, tables_result: str) -> List[str]:
        """Parse table names from the MCP result"""
        # Try to extract table names from the result string
        if "Available tables" in tables_result:
            # Extract from: "Available tables (6): ['cliente', 'cliente_cedi', ...]"
            start = tables_result.find("[")
            end = tables_result.find("]")
            if start != -1 and end != -1:
                tables_str = tables_result[start+1:end]
                # Remove quotes and split
                table_names = [name.strip().strip("'\"") for name in tables_str.split(",")]
                return table_names
        
        # If parsing fails, raise an error
        raise ValueError(f"Could not parse table names from: {tables_result}")
    
    def _build_relationships(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Define table relationships based on schema analysis
        """
        return {
            "cliente": [
                {"table": "cliente_cedi", "type": "one_to_many", "key": "customer_id"},
                {"table": "segmentacion", "type": "one_to_many", "key": "customer_id"}
            ],
            "producto": [
                {"table": "segmentacion", "type": "one_to_many", "key": "Material -> material_id"}
            ],
            "tiempo": [
                {"table": "segmentacion", "type": "one_to_many", "key": "Fecha -> calday"},
                {"table": "segmentacion", "type": "one_to_many", "key": "CALMONTH"}
            ],
            "mercado": [
                {"table": "cliente_cedi", "type": "one_to_many", "key": "CEDIid -> cedi_id"}
            ],
            "cliente_cedi": [
                {"table": "cliente", "type": "many_to_one", "key": "customer_id"},
                {"table": "mercado", "type": "many_to_one", "key": "cedi_id -> CEDIid"}
            ],
            "segmentacion": [
                {"table": "cliente", "type": "many_to_one", "key": "customer_id"},
                {"table": "producto", "type": "many_to_one", "key": "material_id -> Material"},
                {"table": "tiempo", "type": "many_to_one", "key": "calday -> Fecha"},
                {"table": "tiempo", "type": "many_to_one", "key": "CALMONTH"}
            ]
        }
    
    def _get_business_context(self) -> Dict[str, Any]:
        """
        Provide business context for better SQL generation
        """
        return {
            "domain": "Beverage/Retail Analytics",
            "purpose": "Sales and customer analytics for distribution centers",
            "key_metrics": [
                "VentasCajasUnidad (Unit Sales)",
                "IngresoNetoSImpuestos (Net Revenue)",
                "net_revenue",
                "bottles_sold_m"
            ],
            "date_fields": [
                "tiempo.Fecha (Primary date field)",
                "segmentacion.calday (Transaction date)",
                "tiempo.CALMONTH (Calendar month)"
            ],
            "common_filters": [
                "Year, Quarter, Month from tiempo table",
                "Territory, Region from cliente_cedi/mercado",
                "Product Category from producto table",
                "Commercial Channel from cliente table"
            ],
            "schema_prefix": "dev",
            "important_notes": [
                "Always use 'dev.' prefix for table names",
                "segmentacion is the main fact table with sales data",
                "cliente, producto, tiempo are dimension tables",
                "Use JOINS to combine data across tables",
                "SQL Server syntax - no LIMIT clause, use TOP instead"
            ]
        }
    
    
    def get_schema_for_query(self, tables_mentioned: List[str]) -> str:
        """
        Get relevant schema information for specific tables mentioned in a query
        """
        schema_info = []
        
        for table in tables_mentioned:
            if table in self.tables_info:
                schema_info.append(f"Table: dev.{table}")
                schema_info.append(self.tables_info[table])
                schema_info.append("")
        
        # Add relationship information
        relationships = []
        for table in tables_mentioned:
            if table in self.relationships:
                for rel in self.relationships[table]:
                    if rel["table"] in tables_mentioned:
                        relationships.append(f"{table} -> {rel['table']}: {rel['key']} ({rel['type']})")
        
        if relationships:
            schema_info.append("Relationships:")
            schema_info.extend(relationships)
        
        return "\n".join(schema_info)
    
    def get_full_schema_summary(self) -> str:
        """
        Get a complete schema summary for context
        """
        if not self.schema_context:
            return "Schema context not initialized"
        
        summary = []
        summary.append("=== DATABASE SCHEMA SUMMARY ===")
        summary.append(f"Database: {self.schema_context.get('database_name', 'Unknown')}")
        summary.append(f"Schema: {self.schema_context.get('schema_name', 'dev')}")
        summary.append("")
        
        summary.append("TABLES:")
        for table_name in self.tables_info.keys():
            summary.append(f"- dev.{table_name}")
        summary.append("")
        
        summary.append("BUSINESS CONTEXT:")
        business_context = self.schema_context.get('business_context', {})
        summary.append(f"Domain: {business_context.get('domain', 'Analytics')}")
        summary.append("")
        
        summary.append("KEY METRICS:")
        for metric in business_context.get('key_metrics', []):
            summary.append(f"- {metric}")
        summary.append("")
        
        summary.append("IMPORTANT NOTES:")
        for note in business_context.get('important_notes', []):
            summary.append(f"- {note}")
        
        return "\n".join(summary)
