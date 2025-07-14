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
        async with self.mcp_plugin:
            # Get list of tables
            tables_result = await self.mcp_plugin._list_tables_raw()
            print(f"Tables found: {tables_result}")
            
            # Parse table names
            table_names = self._parse_table_names(tables_result)
            
            # Get schema for each table
            for table_name in table_names:
                print(f"Getting schema for table: {table_name}")
                schema_info = await self.mcp_plugin._describe_table_raw(table_name)
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
        
        # Add detailed table schemas
        summary.append("DETAILED TABLE SCHEMAS:")
        summary.append("")
        
        # Key table information with important columns highlighted
        summary.append("TABLE: dev.cliente (Customer Master Data)")
        summary.append("- customer_id (varchar, PRIMARY KEY) - Unique customer identifier")
        summary.append("- Nombre_cliente (varchar) - Customer name")
        summary.append("- Canal_Comercial (varchar) - Commercial channel")
        summary.append("- Territorio_del_cliente (varchar) - Customer territory")
        summary.append("- Region (via cliente_cedi join)")
        summary.append("")
        
        summary.append("TABLE: dev.segmentacion (Sales/Revenue Data)")
        summary.append("- customer_id (varchar, FOREIGN KEY) - Links to cliente.customer_id")
        summary.append("- material_id (varchar, FOREIGN KEY) - Links to producto.Material")
        summary.append("- calday (date) - Transaction date")
        summary.append("- IngresoNetoSImpuestos (float) - Net revenue without taxes")
        summary.append("- net_revenue (float) - Net revenue")
        summary.append("- VentasCajasUnidad (float) - Units sold")
        summary.append("- VentasCajasOriginales (float) - Original cases sold")
        summary.append("- bottles_sold_m (float) - Bottles sold")
        summary.append("")
        
        summary.append("TABLE: dev.producto (Product Master Data)")
        summary.append("- Material (varchar, PRIMARY KEY) - Product material code")
        summary.append("- Producto (varchar) - Product name")
        summary.append("- Categoria (varchar) - Product category")
        summary.append("- Subcategoria (varchar) - Product subcategory")
        summary.append("- AgrupadordeMarca (varchar) - Brand grouping")
        summary.append("")
        
        summary.append("TABLE: dev.cliente_cedi (Customer Distribution)")
        summary.append("- customer_id (varchar, FOREIGN KEY) - Links to cliente.customer_id")
        summary.append("- cedi_id (varchar) - Distribution center ID")
        summary.append("- Region (varchar) - Geographic region")
        summary.append("- Territorio (varchar) - Territory")
        summary.append("- Subterritorio (varchar) - Sub-territory")
        summary.append("")
        
        summary.append("TABLE: dev.mercado (Market/Territory Data)")
        summary.append("- CEDIid (varchar, PRIMARY KEY) - Distribution center ID")
        summary.append("- CEDI (varchar) - Distribution center name")
        summary.append("- Zona (varchar) - Zone")
        summary.append("- Territorio (varchar) - Territory")
        summary.append("")
        
        summary.append("TABLE: dev.tiempo (Time Dimension)")
        summary.append("- Fecha (date, PRIMARY KEY) - Date")
        summary.append("- Year (int) - Year")
        summary.append("- NumMes (int) - Month number")
        summary.append("- Mes (varchar) - Month name")
        summary.append("- Q (int) - Quarter")
        summary.append("")
        
        summary.append("COMMON JOIN PATTERNS:")
        summary.append("- Customer Revenue: segmentacion.customer_id = cliente.customer_id")
        summary.append("- Customer Territory: cliente.customer_id = cliente_cedi.customer_id")
        summary.append("- Product Details: segmentacion.material_id = producto.Material")
        summary.append("- Date Analysis: segmentacion.calday = tiempo.Fecha")
        summary.append("")
        
        summary.append("REVENUE METRICS:")
        summary.append("- IngresoNetoSImpuestos: Primary revenue metric (net without taxes)")
        summary.append("- net_revenue: Alternative revenue metric")
        summary.append("- VentasCajasUnidad: Units/cases sold")
        summary.append("- bottles_sold_m: Volume in bottles")
        summary.append("")
        
        summary.append("IMPORTANT NOTES:")
        summary.append("- Always use 'dev.' prefix for table names")
        summary.append("- customer_id is the key field (NOT cliente_id)")
        summary.append("- Nombre_cliente contains customer names")
        summary.append("- Use IngresoNetoSImpuestos for revenue calculations")
        summary.append("- Date filtering should use calday field in segmentacion")
        
        return "\n".join(summary)
