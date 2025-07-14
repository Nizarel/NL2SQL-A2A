# NL2SQL Agent 🤖

A sophisticated Natural Language to SQL converter using **Semantic Kernel 1.34.0** with **MCP (Model Context Protocol)** integration for real-time database access.

## 🎯 Overview

This NL2SQL Agent converts natural language questions into accurate SQL queries and executes them against a business analytics database. It leverages:

- **Semantic Kernel**: Microsoft's AI orchestration framework
- **MCP Integration**: Real-time database schema and query execution
- **Azure OpenAI/OpenAI**: For intelligent SQL generation
- **FastMCP**: High-performance MCP client implementation

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Natural       │    │   NL2SQL Agent   │    │   SQL Database  │
│   Language      │───▶│                  │───▶│   (Azure SQL)   │
│   Question      │    │  Semantic Kernel │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   MCP Server     │
                       │   (Database      │
                       │    Integration)  │
                       └──────────────────┘
```

## 📊 Database Schema

The system works with a **beverage/retail analytics database** containing:

- **`dev.cliente`** - Customer dimension (41 columns)
- **`dev.producto`** - Product dimension (12 columns) 
- **`dev.segmentacion`** - Sales fact table (12 columns)
- **`dev.tiempo`** - Time dimension (13 columns)
- **`dev.mercado`** - Market/territory dimension (6 columns)
- **`dev.cliente_cedi`** - Customer-distribution center bridge (9 columns)

## 🚀 Features

- ✅ **Natural Language Processing**: Convert questions to SQL using AI
- ✅ **Real-time Schema Discovery**: Dynamic database introspection via MCP
- ✅ **Intelligent Query Generation**: Context-aware SQL with proper joins
- ✅ **Query Execution**: Direct database query execution
- ✅ **Result Explanation**: AI-generated explanations of queries and results
- ✅ **Error Handling**: Comprehensive error reporting and validation
- ✅ **Interactive Mode**: Command-line interface for testing

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Access to OpenAI API or Azure OpenAI
- MCP server access

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd NL2SQL
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install semantic-kernel==1.34.0 fastmcp httpx aiohttp python-dotenv
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your credentials
```

### Environment Variables

Create a `.env` file with:

```env
# Azure OpenAI (preferred)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# OR OpenAI (fallback)
OPENAI_API_KEY=your_openai_api_key

# MCP Server
MCP_SERVER_URL=https://azsql-fastmcpserv2.jollyfield-479bc951.eastus2.azurecontainerapps.io/mcp/
```

## 📖 Usage

### Command Line Interface

```bash
python src/main.py
```

### Programmatic Usage

```python
import asyncio
from src.main import NL2SQLAgent

async def main():
    # Initialize agent
    agent = NL2SQLAgent()
    await agent.initialize()
    
    # Ask a question
    result = await agent.ask_question(
        "Show me the top 10 customers by revenue",
        execute=True,
        limit=10
    )
    
    if result["success"]:
        print(f"SQL: {result['sql_query']}")
        print(f"Results: {result['results']}")
    
    await agent.close()

asyncio.run(main())
```

### Example Questions

- "Show me the top 10 customers by total revenue"
- "What are the best selling products by category?"
- "List sales data for customers in the Universidad segment"
- "Which territories have the highest sales volume?"
- "Show me monthly revenue trends for this year"

## 🔧 Configuration

### MCP Configuration

The MCP server connection is configured in `.vscode/mcp.json`:

```json
{
  "servers": {
    "arca-mcp-srv04": {
      "url": "https://azsql-fastmcpserv2.jollyfield-479bc951.eastus2.azurecontainerapps.io/mcp/"
    }
  }
}
```

### AI Service Setup

The agent supports both Azure OpenAI and OpenAI:

1. **Azure OpenAI** (recommended for enterprise)
2. **OpenAI** (fallback option)

## 📁 Project Structure

```
NL2SQL/
├── src/
│   ├── main.py                 # Main application entry point
│   ├── plugins/
│   │   ├── __init__.py
│   │   └── mcp_database_plugin.py  # MCP integration plugin
│   └── services/
│       ├── __init__.py
│       ├── schema_service.py       # Database schema management
│       └── nl2sql_service.py       # Core NL2SQL conversion logic
├── .vscode/
│   └── mcp.json               # MCP server configuration
├── .env                       # Environment variables (create from .env.example)
├── .gitignore
└── README.md
```

## 🧪 Testing

### Manual Testing

```python
# Test database connection
result = await agent.get_database_info()
print(result)

# Test schema discovery
schema = await agent.get_schema_context()
print(schema)

# Test query generation
result = await agent.ask_question("List all customers", execute=False)
print(result["sql_query"])
```

### Example Outputs

**Question**: "Show me top 5 customers by revenue"

**Generated SQL**:
```sql
SELECT TOP 5 
    c.customer_id,
    c.Nombre_cliente,
    SUM(s.IngresoNetoSImpuestos) as total_revenue
FROM dev.cliente c
JOIN dev.segmentacion s ON c.customer_id = s.customer_id
WHERE s.IngresoNetoSImpuestos > 0
GROUP BY c.customer_id, c.Nombre_cliente
ORDER BY total_revenue DESC
```

## 🔍 Key Components

### 1. **NL2SQLAgent** (`main.py`)
- Main orchestrator class
- Handles initialization and coordination
- Provides high-level API

### 2. **MCPDatabasePlugin** (`plugins/mcp_database_plugin.py`)
- MCP server integration
- Database operations (list tables, describe schema, execute queries)
- Uses fastmcp.Client for communication

### 3. **SchemaService** (`services/schema_service.py`)
- Database schema discovery and caching
- Relationship mapping
- Business context management

### 4. **NL2SQLService** (`services/nl2sql_service.py`)
- Core NL2SQL conversion logic
- AI-powered SQL generation
- Query execution and result processing

## 🚨 Error Handling

The system uses a "fail-fast" approach with no fallbacks:

- **AI Service Required**: Must configure OpenAI or Azure OpenAI
- **MCP Server Required**: Must be accessible and responding
- **Schema Discovery Required**: Database must be introspectable
- **Clear Error Messages**: All failures provide detailed diagnostics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Technologies

- [Semantic Kernel](https://github.com/microsoft/semantic-kernel) - AI orchestration framework
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) - Standardized context sharing
- [FastMCP](https://github.com/jlowin/fastmcp) - High-performance MCP client
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) - Enterprise AI service

## 📞 Support

For questions and support:
- Create an issue in this repository
- Review the error messages for configuration guidance
- Check the MCP server connectivity

---

**Built with ❤️ using Semantic Kernel and MCP** Agent

A sophisticated Natural Language to SQL (NL2SQL) converter using Semantic Kernel 1.34.0 with MCP (Model Context Protocol) database integration.

## Features

- 🧠 **Semantic Kernel Integration**: Uses Semantic Kernel 1.34.0 for advanced AI-powered query generation
- 🔌 **MCP Database Plugin**: Seamless integration with MCP server for database operations
- 📊 **Schema-Aware**: Automatically discovers and uses database schema for accurate query generation
- 🎯 **Business Context**: Understands business relationships and generates meaningful queries
- 🔄 **Real-time Execution**: Converts natural language to SQL and executes queries instantly
- 💡 **Query Explanation**: Provides business-friendly explanations of generated queries and results

## Architecture

```
NL2SQL Agent
├── Semantic Kernel Core
├── MCP Database Plugin
│   ├── list_tables()
│   ├── describe_table()
│   ├── read_data()
│   └── database_info()
├── Schema Service
│   ├── Schema Caching
│   ├── Relationship Mapping
│   └── Context Generation
└── NL2SQL Service
    ├── Query Generation
    ├── Execution
    └── Explanation
```

## Database Schema

The system works with a business analytics database containing:

- **Fact Table**: `dev.segmentacion` (Sales data)
- **Dimensions**: 
  - `dev.cliente` (Customers)
  - `dev.producto` (Products) 
  - `dev.tiempo` (Time)
  - `dev.mercado` (Markets/Territories)
- **Bridge**: `dev.cliente_cedi` (Customer-Distribution Center)

## Installation

1. **Clone and setup:**
```bash
git clone <repository>
cd NL2SQL
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
Create `.env` file with your AI service credentials:
```env
# For OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# OR for Azure OpenAI
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name_here

# MCP Server (already configured)
MCP_SERVER_URL=https://azsql-fastmcpserv2.jollyfield-479bc951.eastus2.azurecontainerapps.io/mcp/
```

## Usage

### Basic Usage

```python
from src.main import NL2SQLAgent

# Initialize agent
agent = NL2SQLAgent()
await agent.initialize()

# Ask questions in natural language
result = await agent.ask_question("Show me the top 10 customers by revenue")

if result["success"]:
    print(f"SQL: {result['sql_query']}")
    print(f"Results: {result['results']}")
```

### Interactive Mode

```bash
cd src
python main.py
```

### Example Questions

- "Show me the top 10 customers by total revenue"
- "What are the best selling products by category?"
- "Show me sales data for the last month"
- "Which territories have the highest sales?"
- "List all customers in the Universidad segment"
- "What is the total revenue for each product category?"