# NL2SQL Agent 🤖

A sophisticated Natural Language to SQL converter using **Semantic Kernel 1.34.0** with **MCP (Model Context Protocol)** integration for real-time database access.

## 🎯 Overview

This NL2SQL Agent converts natural language questions into accurate SQL queries and executes them against a business analytics database. It leverages:

- **Semantic Kernel**: Microsoft's AI orchestration framework
- **MCP Integration**: Real-time database schema and query execution
- **Azure OpenAI/OpenAI**: For intelligent SQL generation
- **FastMCP**: High-performance MCP client implementation

## 🏗️ Architecture

The NL2SQL system now supports both **direct integration** and **A2A (Agent-to-Agent) protocol**:

### Direct Integration Architecture
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

### A2A Protocol Architecture
```
┌─────────────────┐    A2A Protocol    ┌─────────────────┐    ┌─────────────────┐
│   Host Agent    │◄──────────────────►│   A2A Server    │    │   SQL Database  │
│                 │                    │                 │───▶│   (Azure SQL)   │
│ ┌─────────────┐ │                    │ ┌─────────────┐ │    │                 │
│ │ Semantic    │ │                    │ │ Orchestrator│ │    │                 │
│ │ Kernel      │ │                    │ │ Agent       │ │    │                 │
│ │ Client      │ │                    │ │ Executor    │ │    │                 │
│ └─────────────┘ │                    │ └─────────────┘ │    │                 │
└─────────────────┘                    │                 │    │                 │
                                       │ ┌─────────────┐ │    │                 │
                                       │ │ MCP Plugin  │ │    │                 │
                                       │ └─────────────┘ │    │                 │
                                       └─────────────────┘    └─────────────────┘
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

### Core NL2SQL Features
- ✅ **Natural Language Processing**: Convert questions to SQL using AI
- ✅ **Real-time Schema Discovery**: Dynamic database introspection via MCP
- ✅ **Intelligent Query Generation**: Context-aware SQL with proper joins
- ✅ **Query Execution**: Direct database query execution
- ✅ **Result Explanation**: AI-generated explanations of queries and results
- ✅ **Error Handling**: Comprehensive error reporting and validation
- ✅ **Interactive Mode**: Command-line interface for testing

### A2A (Agent-to-Agent) Features
- ✅ **A2A Protocol Compliance**: Full Agent-to-Agent communication protocol
- ✅ **Streaming Responses**: Real-time response streaming for long queries
- ✅ **Host Agent Integration**: Client-side agent for external system integration
- ✅ **Semantic Kernel Client**: High-level client wrapper with conversation support
- ✅ **Session Management**: Persistent connections and session handling
- ✅ **Production Ready**: Scalable server architecture for enterprise deployment

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

### Direct Usage (Traditional)

#### Command Line Interface

```bash
python src/main.py
```

#### Programmatic Usage

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

### A2A Usage (Agent-to-Agent Protocol)

#### Start A2A Server

```bash
# Production server
cd src/
python pure_a2a_server.py --host localhost --port 8002

# Or component server
python a2a_server.py
```

#### Host Agent Client

```python
from src.Host_Agent.nl2sql_client_agent import SemanticKernelNL2SQLAgent

async def main():
    # Initialize A2A client agent
    agent = SemanticKernelNL2SQLAgent("http://localhost:8002")
    await agent.initialize()
    
    # Chat interface
    response = await agent.chat("Show me the top performing regions")
    print(response)
    
    await agent.close()

asyncio.run(main())
```

#### Interactive A2A Testing

```bash
cd src/Host_Agent/
python test_nl2sql_client.py
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
NL2SQL-A2A/
├── src/
│   ├── main.py                 # Main application entry point
│   ├── pure_a2a_server.py      # Production A2A server
│   ├── a2a_server.py           # A2A server component
│   ├── api_server.py           # REST API server
│   ├── start_a2a.py            # A2A server startup script
│   ├── start_api.py            # API server startup script
│   │
│   ├── agents/                 # Agent implementations
│   │   ├── orchestrator_agent.py      # Main orchestrator
│   │   ├── sql_generator_agent.py     # SQL generation
│   │   ├── sql_executor_agent.py      # Query execution
│   │   └── summarizing_agent.py       # Result summarization
│   │
│   ├── a2a_executors/          # A2A protocol executors
│   │   └── orchestrator_executor.py   # A2A orchestrator executor
│   │
│   ├── Host_Agent/             # Client-side A2A integration
│   │   ├── nl2sql_client_agent.py     # A2A client agent
│   │   ├── remote_agent_connection.py # A2A protocol client
│   │   ├── test_nl2sql_client.py      # Client testing suite
│   │   └── README.md                   # Host Agent documentation
│   │
│   ├── plugins/
│   │   └── mcp_database_plugin.py      # MCP integration plugin
│   │
│   ├── services/
│   │   ├── schema_service.py           # Database schema management
│   │   └── nl2sql_service.py           # Core NL2SQL conversion logic
│   │
│   └── templates/              # Jinja2 templates for prompts
│       ├── sql_generation.jinja2
│       ├── intent_analysis.jinja2
│       ├── insights_extraction.jinja2
│       ├── comprehensive_summary.jinja2
│       └── recommendations.jinja2
│
├── tests/                      # Test suites
│   ├── test_a2a_e2e.py         # End-to-end A2A tests
│   ├── test_a2a_simple.py      # Simple A2A tests
│   ├── test_nl2sql.py          # Core NL2SQL tests
│   └── test_schema.py          # Schema service tests
│
├── .vscode/
│   └── mcp.json                # MCP server configuration
│
├── Production_Deployment_Guide.md     # Production deployment guide
├── A2A_Integration_COMPLETE.md        # Complete A2A integration guide
├── A2A_Server_Pattern_Analysis.md     # A2A server analysis
├── Dockerfile                         # Container deployment
├── requirements.txt                   # Python dependencies
├── .env                              # Environment variables
└── README.md                         # This file
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

### Core Components

#### 1. **NL2SQLAgent** (`main.py`)
- Main orchestrator class for direct usage
- Handles initialization and coordination
- Provides high-level API for traditional usage

#### 2. **MCPDatabasePlugin** (`plugins/mcp_database_plugin.py`)
- MCP server integration
- Database operations (list tables, describe schema, execute queries)
- Uses fastmcp.Client for communication

#### 3. **SchemaService** (`services/schema_service.py`)
- Database schema discovery and caching
- Relationship mapping
- Business context management

#### 4. **NL2SQLService** (`services/nl2sql_service.py`)
- Core NL2SQL conversion logic
- AI-powered SQL generation
- Query execution and result processing

### A2A Components

#### 5. **OrchestratorAgent** (`agents/orchestrator_agent.py`)
- Main business logic orchestrator
- Coordinates SQL generation, execution, and summarization
- Manages the complete NL2SQL workflow

#### 6. **A2A Servers**
- **`pure_a2a_server.py`**: Production-ready standalone A2A server
- **`a2a_server.py`**: Component-based A2A server for integration

#### 7. **Host Agent** (`Host_Agent/`)
- **`nl2sql_client_agent.py`**: A2A client implementation
- **`remote_agent_connection.py`**: Low-level A2A protocol handling
- **`test_nl2sql_client.py`**: Comprehensive test suite

#### 8. **Specialized Agents**
- **`sql_generator_agent.py`**: Converts natural language to SQL
- **`sql_executor_agent.py`**: Executes SQL queries safely
- **`summarizing_agent.py`**: Generates business-friendly summaries

## 🚨 Error Handling

### Direct Usage
The system uses a "fail-fast" approach with no fallbacks:

- **AI Service Required**: Must configure OpenAI or Azure OpenAI
- **MCP Server Required**: Must be accessible and responding
- **Schema Discovery Required**: Database must be introspectable
- **Clear Error Messages**: All failures provide detailed diagnostics

### A2A Usage
Comprehensive error handling across the A2A protocol:

- **Connection Management**: Auto-retry with exponential backoff
- **Protocol Compliance**: Graceful error propagation
- **Timeout Handling**: Configurable timeouts for long-running queries
- **Session Recovery**: Automatic reconnection on network failures
- **Error Context**: Rich error information for debugging

## 🚀 Deployment Options

### Development
```bash
# Direct usage
python src/main.py

# A2A server
python src/pure_a2a_server.py --host localhost --port 8002
```

### Production

#### Docker Deployment
```bash
# Build container
docker build -t nl2sql-a2a .

# Run A2A server
docker run -p 8002:8002 nl2sql-a2a

# Or with environment variables
docker run -p 8002:8002 -e AZURE_OPENAI_API_KEY=your_key nl2sql-a2a
```

#### Server Deployment
See [Production Deployment Guide](Production_Deployment_Guide.md) for:
- Production server selection (`pure_a2a_server.py` vs `a2a_server.py`)
- Container orchestration
- Load balancing
- Monitoring and logging
- Security considerations

## 📚 Documentation

- **[A2A Integration Complete Guide](A2A_Integration_COMPLETE.md)**: Comprehensive A2A implementation guide
- **[Production Deployment Guide](Production_Deployment_Guide.md)**: Production deployment best practices
- **[A2A Server Pattern Analysis](A2A_Server_Pattern_Analysis.md)**: Technical analysis of server patterns
- **[Host Agent README](src/Host_Agent/README.md)**: Client-side integration guide

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