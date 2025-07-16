# Host Agent - NL2SQL A2A Client

This directory contains the client-side implementation for connecting to the NL2SQL A2A (Agent-to-Agent) server using the `a2a.client` library and Semantic Kernel integration.

## Overview

The Host Agent provides a bridge between external applications/agents and the NL2SQL orchestrator through the A2A protocol. It enables natural language database queries through a standardized agent interface.

## Components

### 1. NL2SQL Client Agent (`nl2sql_client_agent.py`)

The main client implementation that connects to the A2A server:

#### `NL2SQLClientAgent`
- **Purpose**: Core A2A client for NL2SQL queries
- **Features**: 
  - Async connection management
  - Streaming response support
  - Session management
  - Error handling and retries
- **Key Methods**:
  - `initialize_connection()`: Establish A2A connection
  - `query_database(query)`: Execute NL2SQL query
  - `query_database_streaming(query)`: Stream responses
  - `get_capabilities()`: Get agent metadata

#### `SemanticKernelNL2SQLAgent`
- **Purpose**: High-level wrapper with Semantic Kernel integration
- **Features**:
  - Conversational interface
  - Auto-reconnection
  - Simplified API
  - Built-in prompt management
- **Key Methods**:
  - `initialize()`: Setup kernel and connection
  - `chat(message)`: Natural language conversation
  - `close()`: Clean shutdown

### 2. Remote Agent Connection (`remote_agent_connection.py`)

Handles low-level A2A protocol communication:
- Connection establishment
- Message serialization/deserialization
- Protocol compliance
- Error handling

### 3. Test Client (`test_nl2sql_client.py`)

Comprehensive testing suite for the client agents:
- Basic client functionality tests
- Semantic Kernel integration tests
- Streaming response tests
- Interactive testing mode

## Usage Examples

### Basic Client Usage

```python
from nl2sql_client_agent import NL2SQLClientAgent
from semantic_kernel import Kernel

# Create and initialize client
kernel = Kernel()
client = NL2SQLClientAgent(kernel, "http://localhost:8002")
await client.initialize_connection()

# Execute query
result = await client.query_database("Show me sales data for 2025")
print(result)

await client.close()
```

### Semantic Kernel Agent Usage

```python
from nl2sql_client_agent import SemanticKernelNL2SQLAgent

# Create and initialize agent
agent = SemanticKernelNL2SQLAgent("http://localhost:8002")
await agent.initialize()

# Chat interface
response = await agent.chat("What are the top performing regions?")
print(response)

await agent.close()
```

### Streaming Responses

```python
# Stream long-running queries
async for chunk in client.query_database_streaming(query):
    print(chunk, end='', flush=True)
```

## Setup and Configuration

### Prerequisites

1. **A2A Server Running**: The NL2SQL A2A server must be running
   ```bash
   cd src/
   python pure_a2a_server.py --host localhost --port 8002
   ```

2. **Dependencies**: Required packages installed
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

The client agents can be configured with:
- **Server URL**: Default `http://localhost:8002`
- **Timeout Settings**: Connection and request timeouts
- **Retry Logic**: Automatic retry on failures
- **Logging Level**: Debug, info, warning, error

### Environment Variables

Optional environment variables:
```bash
A2A_SERVER_URL=http://localhost:8002
A2A_TIMEOUT=30
A2A_MAX_RETRIES=3
LOG_LEVEL=INFO
```

## Testing

### Running Tests

```bash
# Interactive test suite
cd src/Host_Agent/
python test_nl2sql_client.py

# Or run specific tests
python -c "
import asyncio
from test_nl2sql_client import test_basic_client
asyncio.run(test_basic_client())
"
```

### Test Modes

1. **Basic Client Test**: Tests core A2A functionality
2. **Semantic Kernel Test**: Tests SK integration
3. **Streaming Test**: Tests streaming responses
4. **Interactive Mode**: Manual testing interface
5. **All Tests**: Comprehensive test suite

### Example Test Queries

- "Show me sales data for 2025"
- "What are the top 5 products by revenue?"
- "Which regions have the highest growth?"
- "Find customers who haven't purchased in 6 months"
- "Analyze monthly sales trends this year"

## Architecture

```
┌─────────────────────┐    A2A Protocol    ┌─────────────────────┐
│   Host Agent        │◄──────────────────►│   A2A Server        │
│                     │                    │                     │
│ ┌─────────────────┐ │                    │ ┌─────────────────┐ │
│ │ Semantic Kernel │ │                    │ │ Orchestrator    │ │
│ │ NL2SQL Agent   │ │                    │ │ Agent Executor  │ │
│ └─────────────────┘ │                    │ └─────────────────┘ │
│                     │                    │                     │
│ ┌─────────────────┐ │                    │ ┌─────────────────┐ │
│ │ NL2SQL Client   │ │                    │ │ SQL Generator   │ │
│ │ Agent           │ │                    │ │ SQL Executor    │ │
│ └─────────────────┘ │                    │ │ Summarizer      │ │
│                     │                    │ └─────────────────┘ │
└─────────────────────┘                    └─────────────────────┘
```

## Protocol Details

### A2A Communication

The client uses the standard A2A protocol for communication:

1. **Connection**: WebSocket-based persistent connection
2. **Authentication**: Agent card exchange
3. **Messaging**: JSON-based message format
4. **Streaming**: Real-time response streaming
5. **Error Handling**: Protocol-level error management

### Message Flow

```
Client → Server: Query Request
Server → Client: Processing Started
Server → Client: [Streaming Response Chunks]
Server → Client: Final Result
```

## Error Handling

The client implements comprehensive error handling:

- **Connection Errors**: Auto-retry with exponential backoff
- **Protocol Errors**: Graceful degradation
- **Timeout Handling**: Configurable timeouts
- **Server Errors**: Error propagation with context
- **Network Issues**: Resilient reconnection

## Monitoring and Logging

### Logging

Structured logging with configurable levels:
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### Metrics

The client tracks:
- Connection status
- Query response times
- Error rates
- Streaming performance

## Production Considerations

### Performance

- **Connection Pooling**: Reuse connections for multiple queries
- **Async Operations**: Non-blocking operations
- **Streaming**: Efficient for large responses
- **Caching**: Response caching for repeated queries

### Security

- **HTTPS**: Use secure connections in production
- **Authentication**: Implement proper auth mechanisms
- **Input Validation**: Sanitize all inputs
- **Rate Limiting**: Respect server rate limits

### Deployment

- **Container Support**: Docker-ready
- **Environment Config**: 12-factor app principles
- **Health Checks**: Built-in health monitoring
- **Graceful Shutdown**: Clean resource cleanup

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check A2A server is running
   - Verify URL and port
   - Check network connectivity

2. **Timeout Errors**
   - Increase timeout settings
   - Check server performance
   - Verify query complexity

3. **Protocol Errors**
   - Check A2A server version compatibility
   - Verify message format
   - Review server logs

### Debug Mode

Enable debug logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When extending the client agent:

1. **Follow Patterns**: Use existing async patterns
2. **Error Handling**: Implement comprehensive error handling
3. **Testing**: Add tests for new functionality
4. **Documentation**: Update this README
5. **Logging**: Add appropriate logging

## Related Documentation

- [A2A Integration Complete Guide](../../A2A_Integration_COMPLETE.md)
- [Production Deployment Guide](../../Production_Deployment_Guide.md)
- [A2A Server Analysis](../../A2A_Server_Pattern_Analysis.md)
- [Main README](../../README.md)
