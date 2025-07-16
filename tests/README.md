# A2A End-to-End Testing Guide

This guide explains how to test the NL2SQL A2A (Agent-to-Agent) integration end-to-end.

## Overview

The A2A server exposes the NL2SQL Orchestrator Agent through the A2A protocol, allowing external AI agents to communicate with it using standardized messaging.

## Test Queries

The following test queries are used to validate the complete NL2SQL workflow:

1. `Analyze revenue by region and show which region performs best in 2025?`
2. `Show the top performing distribution centers (CEDIs) by total sales in 2025`
3. `Generate a query to find customers who haven't made purchases in the last 6 months?`
4. `Which products have declining sales trends and in which regions in May 2025?`
5. `What are the top 5 products by sales in the last quarter?`

## Running the Tests

### Prerequisites

1. Make sure you're in the virtual environment:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 1: Start the A2A Server

```bash
cd src
python start_a2a.py
```

The server will start on `http://localhost:8001` and you should see:
```
✅ A2A Server initialized successfully!
🔗 A2A Endpoint: http://localhost:8001/
📋 Agent Card: http://localhost:8001/agent-card
```

### Step 2: Run the End-to-End Tests

In a new terminal (keep the server running):

```bash
cd tests
python test_a2a_simple.py
```

## Test Results

The test will validate:

1. **Server Health**: Checks if the A2A server is running and healthy
2. **Agent Card**: Validates the agent card structure and content
3. **A2A Messages**: Tests each query through the A2A protocol

Expected output:
```
🎯 A2A NL2SQL END-TO-END TEST RESULTS
================================================================================
⏱️  Total Execution Time: 15.2s
💚 Server Health: ✅
📋 Agent Card: ✅

📝 A2A MESSAGE TESTS:
   Total: 5
   Successful: 5
   Failed: 0
   Success Rate: 100.0%
   Avg Execution Time: 2.8s

🎉 OVERALL RESULT: ✅ ALL TESTS PASSED
```

## A2A Protocol Flow

Each test query goes through this workflow:

1. **JSON-RPC Request**: Client sends message via A2A protocol
2. **SQL Generation**: Agent analyzes query and generates SQL
3. **SQL Execution**: Generated SQL is executed (simulated)
4. **Summarization**: Results are analyzed and insights generated
5. **Response**: Complete results returned via A2A protocol

## Troubleshooting

### Server Not Starting
- Check if port 8001 is available
- Verify all dependencies are installed
- Check Python environment is activated

### Test Failures
- Ensure server is running before running tests
- Check server logs for error messages
- Verify network connectivity to localhost:8001

### Connection Issues
- Confirm firewall isn't blocking port 8001
- Try accessing http://localhost:8001/health directly
- Check if another process is using the port

## Manual Testing

You can also test manually using curl:

```bash
# Check health
curl http://localhost:8001/health

# Get agent card
curl http://localhost:8001/agent-card

# Send A2A message
curl -X POST http://localhost:8001/a2a/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "send_message",
    "params": {
      "message": {
        "messageId": "test-123",
        "role": "user",
        "parts": [{"type": "text", "text": "What are the top 5 products by sales?"}]
      }
    },
    "id": "test-123"
  }'
```

## Next Steps

- Integrate with external AI agents using the A2A SDK
- Connect to real database for production testing
- Implement streaming responses for real-time updates
- Add authentication and authorization for secure agent communication
