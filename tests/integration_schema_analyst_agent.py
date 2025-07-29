import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "src", ".env"))
"""
Integration test for GenericSchemaAnalystAgent using real MCPDatabasePlugin
"""
import asyncio

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from services.schema_service import create_mcp_schema_service
from agents.schema_analyst_agent import GenericSchemaAnalystAgent, SchemaAnalystConfig

# Dummy kernel and embedding service (replace with your real kernel if needed)
class DummyKernel:
    def __init__(self):
        self.services = {}

class DummyEmbeddingService:
    async def generate_embeddings(self, texts):
        return [[float(len(t)) for t in text.split()] for text in texts]

async def main():

    # Configure your MCP connection here
    mcp_config = {
        "mcp_server_url": os.getenv("MCP_SERVER_URL")
        # Add other required fields if needed
    }
    schema_service = create_mcp_schema_service(mcp_config)

    kernel = DummyKernel()
    kernel.services["embedding"] = DummyEmbeddingService()
    config = SchemaAnalystConfig(use_cache=True, cache_max_size=10)
    agent = GenericSchemaAnalystAgent(kernel, schema_service, config)

    input_data = {
        "question": "List all customers who placed orders in the last month.",
        "context": "",
        "analysis_type": "full"
    }

    result = await agent.process(input_data)
    print("Analysis result:", result)

    stats = await agent.get_statistics()
    print("Agent statistics:", stats)

if __name__ == "__main__":
    asyncio.run(main())
