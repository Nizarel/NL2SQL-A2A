"""
Test script for GenericSchemaAnalystAgent
- Uses a mock schema provider and dummy kernel
- Validates core analysis and cache functionality
"""
import asyncio
import sys
import os
from typing import List, Dict, Any
from dataclasses import dataclass

# Ensure src is in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Import the agent and config
from agents.schema_analyst_agent import GenericSchemaAnalystAgent, SchemaAnalystConfig

# Dummy kernel and embedding service
class DummyKernel:
    def __init__(self):
        self.services = {}

class DummyEmbeddingService:
    async def generate_embeddings(self, texts: List[str]):
        # Return a simple vector for each text
        return [[float(len(t)) for t in text.split()] for text in texts]

# Mock schema provider implementing the protocol
class MockSchemaProvider:
    async def get_tables(self) -> List[str]:
        return ["users", "orders", "products"]

    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        schemas = {
            "users": {"columns": [{"name": "id"}, {"name": "name"}, {"name": "email"}]},
            "orders": {"columns": [{"name": "id"}, {"name": "user_id"}, {"name": "product_id"}]},
            "products": {"columns": [{"name": "id"}, {"name": "title"}, {"name": "price"}]},
        }
        return schemas.get(table_name, {"columns": []})

    async def get_relationships(self) -> Dict[str, Any]:
        return {
            "orders": [
                {"table": "users", "type": "many_to_one", "key": "user_id", "foreign_table": "users", "foreign_key": "id"},
                {"table": "products", "type": "many_to_one", "key": "product_id", "foreign_table": "products", "foreign_key": "id"},
            ]
        }

async def main():
    # Setup
    kernel = DummyKernel()
    kernel.services["embedding"] = DummyEmbeddingService()
    schema_provider = MockSchemaProvider()
    config = SchemaAnalystConfig(use_cache=True, cache_max_size=10)
    agent = GenericSchemaAnalystAgent(kernel, schema_provider, config)

    # Test input
    input_data = {
        "question": "Which users placed the most orders?",
        "context": "",
        "analysis_type": "full"
    }

    # Run analysis
    result = await agent.process(input_data)
    print("First run (should compute and cache):", result)

    # Run again to test cache hit
    result2 = await agent.process(input_data)
    print("Second run (should hit cache):", result2)

    # Test statistics
    stats = await agent.get_statistics()
    print("Agent statistics:", stats)

if __name__ == "__main__":
    asyncio.run(main())
