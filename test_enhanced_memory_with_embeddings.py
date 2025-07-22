"""
Enhanced Memory Test: NL2SQL with Vector Embeddings
This demonstrates how to add semantic search capabilities with embeddings
"""

import os
import sys
import asyncio
import time
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Set up environment for Azure Identity
os.environ['USE_AZURE_IDENTITY'] = 'true'
os.environ['COSMOS_DB_ENDPOINT'] = 'https://cosmos-acrasalesanalytics2.documents.azure.com'
os.environ['AZURE_TENANT_ID'] = '433ec967-f454-49f2-b132-d07f81545e02'

from main import NL2SQLMultiAgentSystem
from services.orchestrator_memory_service import OrchestratorMemoryService


class EnhancedMemoryNL2SQLSystem:
    """
    NL2SQL System with Vector Embeddings and Semantic Search
    """
    
    def __init__(self):
        self.nl2sql_system = None
        self.memory_service = None
        self.embedding_service = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize system with embedding service"""
        print("🚀 Initializing Enhanced Memory-NL2SQL System with Embeddings...")
        
        # Initialize NL2SQL system
        self.nl2sql_system = NL2SQLMultiAgentSystem()
        await self.nl2sql_system.initialize()
        
        # Get embedding service from the initialized system
        # Setup dedicated embedding service for memory
        print("🔧 Setting up dedicated embedding service for memory...")
        try:
            from semantic_kernel import Kernel
            from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
            from azure.identity import DefaultAzureCredential
            
            # Create embedding kernel
            embedding_kernel = Kernel()
            
            # Add embedding service
            embedding_service = AzureTextEmbedding(
                deployment_name="text-embedding-3-small",
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-12-01-preview"
            )
            
            embedding_kernel.add_service(embedding_service)
            print("✅ Dedicated embedding kernel created")
            
            # Initialize memory service with embedding kernel
            self.memory_service = OrchestratorMemoryService(kernel=embedding_kernel)
            self.memory_service.initialize()
            
        except Exception as e:
            print(f"❌ Failed to setup embedding service: {e}")
            # Fallback to memory service without embedding
            self.memory_service = OrchestratorMemoryService()
            self.memory_service.initialize()
        
        # Get embedding service from memory service
        if hasattr(self.memory_service, 'embedding_service') and self.memory_service.embedding_service:
            self.embedding_service = self.memory_service.embedding_service
            print("✅ Embedding service found from memory service and ready")
            print(f"🔧 Embedding service type: {type(self.embedding_service)}")
        else:
            print("⚠️ Embedding service not available in memory service - using text-only memory")
            self.embedding_service = None
        
        self.initialized = True
        print("🎉 Enhanced system ready!")
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using Azure OpenAI"""
        if not self.embedding_service:
            return None
        
        try:
            # Generate embedding using Semantic Kernel embedding service
            embedding_result = await self.embedding_service.generate_embeddings([text])
            if embedding_result is not None and len(embedding_result) > 0:
                return np.array(embedding_result[0])
        except Exception as e:
            print(f"⚠️ Embedding generation failed: {e}")
        
        return None
    
    async def store_conversation_with_embedding(
        self,
        user_id: str,
        session_id: str,
        question: str,
        assistant_response: str,
        metadata: dict = None
    ) -> str:
        """Store conversation with vector embedding"""
        
        # Generate embedding for the user question
        print("🧠 Generating embedding for semantic search...")
        question_embedding = await self.generate_embedding(question)
        
        # Store basic conversation
        entry_id = self.memory_service.store_conversation_memory(
            conversation_id=f"{user_id}/{session_id}",
            user_message=question,
            assistant_response=assistant_response,
            metadata=metadata
        )
        
        # Store embedding separately if generated
        if question_embedding is not None:
            await self.store_vector_embedding(
                entry_id=entry_id,
                user_id=user_id,
                session_id=session_id,
                text=question,
                embedding=question_embedding,
                metadata=metadata
            )
            print("✅ Vector embedding stored for semantic search")
        else:
            print("⚠️ No embedding generated - using text-only storage")
        
        return entry_id
    
    async def store_vector_embedding(
        self,
        entry_id: str,
        user_id: str,
        session_id: str,
        text: str,
        embedding: np.ndarray,
        metadata: dict = None
    ):
        """Store vector embedding document in Cosmos DB"""
        
        # Create embedding document
        embedding_doc = {
            "id": f"embedding_{entry_id}",
            "entry_id": entry_id,
            "user_id": user_id,
            "session_id": session_id,
            "text": text,
            "embedding": embedding.tolist(),  # Convert numpy array to list
            "timestamp": datetime.now().isoformat(),
            "document_type": "vector_embedding",
            "partition_key": f"{user_id}/{session_id}",
            "dimensions": len(embedding),
            "embedding_model": "text-embedding-3-small"
        }
        
        if metadata:
            embedding_doc["metadata"] = metadata
        
        try:
            # Store in the same container as conversation logs
            self.memory_service.chat_logger.container.create_item(
                body=embedding_doc
            )
            print(f"✅ Vector embedding stored: {len(embedding)} dimensions")
            
        except Exception as e:
            print(f"❌ Failed to store vector embedding: {e}")
    
    async def semantic_search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> list:
        """Find similar conversations using semantic search"""
        
        if not self.embedding_service:
            print("⚠️ Semantic search not available - no embedding service")
            return []
        
        # Generate embedding for search query
        query_embedding = await self.generate_embedding(query)
        if query_embedding is None:
            return []
        
        print(f"🔍 Performing semantic search for: {query[:50]}...")
        
        # This is where you'd implement Cosmos DB vector search
        # For now, return empty list as placeholder
        # In production, you'd use Cosmos DB vector search capabilities
        
        print("💡 Note: Full vector search requires Cosmos DB vector search preview features")
        return []
    
    async def process_query_with_enhanced_memory(
        self,
        user_id: str,
        question: str,
        session_id: str = None,
        execute: bool = True,
        limit: int = 10
    ) -> dict:
        """Process query with enhanced memory including embeddings"""
        
        if not self.initialized:
            raise ValueError("System not initialized")
        
        print(f"\n🤔 Processing with enhanced memory: {question}")
        
        # Check for similar previous questions using semantic search
        similar_conversations = await self.semantic_search(
            user_id=user_id,
            query=question,
            limit=3
        )
        
        if similar_conversations:
            print(f"🧠 Found {len(similar_conversations)} similar conversations")
        
        # Start or continue session
        if not session_id:
            session = self.memory_service.start_conversation(user_id)
            session_id = session.session_id
        
        # Process with NL2SQL system
        start_time = time.time()
        
        try:
            result = await self.nl2sql_system.ask_question(
                question=question,
                execute=execute,
                limit=limit,
                include_summary=True
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Format assistant response
            assistant_response = self._format_assistant_response(result)
            
            # Store with enhanced memory including embeddings
            entry_id = await self.store_conversation_with_embedding(
                user_id=user_id,
                session_id=session_id,
                question=question,
                assistant_response=assistant_response,
                metadata={
                    'processing_time_ms': processing_time_ms,
                    'success': result.get('success', False),
                    'sql_query': result.get('sql_query', ''),
                    'semantic_search_enabled': self.embedding_service is not None,
                    'similar_conversations_found': len(similar_conversations)
                }
            )
            
            result['enhanced_memory_info'] = {
                'session_id': session_id,
                'entry_id': entry_id,
                'embeddings_enabled': self.embedding_service is not None,
                'semantic_search_results': len(similar_conversations),
                'processing_time_ms': processing_time_ms
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _format_assistant_response(self, result: dict) -> str:
        """Format NL2SQL results into assistant response"""
        if not result.get('success', False):
            return f"Error: {result.get('error', 'Unknown error')}"
        
        parts = []
        if 'sql_query' in result:
            parts.append(f"SQL: {result['sql_query']}")
        if 'results' in result:
            parts.append(f"Returned {len(result.get('results', []))} rows")
        if 'summary' in result:
            parts.append(f"Summary: {result['summary']}")
        
        return " | ".join(parts) if parts else "Query processed"
    
    async def close(self):
        """Close connections"""
        if self.nl2sql_system:
            await self.nl2sql_system.close()
        if self.memory_service:
            self.memory_service.close()


async def main():
    """Test enhanced memory with embeddings"""
    
    print("🧪 Testing Enhanced Memory with Vector Embeddings")
    print("=" * 60)
    
    system = EnhancedMemoryNL2SQLSystem()
    
    try:
        await system.initialize()
        
        # Test with embedding capabilities
        test_user = "analyst_002"
        
        result = await system.process_query_with_enhanced_memory(
            user_id=test_user,
            question="What are the sales trends this quarter?",
            execute=True,
            limit=5
        )
        
        print("\n📊 Enhanced Memory Result:")
        print(f"Success: {result.get('success', False)}")
        if 'enhanced_memory_info' in result:
            memory_info = result['enhanced_memory_info']
            print(f"Embeddings Enabled: {memory_info.get('embeddings_enabled', False)}")
            print(f"Semantic Search Results: {memory_info.get('semantic_search_results', 0)}")
            print(f"Entry ID: {memory_info.get('entry_id', 'N/A')}")
        
        print("\n✅ Enhanced memory test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await system.close()


if __name__ == "__main__":
    asyncio.run(main())
