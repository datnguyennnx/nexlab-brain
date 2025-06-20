from typing import List, AsyncGenerator, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import json

from ..repositories.message_repository import MessageRepository
from ..models.message import MessageRole
from .embedding_service import embedding_service
from .generation_service import generation_service
from ..core.langfuse_client import langfuse


class RAGService:
    def __init__(self, message_repo: MessageRepository, db: AsyncSession):
        self.message_repo = message_repo
        self.db = db
        self.memory_window_size = 10  # Keep the last 10 messages for context

    async def _hybrid_search(self, query: str, top_k: int = 5, k_val: int = 60) -> List[Dict]:
        """
        Performs hybrid search combining vector and full-text search using Reciprocal Rank Fusion (RRF).
        """
        with langfuse.start_as_current_span(name="hybrid-search-retrieval", input={"query": query}) as span:
            # 1. Embed the user's query
            query_embedding = embedding_service.encode([query])[0]

            # 2. Vector Search
            vector_stmt = text("""
                SELECT content, (1 - (embedding <=> :query_embedding)) as score
                FROM document_store
                ORDER BY score DESC
                LIMIT 10
            """)
            vector_result = await self.db.execute(
                vector_stmt,
                {"query_embedding": str(query_embedding.tolist())}
            )
            vector_docs = [dict(doc) for doc in vector_result.mappings().all()]

            # 3. Full-Text Search
            fts_stmt = text("""
                SELECT content, ts_rank(content_tsv, websearch_to_tsquery('simple', :query)) AS score
                FROM document_store
                WHERE content_tsv @@ websearch_to_tsquery('simple', :query)
                ORDER BY score DESC
                LIMIT 10
            """)
            fts_result = await self.db.execute(fts_stmt, {"query": query})
            fts_docs = [dict(doc) for doc in fts_result.mappings().all()]

            # 4. Reciprocal Rank Fusion (RRF)
            rrf_scores = {}
            all_docs = {}

            # Process vector search results
            for rank, doc in enumerate(vector_docs):
                content = doc['content']
                if content not in rrf_scores:
                    rrf_scores[content] = 0
                    all_docs[content] = doc
                rrf_scores[content] += 1 / (k_val + rank)

            # Process FTS results
            for rank, doc in enumerate(fts_docs):
                content = doc['content']
                if content not in rrf_scores:
                    rrf_scores[content] = 0
                    all_docs[content] = doc
                rrf_scores[content] += 1 / (k_val + rank)

            # Sort by RRF score
            sorted_content = sorted(
                rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

            # Get the final documents in the new sorted order
            final_docs = [all_docs[content] for content in sorted_content][:top_k]

            # Create a meaningful trace output for LLMOps
            trace_output = {
                "search_results": {
                    "vector": vector_docs,
                    "full_text": fts_docs,
                },
                "rrf_scores": rrf_scores,
                "final_documents_for_llm": final_docs,
            }

            span.update(output=trace_output)
            return final_docs

    async def stream_rag_response(self, conversation_id: int, user_query: str, history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """
        Orchestrates the full RAG streaming pipeline.
        """
        with langfuse.start_as_current_span(name="rag-pipeline", input={"user_query": user_query, "history": history}) as span:
            # Perform hybrid search with RRF
            retrieved_docs = await self._hybrid_search(user_query)

            # Stream the response from the generation service
            final_content = ""
            try:
                # The underlying LLM call is already traced by the OpenAI integration.
                # This observation serves as a logical grouping for the RAG response generation.
                async for chunk in generation_service.stream_generate_response(user_query, retrieved_docs, history):
                    yield chunk
                    # Accumulate content to save at the end
                    if chunk.strip().startswith('data:'):
                        try:
                            data_str = chunk.strip()[5:]
                            event_data = json.loads(data_str)
                            if event_data.get('type') == 'content_chunk':
                                final_content += event_data.get('data', {}).get('content', '')
                        except json.JSONDecodeError:
                            pass
                span.update(output={"response": final_content})
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                # The orchestrator will handle saving the message.
                raise 