from typing import List, AsyncGenerator, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import json
import uuid
import re

from ..repositories.message_repository import MessageRepository
from ..models.message import MessageRole
from .embedding_service import embedding_service
from .generation_service import generation_service
from ..core.langfuse_client import langfuse


class RAGService:
    def __init__(self, message_repo: MessageRepository, db: AsyncSession):
        self.message_repo = message_repo
        self.db = db

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize query for PostgreSQL full-text search."""
        # Remove special FTS characters and join with AND
        sanitized = re.sub(r'[^\w\s]', ' ', query)
        words = [word for word in sanitized.split() if len(word) > 1]
        # Fallback for empty or very short queries to avoid FTS errors
        return ' & '.join(words) if words else 'a'

    def _extract_content_from_chunk(self, chunk: str) -> str:
        """Extract content from an SSE chunk."""
        if not chunk.strip().startswith('data:'):
            return ""
        try:
            data_str = chunk.strip()[5:]
            event_data = json.loads(data_str)
            if event_data.get('type') == 'content_chunk':
                return event_data.get('data', {}).get('content', '')
        except json.JSONDecodeError:
            pass
        return ""

    async def _hybrid_search(self, query: str, top_k: int = 5, k_val: int = 60) -> List[Dict]:
        """
        Performs hybrid search combining vector and full-text search using Reciprocal Rank Fusion (RRF).
        This method is optimized to run a single, combined query for efficiency.

        Args:
            query: The user's search query.
            top_k: The number of final documents to return.
            k_val: The RRF ranking constant. Higher values diminish the impact of rank differences.
                   Formula: RRF_score = sum(1 / (k_val + rank)) for each search method.

        Returns:
            A list of documents ranked by the RRF score.
        """
        with langfuse.start_as_current_span(name="hybrid-search-retrieval", input={"query": query[:100]}) as span:
            if not query or not query.strip():
                return []

            try:
                sanitized_query = self._sanitize_fts_query(query)
                query_embedding = embedding_service.encode([query])[0]

                hybrid_stmt = text("""
                    WITH vector_search AS (
                        SELECT id, content, filename,
                               (1 - (embedding <=> CAST(:query_embedding AS vector))) as vector_score,
                               ROW_NUMBER() OVER (ORDER BY (1 - (embedding <=> CAST(:query_embedding AS vector))) DESC) as vector_rank
                        FROM document_store
                        ORDER BY vector_score DESC
                        LIMIT 20
                    ),
                    fts_search AS (
                        SELECT id, content, filename,
                               ts_rank(content_tsv, websearch_to_tsquery('simple', :query)) AS fts_score,
                               ROW_NUMBER() OVER (ORDER BY ts_rank(content_tsv, websearch_to_tsquery('simple', :query)) DESC) as fts_rank
                        FROM document_store
                        WHERE content_tsv @@ websearch_to_tsquery('simple', :query)
                        ORDER BY fts_score DESC
                        LIMIT 20
                    )
                    SELECT
                        COALESCE(v.id, f.id) as id,
                        COALESCE(v.content, f.content) as content,
                        COALESCE(v.filename, f.filename) as filename,
                        (1.0 / (:k_val + COALESCE(v.vector_rank, 999)) + 1.0 / (:k_val + COALESCE(f.fts_rank, 999))) as rrf_score
                    FROM vector_search v
                    FULL OUTER JOIN fts_search f ON v.id = f.id
                    ORDER BY rrf_score DESC
                    LIMIT :top_k
                """)

                result = await self.db.execute(hybrid_stmt, {
                    "query_embedding": str(query_embedding.tolist()),
                    "query": sanitized_query,
                    "k_val": k_val,
                    "top_k": top_k
                })
                final_docs = [dict(doc) for doc in result.mappings().all()]

                span.update(output={
                    "results_count": len(final_docs),
                    "query_length": len(query),
                    "top_rrf_score": final_docs[0]['rrf_score'] if final_docs else 0,
                })
                return final_docs

            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                return []

    async def stream_rag_response(self, conversation_id: int, user_query: str, history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """
        Orchestrates the full RAG streaming pipeline with robust error handling and security.
        """
        if not user_query or not user_query.strip():
            error_event = {"type": "error", "data": {"message": "Câu hỏi không được để trống"}}
            yield f"data: {json.dumps(error_event)}\n\n"
            return

        with langfuse.start_as_current_span(name="rag-pipeline", input={"user_query": user_query, "history": history}) as span:
            retrieved_docs = await self._hybrid_search(user_query)

            if retrieved_docs:
                docs_for_client = [
                    {
                        "id": str(doc.get("id")),
                        "filename": doc.get("filename"),
                        "content": doc.get("content", ""),
                    }
                    for doc in retrieved_docs
                ]
                event_data = {"type": "retrieved_documents", "data": {"documents": docs_for_client}}
                yield f"data: {json.dumps(event_data)}\n\n"

            final_content = ""
            try:
                async for chunk in generation_service.stream_generate_response(user_query, retrieved_docs, history):
                    yield chunk
                    final_content += self._extract_content_from_chunk(chunk)
                span.update(output={"response": final_content})
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                error_event = {"type": "error", "data": {"message": "Đã có lỗi xảy ra khi tạo phản hồi. Vui lòng thử lại."}}
                yield f"data: {json.dumps(error_event)}\n\n"
                return 