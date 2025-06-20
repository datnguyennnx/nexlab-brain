from typing import List, Optional, AsyncGenerator, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from loguru import logger

from ..models.conversation import Conversation
from ..repositories.conversation_repository import ConversationRepository
from ..repositories.message_repository import MessageRepository
from ..models.message import MessageRole
from .embedding_service import embedding_service
from .generation_service import generation_service
import json

class ConversationService:
    def __init__(self, 
                 conversation_repo: ConversationRepository, 
                 message_repo: MessageRepository):
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.memory_window_size = 10 # Keep the last 10 messages for context
    
    async def _hybrid_search(self, query: str, db: AsyncSession, top_k: int = 5, k_val: int = 60) -> List[Dict]:
        """
        Performs hybrid search combining vector and full-text search using Reciprocal Rank Fusion (RRF).
        """
        # 1. Embed the user's query
        query_embedding = embedding_service.encode([query])[0]

        # 2. Vector Search
        vector_stmt = text("""
            SELECT content, (1 - (embedding <=> :query_embedding)) as score
            FROM document_store
            ORDER BY score DESC
            LIMIT 10
        """)
        vector_result = await db.execute(
            vector_stmt, 
            {"query_embedding": str(query_embedding.tolist())}
        )
        vector_docs = vector_result.mappings().all()

        # 3. Full-Text Search
        fts_stmt = text("""
            SELECT content, ts_rank(content_tsv, websearch_to_tsquery('simple', :query)) AS score
            FROM document_store
            WHERE content_tsv @@ websearch_to_tsquery('simple', :query)
            ORDER BY score DESC
            LIMIT 10
        """)
        fts_result = await db.execute(fts_stmt, {"query": query})
        fts_docs = fts_result.mappings().all()

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
        sorted_content = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Get the final documents in the new sorted order
        final_docs = [all_docs[content] for content in sorted_content]
        
        return final_docs[:top_k]

    async def stream_rag_response(self, conversation_id: int) -> AsyncGenerator[str, None]:
        """
        Orchestrates the full RAG streaming pipeline.
        """
        history = await self.message_repo.get_by_conversation_id(conversation_id)
        if not history:
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Conversation history not found.'}})}\n\n"
            return

        # The last message is the user's current query
        user_query = history[-1].content
        
        # Get prior conversation history, applying the memory window.
        # The slice is `-(window_size + 1)` to get the window, plus `-1` to exclude the current user query.
        conversation_history_msgs = history[-(self.memory_window_size + 1):-1]
        conversation_history = [
            {"role": msg.role.value, "content": msg.content} 
            for msg in conversation_history_msgs
        ]

        # Perform hybrid search with RRF
        retrieved_docs = await self._hybrid_search(user_query, self.conversation_repo.db)

        # Stream the response from the generation service
        final_content = ""
        try:
            async for chunk in generation_service.stream_generate_response(user_query, retrieved_docs, conversation_history):
                yield chunk
                # Accumulate content to save at the end
                if chunk.strip().startswith('data:'):
                    try:
                        data_str = chunk.strip()[5:]
                        event_data = json.loads(data_str)
                        if event_data.get('type') == 'content_chunk':
                            final_content += event_data.get('data', {}).get('content', '')
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode SSE chunk: {chunk}")
        finally:
            # Save the complete assistant message once streaming is finished
            if final_content:
                await self.message_repo.create(
                    content=final_content,
                    role=MessageRole.ASSISTANT,
                    conversation_id=conversation_id
                )
    
    # --- Existing methods that are still needed ---
    async def create_conversation(self, user_id: int, title: Optional[str] = None) -> Conversation:
        return await self.conversation_repo.create(user_id=user_id, title=title)
    
    async def get_conversation_with_messages(self, conversation_id: int) -> Optional[Conversation]:
        return await self.conversation_repo.get_by_id(conversation_id)
    
    async def get_user_conversations(self, user_id: int) -> List[Conversation]:
        return await self.conversation_repo.get_by_user_id(user_id)
        
    async def delete_conversation(self, conversation_id: int) -> None:
        return await self.conversation_repo.delete(conversation_id)

    async def update_conversation_title(self, conversation_id: int, title: str) -> Optional[Conversation]:
        return await self.conversation_repo.update(conversation_id, title)