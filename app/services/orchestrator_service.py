import json
from typing import AsyncGenerator, List, Dict
from loguru import logger

from ..services.conversation_service import ConversationService
from ..services.openai_chat_service import OpenAIChatService
from ..models.message import Message, MessageRole


class OrchestratorService:
    def __init__(
        self,
        conversation_service: ConversationService,
        chat_service: OpenAIChatService,
    ):
        self.conversation_service = conversation_service
        self.chat_service = chat_service
        self.routing_model = "gpt-4o-mini" # Use a fast model for routing
        self.memory_window_size = 10 # Keep the last 10 messages for context

    async def _get_routing_decision(self, user_query: str, history: List[Dict[str, str]]) -> str:
        """
        Uses an LLM to decide whether to use the RAG pipeline or a simple chat response.
        """
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

        prompt = f"""
        Given the conversation history and the user's latest query, please classify the query into one of two categories:
        1. "rag_query": The user is asking a question that requires specific knowledge from a document base. Examples: "dịch vụ tắm trắng giá bao nhiêu?", "làm thế nào để đặt lịch hẹn?", "cơ sở ở đâu?".
        2. "chat_query": The user is making a general conversational statement. Examples: "xin chào", "cảm ơn bạn", "bạn là ai?", or follow-up comments on the assistant's previous answer that don't ask for new information.

        Return ONLY "rag_query" or "chat_query" as the answer.

        Conversation History:
        {history_str}

        User Query:
        {user_query}

        Classification:
        """
        
        try:
            # We need a non-streaming call here to get the decision before proceeding
            response = await self.chat_service.client.chat.completions.create(
                model=self.routing_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10,
            )
            decision = response.choices[0].message.content.strip()
            logger.info(f"Routing decision: '{decision}' for query: '{user_query}'")
            if decision not in ["rag_query", "chat_query"]:
                logger.warning("Router returned an invalid decision. Defaulting to RAG.")
                return "rag_query"
            return decision
        except Exception as e:
            logger.error(f"Error getting routing decision: {e}. Defaulting to RAG.")
            return "rag_query"

    async def stream_response(
        self, conversation_id: int, user_message: Message
    ) -> AsyncGenerator[str, None]:
        """
        Orchestrates the response generation by routing between RAG and simple chat.
        """
        history = await self.conversation_service.message_repo.get_by_conversation_id(
            conversation_id
        )
        
        # The history includes the new user message, so we take the part before it.
        conversation_history_msgs = history[:-1]

        # Apply the memory window to the history
        windowed_history_msgs = conversation_history_msgs[-self.memory_window_size:]
        
        conversation_history_for_router = [
            {"role": msg.role.value, "content": msg.content}
            for msg in windowed_history_msgs
        ]

        # Get the routing decision
        routing_decision = await self._get_routing_decision(
            user_message.content, conversation_history_for_router
        )

        if routing_decision == "rag_query":
            logger.info(f"Routing to RAG for conversation {conversation_id}")
            # Use the existing RAG streaming logic
            async for chunk in self.conversation_service.stream_rag_response(
                conversation_id
            ):
                yield chunk
        else:
            logger.info(f"Routing to Chat for conversation {conversation_id}")
            # Use the simple chat service
            # We need to save the assistant's response at the end
            final_content = ""
            # The history for the chat model should include the user's latest message
            chat_history = conversation_history_for_router + [{"role": user_message.role.value, "content": user_message.content}]
            try:
                async for chunk in self.chat_service.stream_chat_completion(chat_history):
                    yield chunk
                    if chunk.strip().startswith('data:'):
                        try:
                            data_str = chunk.strip()[5:]
                            event_data = json.loads(data_str)
                            if event_data.get('type') == 'content_chunk':
                                final_content += event_data.get('data', {}).get('content', '')
                        except json.JSONDecodeError:
                            pass # Ignore non-json chunks
            finally:
                if final_content:
                    await self.conversation_service.message_repo.create(
                        content=final_content,
                        role=MessageRole.ASSISTANT,
                        conversation_id=conversation_id
                    ) 