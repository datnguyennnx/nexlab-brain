import json
from typing import AsyncGenerator, List, Dict
from langfuse import observe

from ..services.rag_service import RAGService
from ..services.openai_chat_service import OpenAIChatService
from ..models.message import Message, MessageRole
from ..repositories.message_repository import MessageRepository
from ..core.langfuse_client import langfuse


class OrchestratorService:
    def __init__(
        self,
        rag_service: RAGService,
        chat_service: OpenAIChatService,
        message_repo: MessageRepository,
    ):
        self.rag_service = rag_service
        self.chat_service = chat_service
        self.message_repo = message_repo
        self.routing_model = "gpt-4o-mini"
        self.memory_window_size = 10

    async def _get_routing_decision(self, user_query: str, history: List[Dict[str, str]]) -> str:
        """
        Uses an LLM to decide whether to use the RAG pipeline or a simple chat response.
        """
        history_str = "\\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

        prompt = f"""
        You are an expert linguistic analyst for a customer service chatbot at Diva Beauty Salon (Viện thẩm mỹ Diva).
        Your task is to analyze the user's query in Vietnamese and classify it into one of two categories: `rag_query` or `chat_query`.

        Follow these steps to determine the classification:
        1.  **Analyze Query**: What is the core intent of the user's latest message? Are they asking for specific facts (price, location, how-to) or engaging in conversation?
        2.  **Evaluate against Categories**:
            - `rag_query`: The user needs specific, factual information that must be looked up in the knowledge base. Examples: "giá dịch vụ X là bao nhiêu?", "địa chỉ chi nhánh Y?", "hướng dẫn đặt lịch hẹn".
            - `chat_query`: The user is having a general conversation. Examples: "xin chào", "cảm ơn bạn", "bạn là ai vậy?".
        3.  **Conclusion**: Based on your analysis, provide ONLY the final classification (`rag_query` or `chat_query`) and nothing else.

        ---
        Conversation History:
        {history_str}

        User Query:
        {user_query}

        Classification:
        """
        
        with langfuse.start_as_current_span(
            name="routing-decision",
            input={"query": user_query, "history": history}
        ) as span:
            try:
                response = await self.chat_service.client.chat.completions.create(
                    model=self.routing_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=10,
                )
                decision = response.choices[0].message.content.strip()
                
                span.update(output=decision)

                if decision not in ["rag_query", "chat_query"]:
                    return "rag_query"
                return decision
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                return "rag_query"

    async def stream_response(
        self, conversation_id: int, user_message: Message
    ) -> AsyncGenerator[str, None]:
        """
        Orchestrates the response generation by routing between RAG and simple chat,
        and centrally handles saving the assistant's final message.
        """
        with langfuse.start_as_current_span(
            name="request-orchestration",
            input={"query": user_message.content}
        ) as root_span:
            root_span.update_trace(
                user_id="nexlab_user_test",
                session_id=str(conversation_id),
            )
            history = await self.message_repo.get_by_conversation_id(
                conversation_id
            )
            
            conversation_history_msgs = history[:-1]
            windowed_history_msgs = conversation_history_msgs[-self.memory_window_size:]
            
            history_for_llm = [
                {"role": msg.role.value, "content": msg.content}
                for msg in windowed_history_msgs
            ]

            routing_decision = await self._get_routing_decision(
                user_message.content, history_for_llm
            )
            
            final_content = ""
            try:
                if routing_decision == "rag_query":
                    stream_generator = self.rag_service.stream_rag_response(
                        conversation_id, user_message.content, history_for_llm
                    )
                else:
                    chat_history = history_for_llm + [{"role": user_message.role.value, "content": user_message.content}]
                    stream_generator = self.chat_service.stream_chat_completion(chat_history)

                # Stream response to the client and accumulate final content in parallel
                async for chunk in stream_generator:
                    yield chunk
                    if chunk.strip().startswith('data:'):
                        try:
                            data_str = chunk.strip()[5:]
                            event_data = json.loads(data_str)
                            if event_data.get('type') == 'content_chunk':
                                final_content += event_data.get('data', {}).get('content', '')
                        except json.JSONDecodeError:
                            pass
            finally:
                root_span.update_trace(output={"response": final_content})
                if final_content:
                    await self.message_repo.create(
                        content=final_content,
                        role=MessageRole.ASSISTANT,
                        conversation_id=conversation_id
                    ) 