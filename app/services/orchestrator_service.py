import json
from typing import AsyncGenerator, List, Dict
from langfuse import observe
import logging

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

        system_prompt = """
        Bạn là chuyên gia phân tích ngôn ngữ cho chatbot chăm sóc khách hàng tại Viện thẩm mỹ Diva.
        Nhiệmvụ của bạn là phân tích câu hỏi của người dùng bằng tiếng Việt và phân loại thành một trong hai loại: `rag_query` hoặc `chat_query`.

        - `rag_query` được sử dụng khi người dùng cần thông tin cụ thể, thực tế phải được tra cứu trong cơ sở tri thức, chẳng hạn như hỏi về giá cả, dịch vụ, địa chỉ, hoặc hướng dẫn.
        - `chat_query` được sử dụng cho các cuộc trò chuyện thông thường, chẳng hạn như chào hỏi, cảm ơn, hoặc các cuộc trò chuyện xã giao.

        Chỉ trả lời bằng `rag_query` hoặc `chat_query`.
        """

        user_prompt = f"""
        Lịch sử hội thoại:
        {history_str}

        Câu hỏi người dùng:
        {user_query}
        """
        
        with langfuse.start_as_current_span(
            name="routing-decision",
            input={"query": user_query, "history": history}
        ) as span:
            try:
                response = await self.chat_service.client.chat.completions.create(
                    model=self.routing_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                    max_tokens=10,
                )
                decision = response.choices[0].message.content.strip()
                
                span.update(output=decision)

                if decision not in ["rag_query", "chat_query"]:
                    # Simple heuristic fallback
                    query_lower = user_query.lower().strip()
                    greeting_keywords = ['xin chào', 'chào', 'hello', 'hi', 'cảm ơn', 'thanks']
                    
                    if any(keyword in query_lower for keyword in greeting_keywords):
                        return "chat_query"
                    else:
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
        if not user_message.content or not user_message.content.strip():
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Tin nhắn không được để trống'}})}\\n\\n"
            return
            
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
                        except json.JSONDecodeError as e:
                            logging.warning(f"Failed to parse SSE chunk: {chunk[:100]}... Error: {e}")
                            pass
            finally:
                root_span.update_trace(output={"response": final_content})
                if final_content.strip():
                    try:
                        await self.message_repo.create(
                            content=final_content,
                            role=MessageRole.ASSISTANT,
                            conversation_id=conversation_id
                        )
                    except Exception as e:
                        logging.error(f"Failed to save assistant message: {e}") 