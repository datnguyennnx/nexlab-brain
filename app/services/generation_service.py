import os
from openai import AsyncOpenAI
from typing import List, Dict, AsyncGenerator
from loguru import logger
import json

from ..core.config import settings

# Initialize the async client, reading the API key from environment variables
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

class GenerationService:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def _build_prompt(self, user_query: str, context_docs: List[Dict], history: List[Dict[str, str]] = None) -> str:
        """Builds the prompt for the LLM."""
        context_str = "\n\n---\n\n".join([doc['content'] for doc in context_docs])

        history_str = ""
        if history:
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
            history_str = f"Lịch sử cuộc trò chuyện:\n{history_str}\n"

        prompt = f"""
        Bạn là một trợ lý ảo am hiểu về các tài liệu được cung cấp. Dựa vào lịch sử trò chuyện và bối cảnh dưới đây, hãy trả lời câu hỏi của người dùng một cách chính xác và thân thiện bằng tiếng Việt.
        Nếu thông tin không có trong bối cảnh, hãy nói rằng bạn không tìm thấy thông tin.

        {history_str}
        Bối cảnh tài liệu:
        {context_str}

        Câu hỏi của người dùng:
        {user_query}

        Câu trả lời của bạn:
        """
        return prompt

    async def generate_response(self, user_query: str, context_docs: List[Dict]) -> str:
        """
        Generates a response using the LLM with the user query and retrieved context.
        """
        if not context_docs:
            return "Xin lỗi, tôi không tìm thấy thông tin nào liên quan đến câu hỏi của bạn."

        prompt = self._build_prompt(user_query, context_docs)

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant fluent in Vietnamese."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3, # Lower temperature for more factual answers
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "Đã có lỗi xảy ra khi tạo câu trả lời. Vui lòng thử lại sau."

    async def stream_generate_response(self, user_query: str, context_docs: List[Dict], history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        """
        Generates and streams a response using the LLM with context and history.
        """
        if not context_docs:
            message = "Xin lỗi, tôi không tìm thấy thông tin nào liên quan đến câu hỏi của bạn."
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': message}})}\n\n"
            return

        prompt = self._build_prompt(user_query, context_docs, history)
        
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant fluent in Vietnamese."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500,
                stream=True,
            )
            
            first_chunk = True
            message_id = None
            async for chunk in stream:
                if first_chunk:
                    message_id = chunk.id
                    yield f"data: {json.dumps({'type': 'assistant_message_start', 'data': {'id': message_id}})}\n\n"
                    first_chunk = False

                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'type': 'content_chunk', 'data': {'message_id': message_id, 'content': content_chunk}})}\n\n"

        except Exception as e:
            logger.error(f"Error streaming from OpenAI: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': {'error': str(e)}})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'stream_end', 'data': {}})}\n\n"

# Singleton instance
generation_service = GenerationService() 