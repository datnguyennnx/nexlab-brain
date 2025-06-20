import json
from typing import AsyncGenerator, List, Dict
from openai import AsyncOpenAI
from loguru import logger
from langfuse import observe
from ..core.config import settings

class OpenAIChatService:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model

    @observe(as_type="generation")
    async def stream_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            
            first_chunk = True
            message_id = None
            accumulated_content = ""

            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                
                if first_chunk:
                    message_id = chunk.id
                    yield f"data: {json.dumps({'type': 'assistant_message_start', 'data': {'id': message_id}})}\n\n"
                    first_chunk = False
                
                if delta and delta.content:
                    yield f"data: {json.dumps({'type': 'content_chunk', 'data': {'message_id': message_id, 'content': delta.content}})}\n\n"
            
            yield f"data: {json.dumps({'type': 'stream_end', 'data': {}})}\n\n"

        except Exception as e:
            logger.error(f"Error streaming from OpenAI: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': {'error': str(e)}})}\n\n" 