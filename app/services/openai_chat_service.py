import json
from typing import AsyncGenerator, List, Dict
from openai import AsyncOpenAI
from ..core.langfuse_client import langfuse
from ..core.config import settings
from ..utils.stream import stream_sse_from_openai

class OpenAIChatService:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model

    def _get_system_prompt(self) -> str:
        """Returns the general system prompt for chat interactions."""
        return """
        Bạn là trợ lý AI chăm sóc khách hàng chuyên nghiệp và thân thiện của Viện thẩm mỹ Diva. 
        Bạn phải luôn giao tiếp bằng tiếng Việt với giọng điệu ấm áp và chuyên nghiệp.
        Đối với các câu hỏi phức tạp hoặc yêu cầu thông tin cụ thể, hãy đề nghị người dùng hỏi rõ hơn để bạn có thể tra cứu thông tin chính xác.
        """

    async def stream_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Streams a chat completion response from OpenAI using the shared utility."""
        
        final_messages = [{"role": "system", "content": self._get_system_prompt()}] + messages

        with langfuse.start_as_current_generation(
            name="openai-chat-completion",
            input=final_messages,
            model=self.model,
            metadata=kwargs,
        ) as generation:
            final_content = ""
            try:
                async for chunk in stream_sse_from_openai(
                    self.client, self.model, final_messages, **kwargs
                ):
                    yield chunk
                    if chunk.strip().startswith('data:'):
                        try:
                            data_str = chunk.strip()[5:]
                            event_data = json.loads(data_str)
                            if event_data.get('type') == 'content_chunk':
                                final_content += event_data.get('data', {}).get('content', '')
                        except json.JSONDecodeError:
                            pass
                generation.update(output=final_content)
            except Exception as e:
                generation.update(level="ERROR", status_message=str(e))
                raise 