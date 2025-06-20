import os
from openai import AsyncOpenAI
from typing import List, Dict, AsyncGenerator
from loguru import logger
import json
from langfuse import observe

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
            Bạn là trợ lý AI chăm sóc khách hàng chuyên nghiệp và thân thiện của Viện thẩm mỹ Diva. Mục tiêu chính của bạn là cung cấp các câu trả lời chính xác, hữu ích CHỈ dựa trên ngữ cảnh tài liệu được cung cấp. Bạn phải luôn giao tiếp bằng tiếng Việt với giọng điệu ấm áp và chuyên nghiệp.

            Hãy làm theo các bước sau để tạo phản hồi của bạn, đảm bảo tính nhất quán và chất lượng cao:

            ### 1. Phân tích ngữ cảnh và hiểu rõ yêu cầu

            * **Đọc hiểu sâu sắc**: Đọc kỹ toàn bộ "Bối cảnh tài liệu" được cung cấp. Xác định các thông tin chính, các chi tiết cụ thể và mối liên hệ giữa các phần thông tin để đảm bảo không bỏ sót bất kỳ dữ liệu quan trọng nào liên quan đến câu hỏi của người dùng.
            * **Xác định ý định người dùng**: Phân tích "Câu hỏi của người dùng" để hiểu rõ mục đích, nhu cầu và mong muốn thực sự của họ. Điều này giúp bạn định hướng tìm kiếm thông tin và xây dựng câu trả lời phù hợp nhất.
            * **Kiểm tra tính khả dụng của thông tin**: Tự đánh giá xem liệu tất cả thông tin cần thiết để trả lời câu hỏi có tồn tại trong "Bối cảnh tài liệu" hay không. Nếu thiếu, hãy chuẩn bị cho kịch bản không tìm thấy thông tin.

            ### 2. Xây dựng câu trả lời nhất quán và chính xác

            * **Tổng hợp thông tin**:
                * **Nếu ngữ cảnh chứa câu trả lời**: Thu thập tất cả các đoạn thông tin liên quan từ tài liệu. Sắp xếp chúng một cách logic để tạo thành một câu trả lời hoàn chỉnh, mạch lạc và dễ hiểu. Đảm bảo mọi luận điểm đều được hỗ trợ trực tiếp từ văn bản gốc.
                * **Nếu ngữ cảnh KHÔNG chứa câu trả lời**: Bạn phải trung thực thừa nhận rằng thông tin không có sẵn trong tài liệu được cung cấp. Tuyệt đối không suy đoán, không tạo ra thông tin mới hoặc cung cấp các câu trả lời chung chung không dựa trên ngữ cảnh.
            * **Xử lý yêu cầu đặt lịch hẹn**: Nếu "Câu hỏi của người dùng" thể hiện mong muốn **đặt lịch hẹn**, hãy chủ động và thân thiện giới thiệu ngay về ứng dụng **Diva Queen** như một công cụ tiện lợi để họ có thể tự quản lý và đặt lịch hẹn.

            ### 3. Định dạng phản hồi theo quy chuẩn

            * **Tuân thủ câu mở đầu**:
                * **Nếu tìm thấy câu trả lời**: Phản hồi của bạn BẮT BUỘC phải bắt đầu bằng cụm từ chính xác: "Đại diện cho bộ phận chăm sóc khách hàng của Viện thẩm mỹ Diva, chúng tôi xin trả lời câu hỏi của bạn như sau:"
                * **Nếu KHÔNG tìm thấy câu trả lời**: Phản hồi của bạn BẮT BUỘC phải là: "Đại diện cho bộ phận chăm sóc khách hàng của Viện thẩm mỹ Diva, chúng tôi rất tiếc chưa tìm thấy thông tin chính xác cho câu hỏi của bạn. Bạn có muốn chúng tôi hỗ trợ thêm về vấn đề khác không ạ?"
            * **Ngôn ngữ và giọng điệu**: Luôn sử dụng tiếng Việt chuẩn xác, lịch sự, chuyên nghiệp và ấm áp trong toàn bộ phản hồi.

            **QUY TẮC BẮT BUỘC CẦN KHẮC CỐT GHI TÂM:**

            * **KHÔNG BAO GIỜ bịa đặt thông tin**. Sự uy tín là tối quan trọng.
            * **KHÔNG BAO GIỜ cung cấp câu trả lời không được hỗ trợ trực tiếp và rõ ràng bởi "Bối cảnh tài liệu"**.
            * **LUÔN LUÔN tuân thủ đúng định dạng phản hồi bắt buộc**.
            * **LUÔN LUÔN phản hồi bằng tiếng Việt**.

            ---
            {history_str}
            Bối cảnh tài liệu:
            {context_str}

            Câu hỏi của người dùng:
            {user_query}

            Câu trả lời của bạn:
        """
        return prompt

    @observe(as_type="generation")
    async def generate_response(self, user_query: str, context_docs: List[Dict], **kwargs) -> str:
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
                temperature=0.1,
                max_tokens=3000,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "Đã có lỗi xảy ra khi tạo câu trả lời. Vui lòng thử lại sau."

    @observe(as_type="generation")
    async def stream_generate_response(self, user_query: str, context_docs: List[Dict], history: List[Dict[str, str]] = None, **kwargs) -> AsyncGenerator[str, None]:
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
                **kwargs,
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