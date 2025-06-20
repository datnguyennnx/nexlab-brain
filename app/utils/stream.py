import json
from typing import AsyncGenerator, List, Dict
from openai import AsyncOpenAI

async def stream_sse_from_openai(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    **kwargs,
) -> AsyncGenerator[str, None]:
    """
    A utility function to handle streaming Server-Sent Events (SSE) from OpenAI's
    chat completion API. It standardizes the event format.
    """
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        
        first_chunk = True
        message_id = None
        async for chunk in stream:
            # The first chunk contains metadata but no choices. Let's get the ID.
            if first_chunk:
                message_id = chunk.id
                yield f"data: {json.dumps({'type': 'assistant_message_start', 'data': {'id': message_id}})}\n\n"
                first_chunk = False

            # Subsequent chunks have content.
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content_chunk = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'type': 'content_chunk', 'data': {'message_id': message_id, 'content': content_chunk}})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': {'error': str(e)}})}\n\n"
    finally:
        # Signal the end of the stream to the client
        yield f"data: {json.dumps({'type': 'stream_end', 'data': {}})}\n\n" 