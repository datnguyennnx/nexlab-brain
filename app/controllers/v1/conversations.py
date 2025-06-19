from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from ...database.session import get_db
from ...views.conversation import (
    ConversationCreate, 
    ConversationResponse, 
    ConversationPreviewResponse,
    ConversationUpdate
)
from ...views.message import MessageCreate, MessageResponse
from ...services.conversation_service import ConversationService
from ...services.openai_chat_service import OpenAIChatService
from ...repositories.conversation_repository import ConversationRepository
from ...repositories.message_repository import MessageRepository
from ...models.message import Message
from fastapi.responses import StreamingResponse
import asyncio
import json

router = APIRouter()

def get_conversation_service(db: AsyncSession = Depends(get_db)) -> ConversationService:
    conversation_repo = ConversationRepository(db)
    message_repo = MessageRepository(db)
    openai_chat_service = OpenAIChatService()
    return ConversationService(conversation_repo, message_repo, openai_chat_service)

@router.get("/", response_model=List[ConversationPreviewResponse])
async def get_user_conversations(
    service: ConversationService = Depends(get_conversation_service)
):
    # In a real app, user_id would come from an auth system.
    user_id = 1 
    return await service.get_user_conversations(user_id=user_id)

@router.post("/", response_model=ConversationResponse)
async def create_conversation(
    conversation: ConversationCreate,
    service: ConversationService = Depends(get_conversation_service)
):
    # For now, we'll assume a user_id is passed.
    # In a real app, you'd get this from auth.
    return await service.create_conversation(
        user_id=conversation.user_id,
        title=conversation.title
    )

@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: int,
    service: ConversationService = Depends(get_conversation_service)
):
    conversation = await service.get_conversation_with_messages(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation

@router.put("/{conversation_id}", response_model=ConversationPreviewResponse)
async def update_conversation_title(
    conversation_id: int,
    conversation_update: ConversationUpdate,
    service: ConversationService = Depends(get_conversation_service),
):
    conversation = await service.update_conversation_title(
        conversation_id=conversation_id, title=conversation_update.title
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation

@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: int,
    service: ConversationService = Depends(get_conversation_service),
):
    await service.delete_conversation(conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: int,
    service: ConversationService = Depends(get_conversation_service)
):
    conversation = await service.get_conversation_with_messages(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation.messages

@router.post("/{conversation_id}/stream")
async def stream_conversation(
    conversation_id: int,
    message: MessageCreate,
    service: ConversationService = Depends(get_conversation_service)
):
    # 1. Save user message
    await service.message_repo.create(
        content=message.content,
        role=message.role,
        conversation_id=conversation_id
    )
    
    # 2. Stream AI response
    return StreamingResponse(
        service.stream_ai_response(conversation_id),
        media_type="text/event-stream"
    ) 