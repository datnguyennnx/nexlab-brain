from fastapi import APIRouter
from . import conversations
 
api_router = APIRouter()
api_router.include_router(conversations.router, prefix="/conversations", tags=["conversations"]) 