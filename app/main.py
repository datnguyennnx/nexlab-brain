from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
from .controllers.v1.router import api_router
from .database.session import AsyncSessionLocal
from .repositories.user_repository import UserRepository

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    # On startup
    async with AsyncSessionLocal() as session:
        user_repo = UserRepository(session)
        user = await user_repo.get_by_id(1)
        if not user:
            logger.info("Default user not found, creating one.")
            await user_repo.create(id=1)
        else:
            logger.info("Default user found.")
    yield
    # On shutdown
    logger.info("Shutting down...")

app = FastAPI(title="Nexlab Brain API", lifespan=lifespan)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "ok"} 