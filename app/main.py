from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
import logging
import sys

from .controllers.v1.router import api_router
from .database.session import AsyncSessionLocal
from .repositories.user_repository import UserRepository
from .core.langfuse_client import langfuse

# --- Loguru Intercept Handler ---
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logging():
    # Intercept everything at the root logger
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    # Configure Loguru to only show INFO and above
    logger.configure(handlers=[{"sink": sys.stdout, "serialize": False, "level": "INFO"}])
    # Set higher logging levels for noisy libraries to suppress their DEBUG messages
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("langfuse").setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # On startup
    async with AsyncSessionLocal() as session:
        user_repo = UserRepository(session)
        user = await user_repo.get_by_id(1)
        if not user:
            await user_repo.create(id=1)
    yield
    # On shutdown
    langfuse.flush()

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