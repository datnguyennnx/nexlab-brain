from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from sqlalchemy.orm import Session

ModelType = TypeVar("ModelType")

class BaseRepository(ABC, Generic[ModelType]):
    def __init__(self, db: Session):
        self.db = db

    @abstractmethod
    async def create(self, **kwargs) -> ModelType:
        pass

    @abstractmethod
    async def get_by_id(self, id: int) -> ModelType | None:
        pass 