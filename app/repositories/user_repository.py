from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from ..models.user import User
from .base_repository import BaseRepository

class UserRepository(BaseRepository[User]):
    def __init__(self, db: Session):
        super().__init__(db)

    async def create(self, **kwargs) -> User:
        user = User(**kwargs)
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def get_by_id(self, id: int) -> Optional[User]:
        result = await self.db.execute(select(User).filter(User.id == id))
        return result.scalars().first() 