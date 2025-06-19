from pydantic import BaseModel

class UserBase(BaseModel):
    # Add user fields here
    pass

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    id: int

    class Config:
        from_attributes = True
