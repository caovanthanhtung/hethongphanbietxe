from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class User(BaseModel):
    email: EmailStr
    password: str
    role: str = "user"      
    is_active: bool = True
    created_at: datetime = datetime.utcnow()
