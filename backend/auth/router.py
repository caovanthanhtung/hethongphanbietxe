# auth/router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from database import db
from auth.utils import hash_password, verify_password, create_access_token

auth_router = APIRouter()
users_col = db["users"]

class RegisterSchema(BaseModel):
    email: EmailStr
    password: str

class LoginSchema(BaseModel):
    email: EmailStr
    password: str

@auth_router.post("/register")
async def register(data: RegisterSchema):
    email = data.email.strip().lower()  # normalize email
    password = data.password.strip()

    if await users_col.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already exists")

    user = {
        "email": email,
        "password": hash_password(password),  # hash trực tiếp
        "role": "user",
        "is_active": True
    }

    await users_col.insert_one(user)
    return {"success": True, "message": "User registered successfully"}

@auth_router.post("/login")
async def login(data: LoginSchema):
    email = data.email.strip().lower()
    password = data.password.strip()

    user = await users_col.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({
        "sub": user["email"],
        "role": user["role"]
    })

    return {"access_token": token, "role": user["role"]}
