from fastapi import APIRouter, Depends
from auth.deps import admin_required
from database import db

router = APIRouter(prefix="/api/admin", tags=["Admin"])
users_col = db["users"]

@router.get("/users")
async def get_users(user=Depends(admin_required)):
    users = []
    async for u in users_col.find({}, {"password": 0}):
        u["_id"] = str(u["_id"])
        users.append(u)
    return users
