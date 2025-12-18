from fastapi import APIRouter
from database import db
from datetime import datetime

router = APIRouter(prefix="/api/contact", tags=["Contact"])
contact_col = db["contacts"]

@router.post("/")
async def contact(data: dict):
    data["created_at"] = datetime.utcnow()
    await contact_col.insert_one(data)
    return {"success": True}
