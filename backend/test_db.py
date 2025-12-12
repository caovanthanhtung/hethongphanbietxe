from database import detections_col
import asyncio
from datetime import datetime

async def test():
    res = await detections_col.insert_one({"test": True, "timestamp": datetime.utcnow()})
    print("Inserted:", res.inserted_id)
    cnt = await detections_col.count_documents({})
    print("Total docs:", cnt)

asyncio.run(test())
