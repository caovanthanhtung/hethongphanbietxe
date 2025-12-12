from database import detections_col
import asyncio

async def ensure():
    await detections_col.create_index("timestamp")
    print("Index on 'timestamp' ensured")

asyncio.run(ensure())
