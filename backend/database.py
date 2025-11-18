from motor.motor_asyncio import AsyncIOMotorClient
import os

# Nếu bạn chạy backend trên Windows và MongoDB chạy trong Docker:
# -> Phải dùng localhost, không dùng "mongo"
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")

client = AsyncIOMotorClient(MONGO_URL)

db = client["traffic_db"]
detections_col = db["detections"]
