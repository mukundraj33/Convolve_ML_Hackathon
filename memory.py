"""MongoDB-backed, bounded conversational memory."""
from datetime import datetime, timezone
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
from config import MONGODB_COLLECTION, MONGODB_DATABASE, MONGODB_URI

class MemoryUnavailableError(RuntimeError): pass

class ConversationMemory:
    def __init__(self):
        try:
            self.client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
            self.client.admin.command("ping")
            self.collection = self.client[MONGODB_DATABASE][MONGODB_COLLECTION]
            self.collection.create_index([("session_id", ASCENDING), ("timestamp", ASCENDING)])
        except PyMongoError as error:
            raise MemoryUnavailableError("MongoDB is unavailable; configure MONGODB_URI and start MongoDB.") from error
    def add_message(self, session_id, role, content, sources=None):
        try:
            self.collection.insert_one({"session_id": session_id, "role": role, "content": content,
                "timestamp": datetime.now(timezone.utc), "sources": sources or []})
        except PyMongoError as error: raise MemoryUnavailableError("MongoDB could not persist conversation memory.") from error
    def history(self, session_id, window):
        try:
            messages = list(self.collection.find({"session_id": session_id}, {"_id": 0}).sort("timestamp", -1).limit(window))
            return list(reversed(messages))
        except PyMongoError as error: raise MemoryUnavailableError("MongoDB could not read conversation memory.") from error
