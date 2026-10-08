"""MongoDB-backed, session-scoped conversational memory."""
from datetime import datetime, timezone
import logging
from pymongo import ASCENDING, MongoClient
from pymongo.errors import PyMongoError
from config import (MONGODB_COLLECTION, MONGODB_DATABASE, MONGODB_SERVER_SELECTION_TIMEOUT_MS,
                    MONGODB_URI)

logger = logging.getLogger(__name__)

class MemoryUnavailableError(RuntimeError):
    """Raised when persistent conversation storage cannot be used."""

class ConversationMemory:
    def __init__(self):
        try:
            self.client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=MONGODB_SERVER_SELECTION_TIMEOUT_MS)
            self.client.admin.command("ping")
            self.collection = self.client[MONGODB_DATABASE][MONGODB_COLLECTION]
            self.collection.create_index([("session_id", ASCENDING), ("timestamp", ASCENDING)])
            logger.info("Connected to MongoDB database=%s collection=%s", MONGODB_DATABASE, MONGODB_COLLECTION)
        except PyMongoError as error:
            raise MemoryUnavailableError("MongoDB is unavailable; start MongoDB and configure MONGODB_URI.") from error

    def close(self) -> None:
        self.client.close()

    def add_message(self, session_id: str, role: str, content: str, sources: list[dict] | None = None) -> None:
        try:
            self.collection.insert_one({"session_id": session_id, "role": role, "content": content,
                                        "timestamp": datetime.now(timezone.utc), "sources": sources or []})
        except PyMongoError as error:
            raise MemoryUnavailableError("MongoDB could not persist conversation memory.") from error

    def history(self, session_id: str, window: int) -> list[dict]:
        try:
            messages = list(self.collection.find({"session_id": session_id}, {"_id": 0})
                            .sort("timestamp", -1).limit(window))
            return list(reversed(messages))
        except PyMongoError as error:
            raise MemoryUnavailableError("MongoDB could not read conversation memory.") from error
