import os
from supabase import create_client, Client
import logging

logger = logging.getLogger("SupabaseMemory")

class SupabaseManager:
    _instance = None
    _client: Client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SupabaseManager, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            logger.warning("[Memory] SUPABASE_URL or SUPABASE_KEY not found in environment. Memory system disabled.")
            self._client = None
            return

        try:
            self._client = create_client(url, key)
            logger.info("[Memory] Supabase client initialized.")
        except Exception as e:
            logger.error(f"[Memory] Failed to initialize Supabase client: {e}")
            self._client = None

    @property
    def client(self):
        return self._client

    @property
    def is_enabled(self):
        return self._client is not None
