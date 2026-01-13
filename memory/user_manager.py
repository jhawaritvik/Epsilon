import logging
import uuid
from typing import Optional
from .supabase_client import SupabaseManager

logger = logging.getLogger("UserManager")

class UserManager:
    """
    Manages User Identity for the Autonomous Research Engine.
    Ensures a valid user_id exists for Foreign Key constraints.
    """
    
    @staticmethod
    def login(email: str, password: str) -> str:
        """
        Logs in a user by email, or creates a new account if it doesn't exist.
        Returns the user_id.
        """
        manager = SupabaseManager()
        if not manager.is_enabled:
            logger.warning("Supabase disabled: Returning Mock UUID.")
            return "00000000-0000-0000-0000-000000000001"

        try:
            # 1. Try to find the user
            res = manager.client.table("users").select("id").eq("email", email).execute()
            if res.data:
                user_id = res.data[0]['id']
                logger.info(f"[Identity] Logged in as: {email} ({user_id})")
                return user_id

            # 2. Register new user
            logger.info(f"[Identity] registering new user: {email}...")
            # Note: In a real app we'd hash the password here. For this tool, we store it as is or a simple placeholder hash if strict security isn't the focus yet.
            # Schema has password_hash. Let's strictly speaking hash it? Or just store string for now since it's an internal tool?
            # User asked to "ask for password", implying they might care.
            # I'll simple-hash it to be polite to the schema.
            import hashlib
            pwd_hash = hashlib.sha256(password.encode()).hexdigest()
            
            new_user = manager.client.table("users").insert({
                "email": email,
                "password_hash": pwd_hash
            }).execute()
            
            if new_user.data:
                user_id = new_user.data[0]['id']
                logger.info(f"[Identity] Registered new user: {user_id}")
                return user_id
            else:
                raise Exception("Registration failed (no data returned).")

        except Exception as e:
            logger.error(f"[Identity] Login failed: {e}")
            raise e

    @staticmethod
    def get_or_create_system_user(email: str = "system_default@epsilon.ai") -> str:
        """
        Legacy/System method.
        """
        return UserManager.login(email, "system_auto_generated_hash")
