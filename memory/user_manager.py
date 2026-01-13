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
        Logs in a user by email and password.
        Returns the user_id if successful, raises Exception otherwise.
        """
        manager = SupabaseManager()
        if not manager.is_enabled:
            logger.warning("Supabase disabled: Returning Mock UUID.")
            return "00000000-0000-0000-0000-000000000001"

        try:
            # 1. Try to find the user and their hash
            res = manager.client.table("users").select("id, password_hash").eq("email", email).execute()
            
            if not res.data:
                raise Exception("User not found via email.")
            
            user_record = res.data[0]
            stored_hash = user_record.get('password_hash')
            
            # 2. Verify Password
            import hashlib
            input_hash = hashlib.sha256(password.encode()).hexdigest()
            
            if input_hash == stored_hash:
                user_id = user_record['id']
                logger.info(f"[Identity] Logged in as: {email} ({user_id})")
                return user_id
            else:
                raise Exception("Invalid password.")

        except Exception as e:
            logger.error(f"[Identity] Login failed: {e}")
            raise e

    @staticmethod
    def register(email: str, password: str) -> str:
        """
        Registers a new user.
        Returns the user_id.
        """
        manager = SupabaseManager()
        if not manager.is_enabled:
            logger.warning("Supabase disabled: Returning Mock UUID.")
            return "00000000-0000-0000-0000-000000000001"
            
        try:
            # 1. Check if user already exists
            res = manager.client.table("users").select("id").eq("email", email).execute()
            if res.data:
                raise Exception(f"User {email} already exists.")

            # 2. Register new user
            logger.info(f"[Identity] Registering new user: {email}...")
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
            logger.error(f"[Identity] Registration failed: {e}")
            raise e

    @staticmethod
    def get_or_create_system_user(email: str = "system_default@epsilon.ai") -> str:
        """
        Legacy/System method.
        """
        return UserManager.login(email, "system_auto_generated_hash")
