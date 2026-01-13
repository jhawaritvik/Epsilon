import threading

class ExecutionIdentity:
    """
    Singleton-like context for the current execution identity.
    Ensures agents do not need to manage or guess user_id.
    Thread-local storage is used to support concurrent runs in the future eventually,
    though currently we run single-process.
    """
    _local = threading.local()

    @classmethod
    def set_identity(cls, user_id: str):
        """
        Sets the user_id for the current execution context.
        Should be called by the Controller at start of run.
        """
        cls._local.user_id = user_id

    @classmethod
    def get_user_id(cls) -> str:
        """
        Retrieves the current user_id.
        Raises RuntimeError if identity is not initialized (Safety Guard).
        """
        if not hasattr(cls._local, "user_id") or cls._local.user_id is None:
            raise RuntimeError("CRITICAL: ExecutionIdentity accessed without valid user_id context. Controller must set identity.")
        return cls._local.user_id

    @classmethod
    def clear(cls):
        """Clears identity (e.g. at end of run)."""
        if hasattr(cls._local, "user_id"):
            del cls._local.user_id
