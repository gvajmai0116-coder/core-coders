from typing import Any, Optional
import threading
import os
import atexit

from omagent_core.utils.registry import registry
from pydantic import Field, PrivateAttr
from redis import ConnectionPool, Redis
try:
    from redislite import Redis as RedisLite
except ImportError:
    class RedisLite:
        def __init__(self, *args, **kwargs):
            raise ImportError("redislite is not supported on Windows. Please install and run a local Redis server and set use_lite to False in container.yaml, or use SharedMemSTM.")

from .base import ConnectorBase


class SharedRedisLite:
    """Shared Redis implementation using RedisLite"""
    _instance = None
    _lock = threading.Lock()
    _DB_FILE = os.path.join(os.getcwd(), "shared_redis.db")

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                # Create shared RedisLite instance using fixed database file
                cls._instance = RedisLite(
                    dbfilename=cls._DB_FILE,
                    serverconfig={
                        'save': '',  # Disable persistence
                        'appendonly': 'no'  # Disable AOF
                    }
                )
                # Register cleanup function
                atexit.register(cls._cleanup)
        return cls._instance

    @classmethod
    def _cleanup(cls):
        """Clean up database files when the program exits"""
        if cls._instance is not None:
            cls._instance.shutdown()
            # Remove database file and settings file
            for suffix in ['', '.settings']:
                file_path = cls._DB_FILE + suffix
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass


@registry.register_connector()
class RedisConnector(ConnectorBase):
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    password: Optional[str] = Field(default=None)
    username: Optional[str] = Field(default=None)
    db: int = Field(default=0)
    use_lite: bool = Field(default=True)
    
    _client: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.use_lite:
            try:
                self._client = SharedRedisLite.get_instance()
            except ImportError:
                self._client = None
        else:
            pool = ConnectionPool(
                host=self.host,
                port=self.port,
                password=self.password,
                username=self.username,
                db=self.db,
                decode_responses=False,
            )
            self._client = Redis(connection_pool=pool)

    def check_connection(self) -> bool:
        """Check if Redis connection is valid by executing a simple ping command"""
        try:
            self._client.ping()
            return True
        except Exception as e:
            raise ConnectionError(f"Redis connection failed: {str(e)}")
