"""
Database connection manager for the Item Bank.

Uses psycopg2.connection pooling for efficient connection management.
"""

import contextlib
from typing import Generator, Optional
import psycopg2
from psycopg2 import pool
from psycopg2.extensions import connection
from loguru import logger

from src.config import load_config, DatabaseConfig


class DatabaseManager:
    """Manages PostgreSQL connection pooling for the Item Bank."""

    _pool: Optional[pool.SimpleConnectionPool] = None
    _config: Optional[DatabaseConfig] = None

    @classmethod
    def initialize(cls, config) -> None:
        """
        Initialize the connection pool with the given database configuration.

        Args:
            config: Config or DatabaseConfig instance with connection parameters
        """
        if cls._pool is not None:
            logger.warning("Database pool already initialized. Closing existing pool.")
            cls.close_all()

        # Handle both Config and DatabaseConfig objects
        # If it's a full Config object, extract the database config
        if hasattr(config, 'database'):
            db_config = config.database
        else:
            db_config = config

        cls._config = db_config

        try:
            # Parse the database URL
            # Expected format: postgresql://user:password@host:port/database
            db_url = db_config.url

            # Create connection pool
            cls._pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=db_config.pool_size + db_config.max_overflow,
                dsn=db_url
            )

            logger.info(
                f"Database connection pool initialized: "
                f"min=1, max={db_config.pool_size + db_config.max_overflow}"
            )

        except psycopg2.Error as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise

    @classmethod
    @contextlib.contextmanager
    def get_connection(cls) -> Generator[connection, None, None]:
        """
        Get a connection from the pool as a context manager.

        Automatically handles commit/rollback based on whether an exception occurs.

        Yields:
            psycopg2.connection: A database connection

        Raises:
            RuntimeError: If database pool is not initialized
            psycopg2.Error: If connection fails
        """
        if cls._pool is None:
            raise RuntimeError(
                "Database pool not initialized. Call DatabaseManager.initialize() first."
            )

        conn = None
        try:
            conn = cls._pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn is not None:
                conn.rollback()
            logger.error(f"Database operation failed, transaction rolled back: {e}")
            raise
        finally:
            if conn is not None:
                cls._pool.putconn(conn)

    @classmethod
    def close_all(cls) -> None:
        """Close all connections in the pool and reset the pool."""
        if cls._pool is not None:
            cls._pool.closeall()
            cls._pool = None
            logger.info("Database connection pool closed")

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the database pool has been initialized."""
        return cls._pool is not None


# Global database instance
db = DatabaseManager()


def init_database_from_config() -> None:
    """Initialize the database using the application config."""
    config = load_config()
    db.initialize(config.database)
