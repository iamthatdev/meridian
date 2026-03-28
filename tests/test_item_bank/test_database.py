"""
Tests for the Item Bank database connection manager.

These tests require PostgreSQL to be available. Tests are skipped if
the database is not accessible.
"""

import os
import pytest
from psycopg2 import OperationalError
from psycopg2.extensions import connection

from src.config import DatabaseConfig
from src.item_bank.database import DatabaseManager, db


@pytest.fixture
def test_config():
    """Create a test database configuration."""
    # Try to get database URL from environment
    db_url = os.getenv("TEST_DATABASE_URL")
    if not db_url:
        # Fall back to regular database URL
        db_url = os.getenv("DATABASE_URL", "postgresql://localhost/test_db")

    return DatabaseConfig(
        url=db_url,
        pool_size=2,
        max_overflow=2
    )


@pytest.fixture
def clean_db(test_config):
    """Ensure database manager is clean before and after tests."""
    # Clean up before test
    if DatabaseManager.is_initialized():
        DatabaseManager.close_all()

    yield

    # Clean up after test
    if DatabaseManager.is_initialized():
        DatabaseManager.close_all()


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL") and not os.getenv("TEST_DATABASE_URL"),
    reason="PostgreSQL not configured"
)
class TestDatabaseManager:
    """Test DatabaseManager functionality."""

    def test_initialize(self, test_config, clean_db):
        """Test database initialization."""
        assert not DatabaseManager.is_initialized()

        DatabaseManager.initialize(test_config)

        assert DatabaseManager.is_initialized()

    def test_get_connection(self, test_config, clean_db):
        """Test getting a connection from the pool."""
        DatabaseManager.initialize(test_config)

        with DatabaseManager.get_connection() as conn:
            assert conn is not None
            assert isinstance(conn, connection)

            # Test that we can execute queries
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                assert result[0] == 1

    def test_connection_commit_on_success(self, test_config, clean_db):
        """Test that successful operations are committed."""
        DatabaseManager.initialize(test_config)

        with DatabaseManager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS test_table (
                        id SERIAL PRIMARY KEY,
                        value VARCHAR(50)
                    )
                """)
                cursor.execute("INSERT INTO test_table (value) VALUES ('test')")

        # Open a new connection to verify commit
        with DatabaseManager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM test_table")
                count = cursor.fetchone()[0]
                assert count == 1

            # Cleanup
            with conn.cursor() as cursor:
                cursor.execute("DROP TABLE test_table")

    def test_connection_rollback_on_error(self, test_config, clean_db):
        """Test that failed operations are rolled back."""
        DatabaseManager.initialize(test_config)

        # Create a test table first
        with DatabaseManager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS test_table (
                        id SERIAL PRIMARY KEY,
                        value VARCHAR(50) NOT NULL
                    )
                """)

        # Try to insert invalid data (should rollback)
        with pytest.raises(Exception):
            with DatabaseManager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("INSERT INTO test_table (value) VALUES ('valid')")
                    cursor.execute("INSERT INTO test_table (value) VALUES (NULL)")

        # Verify rollback occurred
        with DatabaseManager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM test_table")
                count = cursor.fetchone()[0]
                assert count == 0  # Both inserts should have been rolled back

            # Cleanup
            with conn.cursor() as cursor:
                cursor.execute("DROP TABLE test_table")

    def test_close_all(self, test_config, clean_db):
        """Test closing all connections."""
        DatabaseManager.initialize(test_config)
        assert DatabaseManager.is_initialized()

        DatabaseManager.close_all()

        assert not DatabaseManager.is_initialized()

    def test_multiple_connections(self, test_config, clean_db):
        """Test that multiple connections can be checked out."""
        DatabaseManager.initialize(test_config)

        conn1 = None
        conn2 = None

        try:
            conn1 = DatabaseManager.get_connection()
            conn2 = DatabaseManager.get_connection()

            # Both should be valid connection contexts
            with conn1 as c1:
                with conn2 as c2:
                    assert c1 is not None
                    assert c2 is not None
                    assert c1 is not c2  # Different connection objects

                    # Verify both work
                    with c1.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        assert cursor.fetchone()[0] == 1

                    with c2.cursor() as cursor:
                        cursor.execute("SELECT 2")
                        assert cursor.fetchone()[0] == 2
        finally:
            # Context managers handle cleanup
            pass

    def test_reinitialize_pool(self, test_config, clean_db):
        """Test reinitializing an existing pool."""
        DatabaseManager.initialize(test_config)

        # Reinitialize should close old pool and create new one
        DatabaseManager.initialize(test_config)

        assert DatabaseManager.is_initialized()

        # Should still work
        with DatabaseManager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                assert cursor.fetchone()[0] == 1

    def test_error_without_initialization(self, clean_db):
        """Test that get_connection raises error if not initialized."""
        assert not DatabaseManager.is_initialized()

        with pytest.raises(RuntimeError, match="not initialized"):
            with DatabaseManager.get_connection() as conn:
                pass


def test_global_db_instance():
    """Test that the global db instance exists."""
    from src.item_bank.database import db
    assert db is not None
    assert isinstance(db, DatabaseManager)


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL") and not os.getenv("TEST_DATABASE_URL"),
    reason="PostgreSQL not configured"
)
def test_database_unavailable():
    """Test behavior when database is unavailable."""
    bad_config = DatabaseConfig(
        url="postgresql://invalid:invalid@localhost:9999/invalid_db",
        pool_size=1,
        max_overflow=0
    )

    # Should raise OperationalError
    with pytest.raises(OperationalError):
        DatabaseManager.initialize(bad_config)
