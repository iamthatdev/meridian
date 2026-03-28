#!/usr/bin/env python
"""
Initialize the Item Bank database schema.

This script reads the init.sql migration file and executes it against
the configured PostgreSQL database.
"""

import sys
from pathlib import Path

from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.item_bank.database import db


def main() -> int:
    """
    Initialize the database schema.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Load configuration
        config = load_config()
        logger.info(f"Initializing database for environment: {config.app_env}")

        # Initialize database connection
        db.initialize(config.database)

        # Read the migration SQL file
        migration_path = Path(__file__).parent.parent / "src" / "item_bank" / "migrations" / "init.sql"
        if not migration_path.exists():
            logger.error(f"Migration file not found: {migration_path}")
            return 1

        logger.info(f"Reading migration file: {migration_path}")
        with open(migration_path, "r") as f:
            sql = f.read()

        # Execute the migration
        logger.info("Executing database schema migration...")
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)

                # Count inserted domains
                cursor.execute("SELECT COUNT(*) FROM domains")
                domain_count = cursor.fetchone()[0]

                # Verify tables exist
                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name IN ('items', 'domains', 'calibration_log', 'review_records', 'audit_log')
                    ORDER BY table_name
                """)
                tables = [row[0] for row in cursor.fetchall()]

        logger.success(f"Database schema initialized successfully!")
        logger.info(f"Tables created: {', '.join(tables)}")
        logger.info(f"Default domains inserted: {domain_count}")

        return 0

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return 1

    finally:
        # Close all connections
        db.close_all()


if __name__ == "__main__":
    sys.exit(main())
