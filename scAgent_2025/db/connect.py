"""
Database connection utilities for scAgent.
"""

import psycopg2
import psycopg2.extras
from typing import Optional, Dict, Any
import logging
from dynaconf import Dynaconf

logger = logging.getLogger(__name__)

# Load configuration
import os
from pathlib import Path

# Get the directory where this module is located
current_dir = Path(__file__).parent.parent
settings_file = current_dir / "settings.yml"

settings = Dynaconf(
    envvar_prefix="SCAGENT",
    settings_files=[str(settings_file), ".secrets.yml"],
    environments=True,
    load_dotenv=True,
)

def get_connection(
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    timeout: Optional[int] = None
) -> psycopg2.extensions.connection:
    """
    Create a PostgreSQL database connection.
    
    Args:
        host: Database host (defaults to config)
        port: Database port (defaults to config)
        user: Database user (defaults to config)
        password: Database password (defaults to config)
        database: Database name (defaults to config)
        timeout: Connection timeout (defaults to config)
        
    Returns:
        psycopg2 connection object
    """
    # Use provided values or fall back to config
    conn_params = {
        "host": host or settings.db_host,
        "port": port or settings.db_port,
        "user": user or settings.db_user,
        "password": password or settings.db_password,
        "database": database or getattr(settings, 'db_name', 'postgres'),
        "connect_timeout": timeout or settings.db_timeout,
    }
    
    try:
        logger.info(f"Connecting to database at {conn_params['host']}:{conn_params['port']}")
        conn = psycopg2.connect(**conn_params)
        conn.set_session(autocommit=True)
        logger.info("Database connection established successfully")
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

def test_connection(
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Test database connection and return connection info.
    
    Returns:
        Dict containing connection status and info
    """
    try:
        conn = get_connection(host, port, user, password, database, timeout)
        
        with conn.cursor() as cur:
            # Get database version
            cur.execute("SELECT version();")
            db_version = cur.fetchone()[0]
            
            # Get current database name
            cur.execute("SELECT current_database();")
            current_db = cur.fetchone()[0]
            
            # Get available tables
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = [row[0] for row in cur.fetchall()]
            
        conn.close()
        
        return {
            "status": "success",
            "database": current_db,
            "version": db_version,
            "tables": tables,
            "connection_params": {
                "host": host or settings.db_host,
                "port": port or settings.db_port,
                "user": user or settings.db_user,
                "database": database or getattr(settings, 'db_name', 'postgres'),
            }
        }
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "connection_params": {
                "host": host or settings.db_host,
                "port": port or settings.db_port,
                "user": user or settings.db_user,
                "database": database or getattr(settings, 'db_name', 'postgres'),
            }
        }

def get_cursor(conn: psycopg2.extensions.connection, cursor_factory=None):
    """
    Get a cursor from the connection.
    
    Args:
        conn: Database connection
        cursor_factory: Optional cursor factory (e.g., RealDictCursor)
        
    Returns:
        Database cursor
    """
    if cursor_factory:
        return conn.cursor(cursor_factory=cursor_factory)
    return conn.cursor()

def close_connection(conn: psycopg2.extensions.connection):
    """
    Close database connection safely.
    
    Args:
        conn: Database connection to close
    """
    try:
        if conn and not conn.closed:
            conn.close()
            logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing connection: {e}") 