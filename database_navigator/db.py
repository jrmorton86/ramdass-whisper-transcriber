"""Database connection and utilities."""
import psycopg2
import json
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from .config import settings


def get_connection():
    """Create a new database connection with Secrets Manager support."""
    try:
        return psycopg2.connect(
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            connect_timeout=10
        )
    except Exception as e:
        raise Exception(f"Failed to connect to database: {str(e)}")


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_document(source: str, s3_bucket: str, s3_key: str, file_name: str, 
                  content_text: str, content_type: str, metadata: dict = None):
    """Save extracted document content to database."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO documents 
                (file_id, source, s3_bucket, s3_key, file_name, content_text, content_type, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, created_at
                """,
                (None, source, s3_bucket, s3_key, file_name, content_text, content_type, 
                 json.dumps(metadata) if metadata else None)
            )
            return cur.fetchone()


def get_documents(limit: int = 100, offset: int = 0):
    """Retrieve documents from database."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, source, file_name, content_type, 
                       LEFT(content_text, 200) as preview, created_at
                FROM documents
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset)
            )
            return cur.fetchall()
