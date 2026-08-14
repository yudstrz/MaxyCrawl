import os
import libsql_client
from dotenv import load_dotenv

load_dotenv()

TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")

_client = None
def get_db_client():
    global _client
    if _client is None:
        if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
            _client = libsql_client.create_client(
                url=TURSO_DATABASE_URL,
                auth_token=TURSO_AUTH_TOKEN
            )
        else:
            _client = libsql_client.create_client(
                url="file:local_maxycrawl.db"
            )
    return _client

async def init_db():
    client = get_db_client()
    if not client:
        print("Database client could not be initialized.")
        return
    
    try:
        # Table for Notebooks
        await client.execute('''
            CREATE TABLE IF NOT EXISTS notebooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for Sources (scraped data)
        await client.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notebook_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                title TEXT,
                content TEXT,
                status TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
            )
        ''')
        
        # Table for Chat Logs
        await client.execute('''
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notebook_id INTEGER,
                query TEXT,
                answer TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
            )
        ''')
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")

async def create_notebook(name: str):
    client = get_db_client()
    if not client:
        return None
    try:
        rs = await client.execute("INSERT INTO notebooks (name) VALUES (?)", [name])
        return {"id": rs.last_insert_rowid, "name": name}
    except Exception as e:
        print(f"Error creating notebook: {e}")
        return None

async def get_notebooks():
    client = get_db_client()
    if not client:
        return []
    try:
        rs = await client.execute("SELECT id, name, created_at FROM notebooks ORDER BY created_at DESC")
        return [{"id": row[0], "name": row[1], "created_at": row[2]} for row in rs.rows]
    except Exception as e:
        print(f"Error getting notebooks: {e}")
        return []

async def add_or_update_source(notebook_id: int, url: str, title: str, content: str, status: str):
    client = get_db_client()
    if not client:
        return None
    try:
        # Check if source exists for this notebook
        rs = await client.execute("SELECT id FROM sources WHERE notebook_id = ? AND url = ?", [notebook_id, url])
        if len(rs.rows) > 0:
            await client.execute(
                "UPDATE sources SET title = ?, content = ?, status = ?, last_updated = CURRENT_TIMESTAMP WHERE notebook_id = ? AND url = ?",
                [title, content, status, notebook_id, url]
            )
        else:
            await client.execute(
                "INSERT INTO sources (notebook_id, url, title, content, status) VALUES (?, ?, ?, ?, ?)",
                [notebook_id, url, title, content, status]
            )
        return True
    except Exception as e:
        print(f"Error adding/updating source: {e}")
        return False

async def get_sources(notebook_id: int):
    client = get_db_client()
    if not client:
        return []
    try:
        rs = await client.execute("SELECT id, url, title, status, last_updated FROM sources WHERE notebook_id = ? ORDER BY last_updated DESC", [notebook_id])
        return [{"id": row[0], "url": row[1], "title": row[2], "status": row[3], "last_updated": row[4]} for row in rs.rows]
    except Exception as e:
        print(f"Error getting sources: {e}")
        return []

async def get_notebook_content(notebook_id: int):
    """Returns combined text content for all successful sources in a notebook for RAG."""
    client = get_db_client()
    if not client:
        return ""
    try:
        rs = await client.execute("SELECT title, url, content FROM sources WHERE notebook_id = ? AND status = 'success'", [notebook_id])
        combined = []
        for row in rs.rows:
            combined.append(f"Title: {row[0]}\nURL: {row[1]}\nContent: {row[2]}")
        return "\n\n---\n\n".join(combined)
    except Exception as e:
        print(f"Error getting notebook content: {e}")
        return ""

async def delete_source(source_id: int, notebook_id: int):
    client = get_db_client()
    if not client:
        return False
    try:
        await client.execute("DELETE FROM sources WHERE id = ? AND notebook_id = ?", [source_id, notebook_id])
        return True
    except Exception as e:
        print(f"Error deleting source: {e}")
        return False

async def delete_notebook(notebook_id: int):
    client = get_db_client()
    if not client:
        return False
    try:
        await client.execute("DELETE FROM notebooks WHERE id = ?", [notebook_id])
        return True
    except Exception as e:
        print(f"Error deleting notebook: {e}")
        return False

async def get_source_detail(source_id: int, notebook_id: int):
    client = get_db_client()
    if not client:
        return None
    try:
        rs = await client.execute("SELECT id, url, title, content, status, last_updated FROM sources WHERE id = ? AND notebook_id = ?", [source_id, notebook_id])
        if rs.rows:
            row = rs.rows[0]
            return {"id": row[0], "url": row[1], "title": row[2], "content": row[3], "status": row[4], "last_updated": row[5]}
        return None
    except Exception as e:
        print(f"Error getting source detail: {e}")
        return None

async def log_chat(notebook_id: int, query: str, answer: str):
    client = get_db_client()
    if not client:
        return
    try:
        await client.execute(
            "INSERT INTO chat_logs (notebook_id, query, answer) VALUES (?, ?, ?)",
            [notebook_id, query, answer]
        )
    except Exception as e:
        print(f"Error logging chat: {e}")

