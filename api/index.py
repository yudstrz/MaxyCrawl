from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import trafilatura
from lxml import etree
from datetime import datetime, timezone
import asyncio
import os
from dotenv import load_dotenv
import numpy as np
import tiktoken
from openai import AsyncOpenAI
from api.database import (
    init_db, 
    create_notebook, 
    get_notebooks, 
    add_or_update_source, 
    get_sources, 
    get_notebook_content, 
    log_chat,
    delete_source,
    delete_notebook,
    get_source_detail
)

load_dotenv()

app = FastAPI(
    title="MaxyCrawl NotebookLM-Style API",
    description="RAG Knowledge Base & Website Crawling System",
    version="2.1.0"
)

@app.on_event("startup")
async def startup_event():
    await init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USER_AGENT = "MaxyCrawl-API/2.1 (+https://github.com/yudstrz/MaxyCrawl)"
TIMEOUT_SECONDS = 15
MAX_SITEMAPS_TO_FETCH = 30

# --- AI & RAG setup ---
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

def chunk_text(text: str, max_tokens: int = 500) -> list[str]:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(encoding.decode(chunk_tokens))
    return chunks

@app.post("/api/notebooks/{notebook_id}/chat", response_model=ChatResponse)
async def chat_with_notebook(notebook_id: int, req: ChatRequest):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI API key belum dikonfigurasi di .env")
        
    full_content = await get_notebook_content(notebook_id)
    if not full_content.strip():
        return ChatResponse(answer="Notebook ini belum memiliki dokumen atau data yang berhasil diimpor. Silakan tambahkan sumber pengetahuan terlebih dahulu.")
        
    chunks = chunk_text(full_content, max_tokens=500)
    
    # 2. Embed query
    query_response = await openai_client.embeddings.create(
        input=req.query,
        model="text-embedding-3-small"
    )
    query_vector = np.array(query_response.data[0].embedding, dtype=np.float32)
    
    # 3. Limit chunks to avoid extreme timeouts/costs
    chunks = chunks[:100] 
    
    embeddings = []
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        response = await openai_client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        embeddings.extend([data.embedding for data in response.data])
        
    # 4. Similarity search
    similarities = []
    for chunk, emb in zip(chunks, embeddings):
        v = np.array(emb, dtype=np.float32)
        sim = np.dot(query_vector, v) / (np.linalg.norm(query_vector) * np.linalg.norm(v))
        similarities.append((sim, chunk))
        
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [item[1] for item in similarities[:5]]
    
    # 5. Generate Answer
    context = "\n\n---\n\n".join(top_chunks)
    prompt = f"Anda adalah Asisten Pengetahuan spesifik untuk Notebook ini. Anda harus menjawab pertanyaan secara akurat HANYA berdasarkan konteks yang diberikan.\n\nKonteks Sumber Data:\n{context}\n\nPertanyaan: {req.query}\n\nJawaban:"
    
    completion = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Anda adalah Asisten Pengetahuan MaxyCrawl. Jawablah pertanyaan dengan ramah, jelas, dan akurat berdasarkan konteks dokumen yang diberikan. Jika informasi tidak ada di dalam konteks dokumen, katakan 'Maaf, informasi tersebut tidak ditemukan dalam sumber pengetahuan Notebook ini.' dan jangan mengarang informasi. Gunakan Bahasa Indonesia kecuali pengguna bertanya dalam bahasa lain."},
            {"role": "user", "content": prompt}
        ]
    )
    
    answer = completion.choices[0].message.content
    await log_chat(notebook_id, req.query, answer)
    
    return ChatResponse(answer=answer)

# --- NOTEBOOK API ---
class NotebookCreate(BaseModel):
    name: str

@app.get("/api/notebooks")
async def list_notebooks():
    return await get_notebooks()

@app.post("/api/notebooks")
async def create_new_notebook(data: NotebookCreate):
    notebook = await create_notebook(data.name)
    if not notebook:
        raise HTTPException(status_code=500, detail="Gagal membuat notebook")
    return notebook

@app.delete("/api/notebooks/{notebook_id}")
async def remove_notebook(notebook_id: int):
    success = await delete_notebook(notebook_id)
    if not success:
        raise HTTPException(status_code=500, detail="Gagal menghapus notebook")
    return {"status": "success", "message": "Notebook berhasil dihapus"}

@app.get("/api/notebooks/{notebook_id}/sources")
async def list_notebook_sources(notebook_id: int):
    return await get_sources(notebook_id)

@app.get("/api/notebooks/{notebook_id}/sources/{source_id}")
async def get_source(notebook_id: int, source_id: int):
    detail = await get_source_detail(source_id, notebook_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan")
    return detail

@app.delete("/api/notebooks/{notebook_id}/sources/{source_id}")
async def remove_source(notebook_id: int, source_id: int):
    success = await delete_source(source_id, notebook_id)
    if not success:
        raise HTTPException(status_code=500, detail="Gagal menghapus dokumen")
    return {"status": "success", "message": "Dokumen berhasil dihapus"}

@app.post("/api/notebooks/{notebook_id}/scrape")
async def scrape_to_notebook(notebook_id: int, url: str = Query(...)):
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            html = response.text
    except Exception as e:
        await add_or_update_source(notebook_id, url, "", "", "failed")
        return {"url": url, "status": "failed", "error": str(e)}

    title, content, status = "", "", "pending"
    metadata = trafilatura.extract_metadata(html, default_url=url)
    if metadata and metadata.title:
        title = metadata.title.strip()

    extracted = trafilatura.extract(
        html, url=url, include_tables=True, include_links=False, 
        include_images=False, output_format="markdown", deduplicate=True
    )
    if extracted and extracted.strip():
        content, status = extracted.strip(), "success"
    else:
        try:
            tree = etree.HTML(html.encode("utf-8"))
            if tree is not None:
                title_nodes = tree.xpath("//title/text()")
                if title_nodes and not title:
                    title = title_nodes[0].strip()
        except Exception:
            pass
        status = "failed" if not content else "success"

    if not title:
        title = url

    await add_or_update_source(notebook_id, url, title, content, status)
    return {"url": url, "title": title, "status": status}

class SitemapResponse(BaseModel):
    base_url: str
    total_urls: int
    urls: list[str]

@app.get("/api/sitemap", response_model=SitemapResponse)
async def discover_sitemap(url: str = Query(...)):
    raw_url = url.strip()
    if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
        raw_url = "https://" + raw_url
        
    base_url = raw_url.rstrip("/")
    sitemap_urls = []
    
    # 1. Check robots.txt
    robots_url = f"{base_url}/robots.txt"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
            resp = await client.get(robots_url, headers={"User-Agent": USER_AGENT})
            if resp.status_code == 200:
                for line in resp.text.splitlines():
                    if line.strip().lower().startswith("sitemap:"):
                        s_url = line.split(":", 1)[1].strip()
                        if s_url:
                            sitemap_urls.append(s_url)
    except Exception:
        pass

    # 2. Add common standard sitemap paths as fallbacks
    common_fallbacks = [
        f"{base_url}/sitemap.xml",
        f"{base_url}/sitemap_index.xml",
        f"{base_url}/wp-sitemap.xml"
    ]
    for fb in common_fallbacks:
        if fb not in sitemap_urls:
            sitemap_urls.append(fb)

    visited_sitemaps = set()
    all_page_urls = set()
    queue = sitemap_urls.copy()
    
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
        while queue and len(visited_sitemaps) < MAX_SITEMAPS_TO_FETCH:
            batch = queue[:10]
            queue = queue[10:]
            
            async def fetch_sitemap(u):
                if u in visited_sitemaps: return
                visited_sitemaps.add(u)
                try:
                    resp = await client.get(u, headers={"User-Agent": USER_AGENT})
                    if resp.status_code != 200: return
                    try:
                        root = etree.fromstring(resp.content)
                        ns = "http://www.sitemaps.org/schemas/sitemap/0.9"
                        if "sitemapindex" in root.tag:
                            for sitemap_elem in root.iter(f"{{{ns}}}sitemap"):
                                loc = sitemap_elem.find(f"{{{ns}}}loc")
                                if loc is not None and loc.text:
                                    queue.append(loc.text.strip())
                        else:
                            for url_elem in root.iter(f"{{{ns}}}url"):
                                loc = url_elem.find(f"{{{ns}}}loc")
                                if loc is not None and loc.text:
                                    page_link = loc.text.strip()
                                    if page_link:
                                        all_page_urls.add(page_link)
                    except Exception:
                        pass
                except Exception:
                    pass

            await asyncio.gather(*(fetch_sitemap(u) for u in batch))

    # If sitemap found nothing, at least return the base url
    if not all_page_urls:
        all_page_urls.add(base_url)

    url_list = sorted(list(all_page_urls))
    return SitemapResponse(base_url=base_url, total_urls=len(url_list), urls=url_list)

# --- UI TEMPLATE ---
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MaxyCrawl - AI Knowledge Assistant</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-base: #090d16;
                --bg-sidebar: #0e1424;
                --bg-card: #151d30;
                --bg-card-hover: #1e2942;
                --bg-input: #101728;
                --border: #222f4c;
                --border-focus: #4f46e5;
                --primary: #6366f1;
                --primary-hover: #4f46e5;
                --primary-subtle: rgba(99, 102, 241, 0.12);
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --text-dim: #64748b;
                --success: #10b981;
                --success-bg: rgba(16, 185, 129, 0.15);
                --error: #ef4444;
                --error-bg: rgba(239, 68, 68, 0.15);
                --warning: #f59e0b;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Plus Jakarta Sans', sans-serif;
                background-color: var(--bg-base);
                color: var(--text-main);
                height: 100vh;
                display: flex;
                overflow: hidden;
                position: relative;
            }
            
            /* Mobile Hamburger Button */
            .mobile-menu-btn {
                display: none;
                position: fixed;
                top: 0.75rem;
                left: 0.75rem;
                z-index: 1100;
                background: var(--bg-card);
                border: 1px solid var(--border);
                color: var(--text-main);
                width: 42px;
                height: 42px;
                border-radius: 0.5rem;
                font-size: 1.35rem;
                cursor: pointer;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            }
            .mobile-menu-btn:hover { background: var(--bg-card-hover); }
            
            /* Sidebar Overlay Backdrop (mobile only) */
            .sidebar-overlay {
                display: none;
                position: fixed;
                inset: 0;
                background: rgba(0,0,0,0.6);
                backdrop-filter: blur(2px);
                z-index: 999;
            }
            .sidebar-overlay.active { display: block; }
            
            /* Sidebar */
            .sidebar {
                width: 280px;
                background: var(--bg-sidebar);
                border-right: 1px solid var(--border);
                display: flex;
                flex-direction: column;
                padding: 1.25rem 1rem;
                flex-shrink: 0;
                transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                z-index: 1000;
            }
            .brand {
                font-size: 1.15rem;
                font-weight: 700;
                color: #a5b4fc;
                display: flex;
                align-items: center;
                gap: 0.6rem;
                margin-bottom: 1.25rem;
                padding: 0 0.5rem;
            }
            .brand-badge {
                font-size: 0.65rem;
                background: var(--primary);
                color: white;
                padding: 0.15rem 0.4rem;
                border-radius: 0.25rem;
                font-weight: 700;
                letter-spacing: 0.05em;
            }
            .notebook-list {
                flex: 1;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 0.35rem;
                margin-top: 0.5rem;
            }
            .notebook-item {
                padding: 0.7rem 0.85rem;
                border-radius: 0.5rem;
                cursor: pointer;
                font-weight: 500;
                font-size: 0.9rem;
                color: var(--text-muted);
                transition: all 0.15s;
                border: 1px solid transparent;
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.5rem;
            }
            .notebook-item:hover {
                background: var(--bg-card);
                color: var(--text-main);
            }
            .notebook-item.active {
                background: var(--bg-card);
                color: #e0e7ff;
                border-color: var(--primary);
                font-weight: 600;
            }
            .notebook-name {
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                flex: 1;
            }
            .btn-delete-nb {
                opacity: 0;
                background: transparent;
                border: none;
                color: var(--text-dim);
                cursor: pointer;
                padding: 0.2rem;
                border-radius: 0.25rem;
                font-size: 0.85rem;
                transition: all 0.15s;
            }
            .notebook-item:hover .btn-delete-nb {
                opacity: 1;
            }
            .btn-delete-nb:hover {
                color: var(--error);
                background: var(--error-bg);
            }
            
            /* Main Content Area */
            .main-content {
                flex: 1;
                display: flex;
                flex-direction: column;
                height: 100%;
                overflow: hidden;
            }
            .empty-state {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100%;
                color: var(--text-muted);
                text-align: center;
                padding: 2rem;
            }
            .empty-state h2 {
                font-size: 1.5rem;
                color: var(--text-main);
                margin-bottom: 0.5rem;
            }
            
            /* Workspace */
            .workspace {
                display: none;
                flex-direction: column;
                height: 100%;
                overflow: hidden;
            }
            .workspace.active { display: flex; }
            
            .workspace-header {
                padding: 1.25rem 2rem;
                border-bottom: 1px solid var(--border);
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: var(--bg-sidebar);
            }
            .workspace-title-area h2 {
                font-size: 1.35rem;
                font-weight: 700;
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            .status-pill {
                font-size: 0.75rem;
                font-weight: 600;
                padding: 0.2rem 0.6rem;
                border-radius: 1rem;
                background: var(--primary-subtle);
                color: #a5b4fc;
                border: 1px solid rgba(99, 102, 241, 0.3);
            }
            
            .workspace-tabs {
                display: flex;
                gap: 1.5rem;
                padding: 0 2rem;
                border-bottom: 1px solid var(--border);
                background: var(--bg-sidebar);
            }
            .tab-btn {
                padding: 0.85rem 0.25rem;
                background: transparent;
                border: none;
                color: var(--text-muted);
                font-weight: 600;
                font-size: 0.925rem;
                cursor: pointer;
                border-bottom: 2px solid transparent;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .tab-btn.active {
                color: #818cf8;
                border-bottom-color: var(--primary);
            }
            .tab-badge {
                font-size: 0.75rem;
                padding: 0.1rem 0.45rem;
                border-radius: 1rem;
                background: var(--bg-card);
                color: var(--text-muted);
            }
            
            .tab-content {
                flex: 1;
                overflow: hidden;
                display: none;
            }
            .tab-content.active { display: flex; }
            
            /* Sources Section */
            .sources-view {
                flex: 1;
                padding: 1.75rem 2rem;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }
            
            /* Add Sources Box */
            .add-source-card {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 0.85rem;
                padding: 1.25rem 1.5rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            }
            .add-source-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            }
            .add-source-tabs {
                display: flex;
                gap: 0.5rem;
                background: var(--bg-input);
                padding: 0.25rem;
                border-radius: 0.5rem;
                border: 1px solid var(--border);
            }
            .source-mode-btn {
                padding: 0.45rem 0.9rem;
                border-radius: 0.35rem;
                font-size: 0.825rem;
                font-weight: 600;
                background: transparent;
                border: none;
                color: var(--text-muted);
                cursor: pointer;
                transition: all 0.15s;
            }
            .source-mode-btn.active {
                background: var(--primary);
                color: white;
            }
            
            .source-input-row {
                display: flex;
                gap: 0.75rem;
            }
            input[type="text"], input[type="url"] {
                flex: 1;
                background: var(--bg-input);
                border: 1px solid var(--border);
                padding: 0.75rem 1rem;
                border-radius: 0.5rem;
                color: var(--text-main);
                font-family: inherit;
                font-size: 0.9rem;
                transition: border-color 0.2s;
            }
            input[type="text"]:focus, input[type="url"]:focus {
                outline: none;
                border-color: var(--primary);
            }
            
            .btn {
                background: var(--primary);
                color: white;
                border: none;
                padding: 0.75rem 1.25rem;
                border-radius: 0.5rem;
                font-weight: 600;
                font-size: 0.9rem;
                cursor: pointer;
                transition: all 0.2s;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 0.5rem;
                white-space: nowrap;
            }
            .btn:hover { background: var(--primary-hover); }
            .btn:disabled { opacity: 0.5; cursor: not-allowed; }
            .btn-secondary {
                background: var(--bg-input);
                border: 1px solid var(--border);
                color: var(--text-muted);
            }
            .btn-secondary:hover {
                background: var(--bg-card-hover);
                color: var(--text-main);
            }
            .btn-danger {
                background: var(--error-bg);
                color: var(--error);
                border: 1px solid rgba(239, 68, 68, 0.3);
            }
            .btn-danger:hover {
                background: var(--error);
                color: white;
            }
            
            /* Quick Select Presets / Chips */
            .quick-select-bar {
                display: flex;
                flex-wrap: wrap;
                gap: 0.4rem;
                align-items: center;
                margin-bottom: 0.75rem;
                padding: 0.5rem 0.75rem;
                background: rgba(16, 23, 40, 0.6);
                border: 1px solid var(--border);
                border-radius: 0.5rem;
            }
            .preset-label {
                font-size: 0.75rem;
                font-weight: 600;
                color: var(--text-dim);
                margin-right: 0.25rem;
            }
            .preset-chip {
                padding: 0.25rem 0.55rem;
                background: var(--bg-card);
                border: 1px solid #2a3754;
                border-radius: 0.35rem;
                color: #cbd5e1;
                font-size: 0.75rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.15s;
            }
            .preset-chip:hover {
                background: var(--primary);
                color: white;
                border-color: var(--primary);
            }
            .preset-chip-active {
                background: var(--primary);
                color: white;
                border-color: var(--primary);
            }
            
            /* Sitemap Discovery Result Panel */
            .discovery-box {
                margin-top: 1.25rem;
                border-top: 1px solid var(--border);
                padding-top: 1.25rem;
                display: none;
            }
            .discovery-stats {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.85rem;
            }
            .discovery-tools {
                display: flex;
                gap: 0.6rem;
                margin-bottom: 0.75rem;
            }
            .url-list-scroll {
                max-height: 280px;
                overflow-y: auto;
                background: var(--bg-input);
                border: 1px solid var(--border);
                border-radius: 0.5rem;
                padding: 0.35rem;
            }
            .url-item {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 0.5rem 0.65rem;
                border-radius: 0.35rem;
                font-size: 0.825rem;
                color: var(--text-muted);
                transition: all 0.15s;
                cursor: pointer;
                user-select: none;
            }
            .url-item:hover {
                background: var(--bg-card-hover);
                color: var(--text-main);
            }
            .url-item.selected {
                background: rgba(99, 102, 241, 0.08);
                color: #e0e7ff;
            }
            .url-item input[type="checkbox"] {
                accent-color: var(--primary);
                cursor: pointer;
                width: 16px;
                height: 16px;
            }
            .url-item label {
                flex: 1;
                cursor: pointer;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            
            /* Progress Bar */
            .progress-container {
                margin-top: 1rem;
                display: none;
                background: var(--bg-input);
                padding: 0.85rem 1rem;
                border-radius: 0.5rem;
                border: 1px solid var(--border);
            }
            .progress-bar-bg {
                width: 100%;
                height: 8px;
                background: #1e293b;
                border-radius: 4px;
                overflow: hidden;
                margin-top: 0.5rem;
            }
            .progress-bar-fill {
                height: 100%;
                width: 0%;
                background: linear-gradient(90deg, #6366f1, #10b981);
                transition: width 0.2s;
            }
            
            /* Source Items List */
            .sources-list {
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
            }
            .source-card {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 0.65rem;
                padding: 1rem 1.25rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: border-color 0.2s;
            }
            .source-card:hover {
                border-color: #3b4d74;
            }
            .source-title {
                font-weight: 600;
                font-size: 0.95rem;
                margin-bottom: 0.25rem;
                color: #e2e8f0;
            }
            .source-link {
                font-size: 0.8rem;
                color: #818cf8;
                text-decoration: none;
                display: inline-block;
                max-width: 500px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .source-link:hover { text-decoration: underline; }
            .source-meta {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            .status-badge {
                font-size: 0.7rem;
                font-weight: 700;
                padding: 0.25rem 0.55rem;
                border-radius: 0.35rem;
                letter-spacing: 0.05em;
            }
            .badge-success { background: var(--success-bg); color: var(--success); }
            .badge-failed { background: var(--error-bg); color: var(--error); }
            .badge-pending { background: rgba(245, 158, 11, 0.15); color: var(--warning); }
            
            .action-btn-group {
                display: flex;
                gap: 0.4rem;
            }
            .action-btn {
                background: var(--bg-input);
                border: 1px solid var(--border);
                color: var(--text-muted);
                padding: 0.35rem 0.65rem;
                border-radius: 0.35rem;
                font-size: 0.75rem;
                cursor: pointer;
                transition: all 0.15s;
                font-weight: 500;
            }
            .action-btn:hover {
                background: var(--bg-card-hover);
                color: var(--text-main);
                border-color: #475569;
            }
            .action-btn-del:hover {
                background: var(--error-bg);
                color: var(--error);
                border-color: var(--error);
            }
            
            /* Chat Section */
            .chat-view {
                flex: 1;
                display: flex;
                flex-direction: column;
                height: 100%;
                background: var(--bg-base);
            }
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 2rem;
                display: flex;
                flex-direction: column;
                gap: 1.25rem;
            }
            .chat-bubble {
                max-width: 80%;
                padding: 1rem 1.25rem;
                border-radius: 0.85rem;
                line-height: 1.6;
                font-size: 0.925rem;
                white-space: pre-wrap;
            }
            .bubble-user {
                background: var(--primary);
                color: white;
                align-self: flex-end;
                border-bottom-right-radius: 0.2rem;
            }
            .bubble-ai {
                background: var(--bg-card);
                border: 1px solid var(--border);
                color: #e2e8f0;
                align-self: flex-start;
                border-bottom-left-radius: 0.2rem;
            }
            .chat-input-container {
                padding: 1.25rem 2rem;
                border-top: 1px solid var(--border);
                background: var(--bg-sidebar);
            }
            .chat-form {
                display: flex;
                gap: 0.75rem;
            }
            
            /* Modals */
            .modal-backdrop {
                position: fixed;
                top: 0; left: 0; right: 0; bottom: 0;
                background: rgba(0,0,0,0.75);
                backdrop-filter: blur(4px);
                display: none;
                align-items: center;
                justify-content: center;
                z-index: 1000;
            }
            .modal-window {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 0.85rem;
                padding: 1.5rem;
                width: 500px;
                max-width: 90%;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }
            .modal-window h3 {
                margin-bottom: 0.75rem;
                font-size: 1.15rem;
            }
            .modal-footer {
                display: flex;
                justify-content: flex-end;
                gap: 0.5rem;
                margin-top: 1.25rem;
            }
            
            /* Toast Notifications */
            #toastBox {
                position: fixed;
                bottom: 2rem;
                right: 2rem;
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
                z-index: 2000;
            }
            .toast-item {
                padding: 0.75rem 1.25rem;
                border-radius: 0.5rem;
                font-size: 0.875rem;
                background: var(--bg-card);
                border: 1px solid var(--border);
                color: var(--text-main);
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                animation: slideIn 0.25s ease-out;
            }
            .toast-success { border-left: 4px solid var(--success); }
            .toast-error { border-left: 4px solid var(--error); }
            @keyframes slideIn { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: translateX(0); } }
            
            /* ========================================
               MOBILE RESPONSIVE (< 768px)
               ======================================== */
            @media (max-width: 768px) {
                .mobile-menu-btn { display: flex; }
                
                .sidebar {
                    position: fixed;
                    top: 0;
                    left: 0;
                    height: 100vh;
                    width: 280px;
                    transform: translateX(-100%);
                    box-shadow: 4px 0 25px rgba(0,0,0,0.5);
                    padding-top: 4rem;
                }
                .sidebar.open {
                    transform: translateX(0);
                }
                
                .main-content {
                    width: 100%;
                    min-width: 0;
                }
                
                .empty-state {
                    padding: 2rem 1.25rem;
                }
                .empty-state h2 {
                    font-size: 1.25rem;
                }
                .empty-state p {
                    font-size: 0.875rem;
                }
                
                /* Workspace Header */
                .workspace-header {
                    padding: 1rem;
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 0.75rem;
                }
                .workspace-header > div:last-child {
                    width: 100%;
                    justify-content: space-between;
                }
                .workspace-title-area h2 {
                    font-size: 1.1rem;
                    word-break: break-word;
                }
                .status-pill {
                    font-size: 0.65rem;
                    padding: 0.15rem 0.45rem;
                }
                
                /* Workspace Tabs */
                .workspace-tabs {
                    padding: 0 0.75rem;
                    gap: 0.5rem;
                    overflow-x: auto;
                    -webkit-overflow-scrolling: touch;
                }
                .tab-btn {
                    font-size: 0.825rem;
                    white-space: nowrap;
                    padding: 0.7rem 0.15rem;
                }
                
                /* Sources View */
                .sources-view {
                    padding: 1rem;
                    gap: 1rem;
                }
                
                /* Add Source Card */
                .add-source-card {
                    padding: 1rem;
                }
                .add-source-header {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 0.75rem;
                }
                .add-source-tabs {
                    width: 100%;
                }
                .source-mode-btn {
                    flex: 1;
                    text-align: center;
                    padding: 0.45rem 0.5rem;
                    font-size: 0.75rem;
                }
                
                /* Source Input Row */
                .source-input-row {
                    flex-direction: column;
                    gap: 0.5rem;
                }
                .source-input-row .btn {
                    width: 100%;
                }
                
                /* Discovery Section */
                .discovery-stats {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 0.5rem;
                }
                .discovery-stats .btn {
                    width: 100%;
                }
                .discovery-tools {
                    flex-direction: column;
                }
                .discovery-tools .btn-secondary {
                    width: 100%;
                }
                .quick-select-bar {
                    padding: 0.4rem 0.5rem;
                    gap: 0.3rem;
                }
                .preset-chip {
                    font-size: 0.7rem;
                    padding: 0.25rem 0.45rem;
                }
                .url-list-scroll {
                    max-height: 200px;
                }
                
                /* Source Cards */
                .source-card {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 0.75rem;
                    padding: 0.85rem 1rem;
                }
                .source-card > div:first-child {
                    margin-right: 0 !important;
                    width: 100%;
                }
                .source-link {
                    max-width: 100%;
                }
                .source-meta {
                    width: 100%;
                    justify-content: space-between;
                }
                
                /* Chat Section */
                .chat-messages {
                    padding: 1rem;
                    gap: 1rem;
                }
                .chat-bubble {
                    max-width: 92%;
                    padding: 0.85rem 1rem;
                    font-size: 0.875rem;
                }
                .chat-input-container {
                    padding: 0.85rem 1rem;
                }
                .chat-form input[type="text"] {
                    font-size: 0.875rem;
                    padding: 0.65rem 0.75rem;
                }
                
                /* Modal Adjustments */
                .modal-window {
                    margin: 0.75rem;
                    width: calc(100% - 1.5rem) !important;
                    max-width: 100% !important;
                    max-height: 90vh;
                    overflow-y: auto;
                }
                
                /* Toast Notifications */
                #toastBox {
                    bottom: 1rem;
                    right: 0.75rem;
                    left: 0.75rem;
                }
                .toast-item {
                    font-size: 0.8rem;
                    padding: 0.65rem 1rem;
                }

                /* Saved docs header */
                .sources-view > div[style*="justify-content: space-between"] {
                    flex-wrap: wrap;
                    gap: 0.5rem;
                }
            }
            
            /* ========================================
               SMALL MOBILE (< 400px)
               ======================================== */
            @media (max-width: 400px) {
                .sidebar {
                    width: 260px;
                }
                .workspace-header {
                    padding: 0.75rem;
                }
                .workspace-title-area h2 {
                    font-size: 1rem;
                }
                .source-mode-btn {
                    font-size: 0.68rem;
                }
                .preset-chip {
                    font-size: 0.65rem;
                    padding: 0.2rem 0.35rem;
                }
                .action-btn {
                    font-size: 0.7rem;
                    padding: 0.3rem 0.5rem;
                }
            }
            
            /* ========================================
               TABLET (769px - 1024px)
               ======================================== */
            @media (min-width: 769px) and (max-width: 1024px) {
                .sidebar {
                    width: 240px;
                }
                .sources-view {
                    padding: 1.25rem 1.5rem;
                }
                .workspace-header {
                    padding: 1rem 1.5rem;
                }
                .workspace-tabs {
                    padding: 0 1.5rem;
                }
                .source-link {
                    max-width: 300px;
                }
            }
        </style>
    </head>
    <body>
        <!-- Mobile Menu Button -->
        <button class="mobile-menu-btn" id="mobileMenuBtn" onclick="toggleMobileSidebar()" aria-label="Toggle Menu">☰</button>
        
        <!-- Sidebar Overlay (mobile) -->
        <div class="sidebar-overlay" id="sidebarOverlay" onclick="closeMobileSidebar()"></div>
        
        <!-- Left Sidebar -->
        <aside class="sidebar" id="sidebar">
            <div class="brand">
                <span>📚</span> MaxyCrawl LM
                <span class="brand-badge">PRO</span>
            </div>
            
            <button class="btn" onclick="openNewNotebookModal()">
                <span>+</span> Notebook Baru
            </button>
            
            <div class="notebook-list" id="notebookList">
                <!-- Dynamically rendered -->
            </div>
        </aside>
        
        <!-- Main Content Area -->
        <main class="main-content">
            <div class="empty-state" id="emptyState">
                <h2>Pilih atau Buat Notebook</h2>
                <p>Notebook adalah ruang kerja pengetahuan untuk mengumpulkan dokumen web dan berdiskusi dengan AI.</p>
                <button class="btn" style="margin-top: 1.25rem;" onclick="openNewNotebookModal()">+ Buat Notebook Sekarang</button>
            </div>
            
            <div class="workspace" id="workspace">
                <!-- Header -->
                <div class="workspace-header">
                    <div class="workspace-title-area">
                        <h2 id="workspaceTitle">Judul Notebook</h2>
                    </div>
                    <div style="display:flex; align-items:center; gap: 0.75rem;">
                        <span class="status-pill">🟢 Knowledge Base Aktif</span>
                        <button class="btn btn-danger" style="padding: 0.45rem 0.75rem; font-size: 0.8rem;" onclick="deleteCurrentNotebook()">Hapus Notebook</button>
                    </div>
                </div>
                
                <!-- Tab Headers -->
                <div class="workspace-tabs">
                    <button class="tab-btn active" id="btnTabSources" onclick="switchTab('sources')">
                        📁 Sumber Pengetahuan <span class="tab-badge" id="sourceCountBadge">0</span>
                    </button>
                    <button class="tab-btn" id="btnTabChat" onclick="switchTab('chat')">
                        💬 Tanya Dokumen (AI)
                    </button>
                </div>
                
                <!-- Tab: Sumber Pengetahuan -->
                <div class="tab-content active" id="tab-sources">
                    <div class="sources-view">
                        
                        <!-- Panel Tambah Sumber -->
                        <div class="add-source-card">
                            <div class="add-source-header">
                                <h3 style="font-size: 1rem;">Tambah Sumber Pengetahuan Baru</h3>
                                <div class="add-source-tabs">
                                    <button class="source-mode-btn active" id="modeSitemapBtn" onclick="setSourceMode('sitemap')">🌐 Seluruh Website (Sitemap)</button>
                                    <button class="source-mode-btn" id="modeSingleBtn" onclick="setSourceMode('single')">📄 1 Halaman Saja</button>
                                </div>
                            </div>
                            
                            <!-- Input Row -->
                            <div class="source-input-row">
                                <input type="url" id="sourceInputUrl" placeholder="https://recruitcrm.io (Domain atau URL Website)" onkeydown="if(event.key==='Enter') executeAddSource()">
                                <button class="btn" id="btnAddSourceAction" onclick="executeAddSource()">
                                    🔍 Temukan Semua Halaman
                                </button>
                            </div>
                            
                            <!-- Discovery Result Area (untuk mode Sitemap) -->
                            <div class="discovery-box" id="discoveryBox">
                                <div class="discovery-stats">
                                    <div>
                                        <span style="font-weight: 700; font-size: 1rem; color: #e2e8f0;" id="discoveryCountText">0 Halaman Ditemukan</span>
                                        <span style="font-size: 0.85rem; color: var(--primary); font-weight: 600; margin-left: 0.5rem;" id="selectedCountBadge">(0 Dipilih)</span>
                                    </div>
                                    <button class="btn" id="btnImportSelected" onclick="importSelectedPages()" style="padding: 0.55rem 1.25rem; font-size: 0.875rem;">
                                        📥 Simpan Halaman Terpilih
                                    </button>
                                </div>
                                
                                <!-- Quick Selection Preset Chips -->
                                <div class="quick-select-bar">
                                    <span class="preset-label">Pilihan Cepat:</span>
                                    <button class="preset-chip" onclick="applyPresetSelection('all')">✅ Pilih Semua</button>
                                    <button class="preset-chip" onclick="applyPresetSelection('none')">❌ Kosongkan</button>
                                    <button class="preset-chip" onclick="applyPresetSelection(10)">10 Teratas</button>
                                    <button class="preset-chip" onclick="applyPresetSelection(25)">25 Teratas</button>
                                    <button class="preset-chip" onclick="applyPresetSelection(50)">50 Teratas</button>
                                    <button class="preset-chip" onclick="applyPresetSelection(100)">100 Teratas</button>
                                    <button class="preset-chip" onclick="applyKeywordPreset('/blogs/')">Hanya /blogs/</button>
                                </div>
                                
                                <div class="discovery-tools">
                                    <input type="text" id="urlFilterInput" placeholder="🔍 Ketik untuk menyaring URL (contoh: /blogs/, /case-studies/, /pricing/)..." oninput="filterDiscoveredUrls()" style="padding: 0.5rem 0.85rem; font-size: 0.85rem;">
                                    <button class="btn btn-secondary" style="padding: 0.5rem 0.85rem; font-size: 0.825rem;" onclick="selectOnlyFilteredUrls()" title="Pilih hanya URL yang saat ini muncul di hasil pencarian">
                                        Pilih Hasil Filter
                                    </button>
                                </div>
                                
                                <div class="url-list-scroll" id="discoveryUrlList">
                                    <!-- URL Checkboxes -->
                                </div>
                                
                                <div class="progress-container" id="progressContainer">
                                    <div style="display:flex; justify-content:space-between; font-size: 0.85rem; font-weight: 600; color: var(--text-main);">
                                        <span id="progressStatusText">Menyimpan halaman...</span>
                                        <span id="progressPercentText" style="color: var(--primary);">0%</span>
                                    </div>
                                    <div class="progress-bar-bg">
                                        <div class="progress-bar-fill" id="progressBarFill"></div>
                                    </div>
                                </div>
                            </div>
                            
                        </div>
                        
                        <!-- List Sumber Tersimpan -->
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
                            <h3 style="font-size: 1.05rem; font-weight: 600;">Daftar Dokumen Tersimpan</h3>
                            <button class="btn btn-secondary" style="padding: 0.35rem 0.65rem; font-size: 0.8rem;" onclick="loadSources()">🔄 Refresh Daftar</button>
                        </div>
                        
                        <div class="sources-list" id="sourcesList">
                            <!-- Dynamically loaded sources -->
                        </div>
                        
                    </div>
                </div>
                
                <!-- Tab: Tanya Dokumen (AI Chat) -->
                <div class="tab-content" id="tab-chat">
                    <div class="chat-view">
                        <div class="chat-messages" id="chatMessages">
                            <div class="chat-bubble bubble-ai">
                                Halo! Saya adalah Asisten AI untuk Notebook ini. Saya telah membaca seluruh dokumen yang tersimpan dan siap menjawab pertanyaan Anda secara akurat berdasarkan data tersebut.
                            </div>
                        </div>
                        <div class="chat-input-container">
                            <form class="chat-form" id="chatForm">
                                <input type="text" id="chatInput" placeholder="Tanyakan apa saja tentang dokumen dalam notebook ini..." required autocomplete="off">
                                <button type="submit" class="btn" id="chatSendBtn">Kirim</button>
                            </form>
                        </div>
                    </div>
                </div>
                
            </div>
        </main>
        
        <!-- Modal: Buat Notebook Baru -->
        <div class="modal-backdrop" id="modalNewNb">
            <div class="modal-window">
                <h3>Buat Notebook Baru</h3>
                <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">Beri nama untuk topik atau proyek pengetahuan Anda.</p>
                <input type="text" id="newNbNameInput" placeholder="Contoh: Dokumen Recruit CRM / Riset Pasar" style="width: 100%;" onkeydown="if(event.key==='Enter') submitCreateNotebook()">
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="closeModal('modalNewNb')">Batal</button>
                    <button class="btn" onclick="submitCreateNotebook()">Buat Notebook</button>
                </div>
            </div>
        </div>
        
        <!-- Modal: Preview Konten Dokumen -->
        <div class="modal-backdrop" id="modalPreview">
            <div class="modal-window" style="width: 750px;">
                <h3 id="previewTitle">Judul Dokumen</h3>
                <p style="font-size: 0.8rem; color: var(--primary); margin-bottom: 0.75rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" id="previewUrl"></p>
                <div id="previewContent" style="max-height: 400px; overflow-y: auto; background: var(--bg-input); border: 1px solid var(--border); padding: 1rem; border-radius: 0.5rem; font-size: 0.85rem; line-height: 1.6; white-space: pre-wrap; color: var(--text-main);">
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="closeModal('modalPreview')">Tutup</button>
                </div>
            </div>
        </div>
        
        <div id="toastBox"></div>

        <script>
            let currentNbId = null;
            let currentMode = 'sitemap'; // 'sitemap' atau 'single'
            let rawDiscoveredUrls = [];
            let currentFilteredUrls = [];
            let selectedUrlSet = new Set();
            
            function showToast(msg, type='success') {
                const box = document.getElementById('toastBox');
                const t = document.createElement('div');
                t.className = `toast-item toast-${type}`;
                t.textContent = msg;
                box.appendChild(t);
                setTimeout(() => t.remove(), 3500);
            }
            
            function openNewNotebookModal() {
                document.getElementById('modalNewNb').style.display = 'flex';
                setTimeout(() => document.getElementById('newNbNameInput').focus(), 100);
            }
            function closeModal(id) {
                document.getElementById(id).style.display = 'none';
            }
            
            function setSourceMode(mode) {
                currentMode = mode;
                const sitemapBtn = document.getElementById('modeSitemapBtn');
                const singleBtn = document.getElementById('modeSingleBtn');
                const input = document.getElementById('sourceInputUrl');
                const actionBtn = document.getElementById('btnAddSourceAction');
                const discoveryBox = document.getElementById('discoveryBox');
                
                if (mode === 'sitemap') {
                    sitemapBtn.classList.add('active');
                    singleBtn.classList.remove('active');
                    input.placeholder = "https://recruitcrm.io (Domain atau URL Website)";
                    actionBtn.innerHTML = "🔍 Temukan Semua Halaman";
                } else {
                    singleBtn.classList.add('active');
                    sitemapBtn.classList.remove('active');
                    input.placeholder = "https://recruitcrm.io/blogs/post-1/ (Tautan 1 Halaman Spesifik)";
                    actionBtn.innerHTML = "📥 Tambah 1 Halaman";
                    discoveryBox.style.display = 'none';
                }
            }
            
            async function loadNotebooks() {
                try {
                    const res = await fetch('/api/notebooks');
                    const nbs = await res.json();
                    const list = document.getElementById('notebookList');
                    list.innerHTML = '';
                    if(!nbs || nbs.length === 0) {
                        list.innerHTML = '<div style="color:var(--text-dim);font-size:0.85rem;padding:0.75rem 0.5rem;text-align:center;">Belum ada notebook</div>';
                        return;
                    }
                    nbs.forEach(nb => {
                        const item = document.createElement('div');
                        item.className = `notebook-item ${currentNbId === nb.id ? 'active' : ''}`;
                        item.innerHTML = `
                            <span class="notebook-name">📓 ${nb.name}</span>
                            <button class="btn-delete-nb" title="Hapus Notebook" onclick="event.stopPropagation(); deleteNotebook(${nb.id}, '${nb.name}')">🗑️</button>
                        `;
                        item.onclick = () => selectNotebook(nb.id, nb.name);
                        list.appendChild(item);
                    });
                } catch(e) {
                    console.error("Error loading notebooks", e);
                }
            }
            
            async function submitCreateNotebook() {
                const input = document.getElementById('newNbNameInput');
                const name = input.value.trim();
                if(!name) return;
                try {
                    const res = await fetch('/api/notebooks', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({name})
                    });
                    if(res.ok) {
                        const nb = await res.json();
                        closeModal('modalNewNb');
                        input.value = '';
                        await loadNotebooks();
                        selectNotebook(nb.id, nb.name);
                        showToast('Notebook baru berhasil dibuat!');
                    } else {
                        showToast('Gagal membuat notebook', 'error');
                    }
                } catch(e) {
                    showToast('Kesalahan koneksi', 'error');
                }
            }
            
            async function deleteNotebook(id, name) {
                if(!confirm(`Apakah Anda yakin ingin menghapus notebook "${name}" beserta seluruh datanya?`)) return;
                try {
                    const res = await fetch(`/api/notebooks/${id}`, { method: 'DELETE' });
                    if(res.ok) {
                        showToast('Notebook berhasil dihapus');
                        if(currentNbId === id) {
                            currentNbId = null;
                            document.getElementById('workspace').classList.remove('active');
                            document.getElementById('emptyState').style.display = 'flex';
                        }
                        loadNotebooks();
                    }
                } catch(e) {
                    showToast('Gagal menghapus notebook', 'error');
                }
            }
            
            function deleteCurrentNotebook() {
                if(!currentNbId) return;
                const title = document.getElementById('workspaceTitle').textContent;
                deleteNotebook(currentNbId, title);
            }
            
            async function selectNotebook(id, name) {
                currentNbId = id;
                document.getElementById('emptyState').style.display = 'none';
                document.getElementById('workspace').classList.add('active');
                document.getElementById('workspaceTitle').textContent = name;
                document.getElementById('discoveryBox').style.display = 'none';
                loadNotebooks();
                loadSources();
                switchTab('sources');
            }
            
            function switchTab(tab) {
                document.getElementById('btnTabSources').classList.remove('active');
                document.getElementById('btnTabChat').classList.remove('active');
                document.getElementById('tab-sources').classList.remove('active');
                document.getElementById('tab-chat').classList.remove('active');
                
                if(tab === 'sources') {
                    document.getElementById('btnTabSources').classList.add('active');
                    document.getElementById('tab-sources').classList.add('active');
                } else {
                    document.getElementById('btnTabChat').classList.add('active');
                    document.getElementById('tab-chat').classList.add('active');
                }
            }
            
            async function loadSources() {
                if(!currentNbId) return;
                try {
                    const res = await fetch(`/api/notebooks/${currentNbId}/sources`);
                    const sources = await res.json();
                    const list = document.getElementById('sourcesList');
                    document.getElementById('sourceCountBadge').textContent = sources.length;
                    list.innerHTML = '';
                    
                    if(sources.length === 0) {
                        list.innerHTML = `
                            <div style="background:var(--bg-card); border:1px dashed var(--border); padding:2rem; border-radius:0.65rem; text-align:center; color:var(--text-muted);">
                                Belum ada sumber pengetahuan di notebook ini.<br>Gunakan form di atas untuk menambahkan dokumen website.
                            </div>
                        `;
                        return;
                    }
                    
                    sources.forEach(s => {
                        const isOk = s.status === 'success';
                        const badgeClass = isOk ? 'badge-success' : 'badge-failed';
                        const badgeText = isOk ? 'TERSIMPAN' : 'GAGAL';
                        
                        const item = document.createElement('div');
                        item.className = 'source-card';
                        item.innerHTML = `
                            <div style="flex:1; margin-right:1rem; overflow:hidden;">
                                <div class="source-title">${s.title || 'Dokumen Tanpa Judul'}</div>
                                <a href="${s.url}" target="_blank" class="source-link">${s.url}</a>
                            </div>
                            <div class="source-meta">
                                <span class="status-badge ${badgeClass}">${badgeText}</span>
                                <div class="action-btn-group">
                                    <button class="action-btn" onclick="previewSource(${s.id})" title="Lihat Isi Dokumen">👁️ Preview</button>
                                    <button class="action-btn" onclick="reSyncSource('${s.url}')" title="Perbarui Data">🔄 Update</button>
                                    <button class="action-btn action-btn-del" onclick="deleteSource(${s.id})" title="Hapus Dokumen">🗑️</button>
                                </div>
                            </div>
                        `;
                        list.appendChild(item);
                    });
                } catch(e) {
                    console.error("Error loading sources", e);
                }
            }
            
            async function previewSource(sourceId) {
                try {
                    const res = await fetch(`/api/notebooks/${currentNbId}/sources/${sourceId}`);
                    if(!res.ok) throw new Error("Gagal mengambil detail dokumen");
                    const data = await res.json();
                    document.getElementById('previewTitle').textContent = data.title || 'Dokumen Tanpa Judul';
                    document.getElementById('previewUrl').textContent = data.url;
                    document.getElementById('previewContent').textContent = data.content || '(Dokumen ini kosong atau gagal diekstrak)';
                    document.getElementById('modalPreview').style.display = 'flex';
                } catch(e) {
                    showToast(e.message, 'error');
                }
            }
            
            async function deleteSource(sourceId) {
                if(!confirm("Hapus dokumen ini dari notebook?")) return;
                try {
                    const res = await fetch(`/api/notebooks/${currentNbId}/sources/${sourceId}`, { method: 'DELETE' });
                    if(res.ok) {
                        showToast('Dokumen berhasil dihapus');
                        loadSources();
                    }
                } catch(e) {
                    showToast('Gagal menghapus dokumen', 'error');
                }
            }
            
            async function reSyncSource(url) {
                showToast(`Memperbarui data dari ${url}...`);
                try {
                    const res = await fetch(`/api/notebooks/${currentNbId}/scrape?url=${encodeURIComponent(url)}`, { method: 'POST' });
                    if(res.ok) {
                        showToast('Dokumen berhasil diperbarui!');
                        loadSources();
                    } else {
                        showToast('Gagal memperbarui dokumen', 'error');
                    }
                } catch(e) {
                    showToast('Kesalahan jaringan', 'error');
                }
            }
            
            async function executeAddSource() {
                const url = document.getElementById('sourceInputUrl').value.trim();
                if(!url) {
                    showToast('Harap masukkan URL terlebih dahulu', 'error');
                    return;
                }
                
                if (currentMode === 'single') {
                    // Single page scrape
                    const btn = document.getElementById('btnAddSourceAction');
                    btn.disabled = true;
                    btn.textContent = "Mengimpor...";
                    showToast(`Mengambil halaman: ${url}...`);
                    try {
                        const res = await fetch(`/api/notebooks/${currentNbId}/scrape?url=${encodeURIComponent(url)}`, { method: 'POST' });
                        if(res.ok) {
                            showToast('Halaman berhasil ditambahkan ke notebook!');
                            document.getElementById('sourceInputUrl').value = '';
                            loadSources();
                        } else {
                            showToast('Gagal mengambil halaman', 'error');
                        }
                    } catch(e) {
                        showToast('Kesalahan jaringan', 'error');
                    }
                    btn.disabled = false;
                    btn.textContent = "📥 Tambah 1 Halaman";
                } else {
                    // Sitemap Discovery
                    discoverSitemapFlow(url);
                }
            }
            
            async function discoverSitemapFlow(url) {
                const btn = document.getElementById('btnAddSourceAction');
                btn.disabled = true;
                btn.textContent = "Menjelajahi...";
                showToast(`Menelusuri sitemap dari ${url}...`);
                
                try {
                    const res = await fetch(`/api/sitemap?url=${encodeURIComponent(url)}`);
                    const data = await res.json();
                    rawDiscoveredUrls = data.urls || [];
                    currentFilteredUrls = [...rawDiscoveredUrls];
                    
                    // Default select ALL discovered URLs so user can scrape all at once
                    selectedUrlSet.clear();
                    rawDiscoveredUrls.forEach(u => selectedUrlSet.add(u));
                    
                    document.getElementById('discoveryBox').style.display = 'block';
                    document.getElementById('discoveryCountText').textContent = `${rawDiscoveredUrls.length.toLocaleString()} Halaman Ditemukan`;
                    document.getElementById('urlFilterInput').value = '';
                    
                    renderDiscoveredUrls(rawDiscoveredUrls);
                    showToast(`Berhasil menemukan ${rawDiscoveredUrls.length.toLocaleString()} halaman! (Semua otomatis terpilih)`);
                } catch(e) {
                    showToast('Gagal membaca sitemap website', 'error');
                }
                btn.disabled = false;
                btn.textContent = "🔍 Temukan Semua Halaman";
            }
            
            function renderDiscoveredUrls(urls) {
                const list = document.getElementById('discoveryUrlList');
                list.innerHTML = '';
                if(urls.length === 0) {
                    list.innerHTML = '<div style="font-size:0.85rem; color:var(--text-dim); padding:1rem; text-align:center;">Tidak ada URL yang cocok dengan filter</div>';
                    updateSelectionUI();
                    return;
                }
                
                const fragment = document.createDocumentFragment();
                urls.forEach((u, idx) => {
                    const isChecked = selectedUrlSet.has(u);
                    const div = document.createElement('div');
                    div.className = `url-item ${isChecked ? 'selected' : ''}`;
                    div.id = `url_row_${idx}`;
                    div.innerHTML = `
                        <input type="checkbox" id="url_cb_${idx}" value="${u}" ${isChecked ? 'checked' : ''}>
                        <label for="url_cb_${idx}" title="${u}">${u}</label>
                    `;
                    
                    // Click on row to toggle selection
                    div.onclick = (e) => {
                        if (e.target.tagName !== 'INPUT') {
                            const cb = div.querySelector('input[type="checkbox"]');
                            cb.checked = !cb.checked;
                        }
                        const isNowChecked = div.querySelector('input[type="checkbox"]').checked;
                        if (isNowChecked) {
                            selectedUrlSet.add(u);
                            div.classList.add('selected');
                        } else {
                            selectedUrlSet.delete(u);
                            div.classList.remove('selected');
                        }
                        updateSelectionUI();
                    };
                    
                    fragment.appendChild(div);
                });
                list.appendChild(fragment);
                
                updateSelectionUI();
            }
            
            function filterDiscoveredUrls() {
                const filter = document.getElementById('urlFilterInput').value.toLowerCase().trim();
                currentFilteredUrls = rawDiscoveredUrls.filter(u => u.toLowerCase().includes(filter));
                renderDiscoveredUrls(currentFilteredUrls);
            }
            
            function applyPresetSelection(type) {
                if (type === 'all') {
                    rawDiscoveredUrls.forEach(u => selectedUrlSet.add(u));
                    showToast(`Semua ${rawDiscoveredUrls.length.toLocaleString()} halaman dipilih`);
                } else if (type === 'none') {
                    selectedUrlSet.clear();
                    showToast('Pilihan dikosongkan');
                } else if (typeof type === 'number') {
                    selectedUrlSet.clear();
                    const count = Math.min(type, rawDiscoveredUrls.length);
                    for (let i = 0; i < count; i++) {
                        selectedUrlSet.add(rawDiscoveredUrls[i]);
                    }
                    showToast(`${count.toLocaleString()} halaman teratas dipilih`);
                }
                renderDiscoveredUrls(currentFilteredUrls);
            }
            
            function applyKeywordPreset(kw) {
                document.getElementById('urlFilterInput').value = kw;
                filterDiscoveredUrls();
                selectOnlyFilteredUrls();
            }
            
            function selectOnlyFilteredUrls() {
                if (currentFilteredUrls.length === 0) return;
                selectedUrlSet.clear();
                currentFilteredUrls.forEach(u => selectedUrlSet.add(u));
                renderDiscoveredUrls(currentFilteredUrls);
                showToast(`${currentFilteredUrls.length.toLocaleString()} halaman hasil filter dipilih`);
            }
            
            function updateSelectionUI() {
                const totalSelected = selectedUrlSet.size;
                const badge = document.getElementById('selectedCountBadge');
                const btn = document.getElementById('btnImportSelected');
                
                badge.textContent = `(${totalSelected.toLocaleString()} Dipilih)`;
                btn.textContent = `📥 Simpan Semua ${totalSelected.toLocaleString()} Halaman Terpilih`;
                btn.disabled = totalSelected === 0;
            }
            
            async function importSelectedPages() {
                const selectedUrls = Array.from(selectedUrlSet);
                if(selectedUrls.length === 0) return;
                
                const total = selectedUrls.length;
                const progContainer = document.getElementById('progressContainer');
                const progStatus = document.getElementById('progressStatusText');
                const progPercent = document.getElementById('progressPercentText');
                const progFill = document.getElementById('progressBarFill');
                const btn = document.getElementById('btnImportSelected');
                
                progContainer.style.display = 'block';
                btn.disabled = true;
                
                let completed = 0;
                // Process in concurrent batches of 6
                const batchSize = 6;
                for (let i = 0; i < selectedUrls.length; i += batchSize) {
                    const batch = selectedUrls.slice(i, i + batchSize);
                    await Promise.all(batch.map(async url => {
                        try {
                            await fetch(`/api/notebooks/${currentNbId}/scrape?url=${encodeURIComponent(url)}`, { method: 'POST' });
                        } catch(e) {}
                        completed++;
                        const pct = Math.round((completed / total) * 100);
                        progFill.style.width = `${pct}%`;
                        progPercent.textContent = `${pct}%`;
                        progStatus.textContent = `Menyimpan ${completed.toLocaleString()} dari ${total.toLocaleString()} halaman (${pct}%)...`;
                    }));
                }
                
                showToast(`Selesai menyimpan ${completed.toLocaleString()} dokumen ke notebook!`);
                btn.disabled = false;
                loadSources();
            }
            
            // AI Chat Submission
            document.getElementById('chatForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                if(!currentNbId) return;
                const input = document.getElementById('chatInput');
                const query = input.value.trim();
                if(!query) return;
                
                const chatBox = document.getElementById('chatMessages');
                chatBox.innerHTML += `<div class="chat-bubble bubble-user">${query}</div>`;
                input.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;
                
                const sendBtn = document.getElementById('chatSendBtn');
                sendBtn.disabled = true;
                sendBtn.textContent = "...";
                
                // Loading indicator
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'chat-bubble bubble-ai';
                loadingDiv.textContent = 'Membaca dokumen & menganalisis...';
                chatBox.appendChild(loadingDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
                
                try {
                    const res = await fetch(`/api/notebooks/${currentNbId}/chat`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query})
                    });
                    const data = await res.json();
                    loadingDiv.textContent = data.answer || data.detail || 'Maaf, terjadi kendala saat memproses respons.';
                } catch(err) {
                    loadingDiv.textContent = 'Gagal terhubung ke AI Assistant.';
                    loadingDiv.style.color = 'var(--error)';
                }
                
                sendBtn.disabled = false;
                sendBtn.textContent = "Kirim";
                chatBox.scrollTop = chatBox.scrollHeight;
            });
            
            // Mobile sidebar toggle
            function toggleMobileSidebar() {
                const sidebar = document.getElementById('sidebar');
                const overlay = document.getElementById('sidebarOverlay');
                const btn = document.getElementById('mobileMenuBtn');
                const isOpen = sidebar.classList.contains('open');
                if (isOpen) {
                    closeMobileSidebar();
                } else {
                    sidebar.classList.add('open');
                    overlay.classList.add('active');
                    btn.textContent = '✕';
                }
            }
            function closeMobileSidebar() {
                const sidebar = document.getElementById('sidebar');
                const overlay = document.getElementById('sidebarOverlay');
                const btn = document.getElementById('mobileMenuBtn');
                sidebar.classList.remove('open');
                overlay.classList.remove('active');
                btn.textContent = '☰';
            }
            
            // Auto-close sidebar on notebook select (mobile)
            const origSelectNotebook = selectNotebook;
            selectNotebook = async function(id, name) {
                closeMobileSidebar();
                return origSelectNotebook(id, name);
            };
            
            loadNotebooks();
        </script>
    </body>
    </html>
    """
