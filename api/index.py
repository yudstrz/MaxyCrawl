from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import trafilatura
from lxml import etree
from datetime import datetime, timezone
from urllib.parse import urljoin
import asyncio
import os
from dotenv import load_dotenv
import numpy as np
import tiktoken
from openai import AsyncOpenAI

load_dotenv()

app = FastAPI(
    title="MaxyCrawl API",
    description="On-demand web scraper API for knowledge bases.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USER_AGENT = "MaxyCrawl-API/1.0 (+https://github.com/yudstrz/MaxyCrawl)"
TIMEOUT_SECONDS = 15

# Detect Vercel environment to adjust limits dynamically
IS_VERCEL = os.environ.get("VERCEL") == "1"
MAX_SITEMAPS_TO_FETCH = 5 if IS_VERCEL else 20  # Limit to 5 on Vercel to avoid timeouts, 20 locally

# --- AI & RAG setup ---
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None

class ChatRequest(BaseModel):
    query: str
    context_items: list[dict] = []

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

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_data(req: ChatRequest):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    if not req.context_items:
        return ChatResponse(answer="No data has been scraped yet. Please extract data first.")
        
    # 1. Prepare and Chunk Context
    chunks = []
    for item in req.context_items:
        text = f"Title: {item.get('title', '')}\nURL: {item.get('url', '')}\nContent: {item.get('content', '')}"
        item_chunks = chunk_text(text, max_tokens=500)
        chunks.extend(item_chunks)
        
    if not chunks:
        return ChatResponse(answer="The scraped data is empty.")
        
    # 2. Embed query
    query_response = await openai_client.embeddings.create(
        input=req.query,
        model="text-embedding-3-small"
    )
    query_vector = np.array(query_response.data[0].embedding, dtype=np.float32)
    
    # 3. Limit chunks to avoid extreme timeouts/costs on Vercel
    chunks = chunks[:50] 
    
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
    
    # 3. Generate Answer
    context = "\n\n---\n\n".join(top_chunks)
    prompt = f"You are a helpful assistant answering questions based on the provided context.\n\nContext:\n{context}\n\nQuestion: {req.query}\n\nAnswer:"
    
    completion = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI that answers questions accurately based ONLY on the provided scraped website context. Jika pengguna hanya menyapa (seperti 'hai' atau 'halo'), sapa kembali dengan ramah dan tanyakan apa yang ingin mereka ketahui dari data ini. Jika jawaban tidak ada di konteks, sampaikan dengan sopan bahwa informasi tersebut tidak tersedia di Knowledge Base saat ini. PENTING: Selalu gunakan Bahasa Indonesia dalam menjawab, kecuali pengguna bertanya dalam bahasa lain."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return ChatResponse(answer=completion.choices[0].message.content)

# --- End AI & RAG setup ---

class ScrapeResponse(BaseModel):
    url: str
    title: str
    language: str
    content: str
    scraped_at: str
    status: str
    error: str = ""

class SitemapResponse(BaseModel):
    base_url: str
    total_urls: int
    urls: list[str]
    error: str = ""

@app.get("/api/sitemap", response_model=SitemapResponse)
async def discover_sitemap(url: str = Query(..., description="The base URL of the website")):
    base_url = url.rstrip("/")
    sitemap_urls = []
    
    # 1. Check robots.txt
    robots_url = f"{base_url}/robots.txt"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            resp = await client.get(robots_url, headers={"User-Agent": USER_AGENT})
            if resp.status_code == 200:
                for line in resp.text.splitlines():
                    if line.strip().lower().startswith("sitemap:"):
                        sitemap_url = line.split(":", 1)[1].strip()
                        sitemap_urls.append(sitemap_url)
    except Exception:
        pass

    if not sitemap_urls:
        sitemap_urls.append(f"{base_url}/sitemap.xml")

    visited_sitemaps = set()
    all_page_urls = set()
    queue = sitemap_urls.copy()
    
    # BFS to fetch sitemaps
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
        while queue and len(visited_sitemaps) < MAX_SITEMAPS_TO_FETCH:
            current_sitemap = queue.pop(0)
            if current_sitemap in visited_sitemaps:
                continue
                
            visited_sitemaps.add(current_sitemap)
            
            try:
                resp = await client.get(current_sitemap, headers={"User-Agent": USER_AGENT})
                if resp.status_code != 200:
                    continue
                    
                try:
                    root = etree.fromstring(resp.content)
                    tag = root.tag
                    ns = "http://www.sitemaps.org/schemas/sitemap/0.9"
                    
                    if "sitemapindex" in tag:
                        for sitemap_elem in root.iter(f"{{{ns}}}sitemap"):
                            loc = sitemap_elem.find(f"{{{ns}}}loc")
                            if loc is not None and loc.text:
                                queue.append(loc.text.strip())
                    else:
                        for url_elem in root.iter(f"{{{ns}}}url"):
                            loc = url_elem.find(f"{{{ns}}}loc")
                            if loc is not None and loc.text:
                                all_page_urls.add(loc.text.strip())
                except etree.XMLSyntaxError:
                    pass
            except Exception:
                pass

    return SitemapResponse(
        base_url=base_url,
        total_urls=len(all_page_urls),
        urls=list(all_page_urls)
    )

@app.get("/api/scrape", response_model=ScrapeResponse)
async def scrape_url(url: str = Query(..., description="The full URL to scrape")):
    scraped_at = datetime.now(timezone.utc).isoformat()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            html = response.text
    except httpx.HTTPStatusError as e:
        return ScrapeResponse(url=url, title="", language="", content="", scraped_at=scraped_at, status="failed", error=f"HTTP Error: {e.response.status_code}")
    except Exception as e:
        return ScrapeResponse(url=url, title="", language="", content="", scraped_at=scraped_at, status="failed", error=f"Failed to fetch URL: {str(e)}")

    title, language, content, status, error = "", "", "", "pending", ""
    metadata = trafilatura.extract_metadata(html, default_url=url)
    if metadata:
        title, language = metadata.title or "", metadata.language or ""

    extracted = trafilatura.extract(
        html, url=url, include_tables=True, include_links=False, 
        include_images=False, output_format="markdown", deduplicate=True
    )
    if extracted:
        content, status = extracted, "success"
    else:
        try:
            tree = etree.HTML(html.encode("utf-8"))
            if tree is not None:
                title_nodes = tree.xpath("//title/text()")
                if title_nodes and not title:
                    title = title_nodes[0].strip()
        except Exception:
            pass
        status, error = "failed", "trafilatura returned no content"

    return ScrapeResponse(url=url, title=title, language=language, content=content, scraped_at=scraped_at, status=status, error=error)

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MaxyCrawl - Extract & Chat</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #4F46E5; --primary-hover: #4338CA;
                --bg: #0F172A; --surface: #1E293B; --surface-hover: #334155;
                --text: #F8FAFC; --text-muted: #94A3B8; --border: #334155;
                --success: #10B981; --error: #EF4444;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: 'Inter', sans-serif; background-color: var(--bg); color: var(--text); padding: 3rem 1rem; }
            .container { max-width: 900px; margin: 0 auto; }
            header { text-align: center; margin-bottom: 2rem; }
            h1 { font-size: 2.5rem; font-weight: 700; background: linear-gradient(to right, #818CF8, #C084FC); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
            
            .card { background: var(--surface); border: 1px solid var(--border); border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
            
            /* Tabs */
            .tabs { display: flex; gap: 1rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
            .tab-btn { background: none; border: none; color: var(--text-muted); padding: 0.75rem 1.5rem; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent; }
            .tab-btn.active { color: var(--primary); border-bottom-color: var(--primary); }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            
            .input-group { display: flex; gap: 0.5rem; }
            input[type="url"], input[type="text"] { flex: 1; background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 0.75rem 1rem; border-radius: 0.5rem; outline: none; }
            input[type="url"]:focus, input[type="text"]:focus { border-color: var(--primary); }
            
            button { background: var(--primary); color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 0.5rem; font-weight: 600; cursor: pointer; transition: 0.2s; display: flex; align-items: center; justify-content: center; gap: 0.5rem; }
            button:hover:not(:disabled) { background: var(--primary-hover); }
            button:disabled { opacity: 0.5; cursor: not-allowed; }
            
            .hidden { display: none !important; }
            
            /* Sitemap List Styles */
            .list-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border); }
            .sitemap-list { max-height: 400px; overflow-y: auto; background: var(--bg); border: 1px solid var(--border); border-radius: 0.5rem; padding: 0.5rem; }
            
            .url-item { display: flex; align-items: center; padding: 0.5rem; border-bottom: 1px solid var(--border); font-size: 0.875rem; gap: 0.75rem; }
            .url-item:last-child { border-bottom: none; }
            .url-item input[type="checkbox"] { width: 1rem; height: 1rem; cursor: pointer; }
            .url-item label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; color: var(--text-muted); }
            
            .status-badge { padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.7rem; font-weight: 600; }
            .status-pending { background: #475569; color: white; }
            .status-success { background: rgba(16, 185, 129, 0.2); color: var(--success); }
            .status-failed { background: rgba(239, 68, 68, 0.2); color: var(--error); }
            
            .progress-container { margin-top: 1.5rem; }
            .progress-bar-bg { width: 100%; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; margin-top: 0.5rem; }
            .progress-bar-fill { height: 100%; background: var(--primary); width: 0%; transition: width 0.3s; }
            
            .json-result { background: #0f172a; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; font-family: monospace; font-size: 0.8rem; overflow-x: auto; max-height: 200px; }
            
            /* Chat UI */
            .chat-window { background: var(--bg); border: 1px solid var(--border); border-radius: 0.5rem; height: 500px; display: flex; flex-direction: column; }
            .chat-messages { flex: 1; overflow-y: auto; padding: 1rem; display: flex; flex-direction: column; gap: 1rem; }
            .chat-message { padding: 0.75rem 1rem; border-radius: 0.5rem; max-width: 85%; line-height: 1.5; font-size: 0.95rem; }
            .msg-user { background: var(--primary); color: white; align-self: flex-end; border-bottom-right-radius: 0; }
            .msg-ai { background: var(--surface-hover); color: var(--text); align-self: flex-start; border-bottom-left-radius: 0; white-space: pre-wrap; }
            .chat-input-area { display: flex; padding: 1rem; border-top: 1px solid var(--border); gap: 0.5rem; }
            
            /* Spinner */
            .spinner { width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-radius: 50%; border-top-color: white; animation: spin 1s linear infinite; }
            @keyframes spin { to { transform: rotate(360deg); } }
            
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: var(--bg); }
            ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
            
            /* Toast Notifications */
            #toastContainer { position: fixed; bottom: 20px; right: 20px; z-index: 50; display: flex; flex-direction: column; gap: 10px; }
            .toast { color: white; padding: 1rem 1.5rem; border-radius: 0.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); transform: translateY(100%); opacity: 0; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); min-width: 250px; font-size: 0.9rem; white-space: pre-wrap; }
            .toast.show { transform: translateY(0); opacity: 1; }
            .toast-success { background: var(--success); color: #0F172A; font-weight: 600; }
            .toast-error { background: var(--error); font-weight: 500; }
            .toast-info { background: var(--primary); font-weight: 500; }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>MaxyCrawl AI</h1>
                <p style="color: var(--text-muted)">Extract pages in bulk & chat with your knowledge base.</p>
            </header>

            <div class="card">
                <div class="tabs">
                    <button class="tab-btn active" onclick="switchTab('extract')">1. Data Extraction</button>
                    <button class="tab-btn" onclick="switchTab('knowledge')">2. Test Knowledge (Chat)</button>
                </div>
                
                <!-- EXTRACTION TAB -->
                <div id="tab-extract" class="tab-content active">
                    <form id="discoveryForm" class="input-group" style="margin-bottom: 1.5rem;">
                        <input type="url" id="baseUrl" placeholder="Enter root URL (e.g. https://recruitcrm.io)" required>
                        <button type="submit" id="discoverBtn">
                            <span id="discoverText">Discover</span>
                            <div class="spinner hidden" id="discoverSpinner"></div>
                        </button>
                    </form>

                    <div id="resultsCard" class="hidden">
                        <div class="list-header">
                            <div>
                                <h2 style="font-size: 1.25rem;">Discovered URLs</h2>
                                <p id="urlCount" style="color: var(--text-muted); font-size: 0.875rem; margin-top: 0.25rem;">0 items found</p>
                            </div>
                            <div style="display: flex; gap: 1rem;">
                                <label style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.875rem; cursor: pointer;">
                                    <input type="checkbox" id="selectAll"> Select All
                                </label>
                                <button id="startBtn">Start Scraping</button>
                            </div>
                        </div>

                        <div class="sitemap-list" id="urlList"></div>

                        <div id="progressArea" class="progress-container hidden">
                            <div style="display: flex; justify-content: space-between; font-size: 0.875rem;">
                                <span>Scraping Progress</span>
                                <span id="progressText">0 / 0</span>
                            </div>
                            <div class="progress-bar-bg">
                                <div class="progress-bar-fill" id="progressBar"></div>
                            </div>
                        </div>
                        
                        <div id="jsonArea" class="hidden">
                            <h3 style="margin-top: 1.5rem; font-size: 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem;">Results</h3>
                            <textarea id="jsonOutput" class="json-result" style="width: 100%; color: var(--success); resize: vertical;" readonly></textarea>
                            <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                                <button id="downloadBtn" style="background: var(--surface-hover);">Download JSONL</button>
                                <button id="goToChatBtn" style="flex: 1; background: var(--primary); color: white;">
                                    <span>Go to Chat</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- KNOWLEDGE TAB -->
                <div id="tab-knowledge" class="tab-content">
                    <div class="chat-window">
                        <div class="chat-messages" id="chatMessages">
                            <div class="chat-message msg-ai">Hi! I will analyze your scraped data dynamically. Please make sure you have successfully scraped some URLs, then ask me anything!</div>
                        </div>
                        <form id="chatForm" class="chat-input-area">
                            <input type="text" id="chatInput" placeholder="Ask a question..." required>
                            <button type="submit" id="chatBtn">
                                <span id="chatText">Send</span>
                                <div class="spinner hidden" id="chatSpinner"></div>
                            </button>
                        </form>
                    </div>
                </div>
                
            </div>
        </div>

        <div id="toastContainer"></div>

        <script>
            function showToast(message, type = 'info') {
                const container = document.getElementById('toastContainer');
                const toast = document.createElement('div');
                toast.className = `toast toast-${type}`;
                toast.textContent = message;
                
                container.appendChild(toast);
                
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        toast.classList.add('show');
                    });
                });
                
                setTimeout(() => {
                    toast.classList.remove('show');
                    setTimeout(() => toast.remove(), 300);
                }, 4000);
            }

            function switchTab(tab) {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                if (tab === 'extract') {
                    document.querySelector('.tab-btn:nth-child(1)').classList.add('active');
                    document.getElementById('tab-extract').classList.add('active');
                } else {
                    document.querySelector('.tab-btn:nth-child(2)').classList.add('active');
                    document.getElementById('tab-knowledge').classList.add('active');
                }
            }
        
            let discoveredUrls = [];
            let scrapedItems = [];
            
            const form = document.getElementById('discoveryForm');
            const baseUrlInput = document.getElementById('baseUrl');
            const discoverBtn = document.getElementById('discoverBtn');
            const discoverText = document.getElementById('discoverText');
            const discoverSpinner = document.getElementById('discoverSpinner');
            
            const resultsCard = document.getElementById('resultsCard');
            const urlList = document.getElementById('urlList');
            const urlCount = document.getElementById('urlCount');
            const selectAll = document.getElementById('selectAll');
            const startBtn = document.getElementById('startBtn');
            
            const progressArea = document.getElementById('progressArea');
            const progressBar = document.getElementById('progressBar');
            const progressText = document.getElementById('progressText');
            const jsonArea = document.getElementById('jsonArea');
            const jsonOutput = document.getElementById('jsonOutput');
            
            const goToChatBtn = document.getElementById('goToChatBtn');

            // 1. Discover URLs
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const url = baseUrlInput.value;
                discoverBtn.disabled = true;
                discoverText.classList.add('hidden');
                discoverSpinner.classList.remove('hidden');
                resultsCard.classList.add('hidden');
                jsonArea.classList.add('hidden');
                
                try {
                    const res = await fetch(`/api/sitemap?url=${encodeURIComponent(url)}`);
                    const data = await res.json();
                    discoveredUrls = data.urls || [];
                    urlCount.textContent = `${discoveredUrls.length} items found`;
                    renderList();
                    resultsCard.classList.remove('hidden');
                } catch (err) {
                    showToast("Failed to fetch sitemap.", "error");
                } finally {
                    discoverBtn.disabled = false;
                    discoverText.classList.remove('hidden');
                    discoverSpinner.classList.add('hidden');
                }
            });

            function renderList() {
                urlList.innerHTML = '';
                discoveredUrls.forEach((url, i) => {
                    const div = document.createElement('div');
                    div.className = 'url-item';
                    div.innerHTML = `
                        <input type="checkbox" id="chk_${i}" class="url-cb" value="${url}">
                        <label for="chk_${i}">${url}</label>
                        <span id="badge_${i}" class="status-badge status-pending">PENDING</span>
                    `;
                    urlList.appendChild(div);
                });
            }

            selectAll.addEventListener('change', (e) => {
                document.querySelectorAll('.url-cb').forEach(cb => {
                    cb.checked = e.target.checked;
                });
            });

            // 2. Start Scraping
            startBtn.addEventListener('click', async () => {
                const checkboxes = document.querySelectorAll('.url-cb:checked');
                if (checkboxes.length === 0) return showToast("Select at least 1 URL!", "error");
                
                startBtn.disabled = true;
                selectAll.disabled = true;
                document.querySelectorAll('.url-cb').forEach(cb => cb.disabled = true);
                
                progressArea.classList.remove('hidden');
                jsonArea.classList.add('hidden');
                scrapedItems = [];
                jsonOutput.value = "";
                
                const total = checkboxes.length;
                let done = 0;
                
                for (let cb of checkboxes) {
                    const url = cb.value;
                    const index = cb.id.split('_')[1];
                    const badge = document.getElementById(`badge_${index}`);
                    
                    badge.textContent = "SCRAPING...";
                    badge.style.background = "var(--primary)";
                    badge.style.color = "white";
                    
                    try {
                        const res = await fetch(`/api/scrape?url=${encodeURIComponent(url)}`);
                        const data = await res.json();
                        
                        if (data.status === 'success') {
                            badge.textContent = "SUCCESS";
                            badge.className = "status-badge status-success";
                            
                            const item = {url: data.url, title: data.title, content: data.content};
                            scrapedItems.push(item);
                            jsonOutput.value += JSON.stringify(item) + "\\n";
                        } else {
                            badge.textContent = "FAILED";
                            badge.className = "status-badge status-failed";
                        }
                    } catch (err) {
                        badge.textContent = "ERROR";
                        badge.className = "status-badge status-failed";
                    }
                    
                    done++;
                    progressBar.style.width = `${(done / total) * 100}%`;
                    progressText.textContent = `${done} / ${total}`;
                    cb.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
                
                startBtn.disabled = false;
                startBtn.textContent = "Scraping Complete!";
                jsonArea.classList.remove('hidden');
            });
            
            document.getElementById('downloadBtn').addEventListener('click', () => {
                const blob = new Blob([jsonOutput.value], { type: 'application/json' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'combined.jsonl';
                a.click();
                window.URL.revokeObjectURL(url);
            });
            
            // 3. Navigate to Chat
            goToChatBtn.addEventListener('click', () => {
                if (scrapedItems.length === 0) return showToast("No successfully scraped items! Please scrape first.", "error");
                showToast("Ready! Your data will be processed on-the-fly when you ask a question.", "success");
                switchTab('knowledge');
            });
            
            // 4. Chat Flow
            const chatForm = document.getElementById('chatForm');
            const chatInput = document.getElementById('chatInput');
            const chatMessages = document.getElementById('chatMessages');
            const chatBtn = document.getElementById('chatBtn');
            const chatText = document.getElementById('chatText');
            const chatSpinner = document.getElementById('chatSpinner');
            
            function appendMessage(role, text) {
                const div = document.createElement('div');
                div.className = `chat-message ${role === 'user' ? 'msg-user' : 'msg-ai'}`;
                div.textContent = text;
                chatMessages.appendChild(div);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            chatForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const query = chatInput.value.trim();
                if (!query) return;
                
                appendMessage('user', query);
                chatInput.value = '';
                
                chatBtn.disabled = true;
                chatText.classList.add('hidden');
                chatSpinner.classList.remove('hidden');
                
                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: query, context_items: scrapedItems})
                    });
                    const data = await res.json();
                    
                    if (res.ok) {
                        appendMessage('ai', data.answer);
                    } else {
                        appendMessage('ai', `Error: ${data.detail || 'Failed to get answer'}`);
                    }
                } catch (err) {
                    appendMessage('ai', "Error communicating with backend.");
                } finally {
                    chatBtn.disabled = false;
                    chatText.classList.remove('hidden');
                    chatSpinner.classList.add('hidden');
                    chatInput.focus();
                }
            });
        </script>
    </body>
    </html>
    """
