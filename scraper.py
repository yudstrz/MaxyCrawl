#!/usr/bin/env python3
"""
MaxyCrawl - AI Knowledge Base Web Scraper
==========================================
Scrapes all pages from a domain via sitemap discovery,
extracts clean content, and outputs Markdown + JSONL files
ready for use in AI/RAG knowledge bases.

Usage:
    python scraper.py --url https://example.com --lang-filter en \
        --path-filter /blogs/ --concurrency 8 --output ./output
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura
from lxml import etree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("maxycrawl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONCURRENCY = 6
DEFAULT_REQUEST_DELAY = 0.5        # seconds between requests per worker
DEFAULT_REQUEST_TIMEOUT = 30       # seconds
DEFAULT_OUTPUT_DIR = "./output"
PROGRESS_FILE = "progress.json"
COMBINED_JSONL = "combined.jsonl"
USER_AGENT = (
    "MaxyCrawl/1.0 (+https://github.com/yudstrz/MaxyCrawl; "
    "AI knowledge base builder)"
)

# HTTP status codes that are worth retrying
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ScrapeConfig:
    """All runtime configuration in one place."""
    base_url: str
    lang_filter: Optional[str]          # e.g. "en"
    path_filter: Optional[str]          # e.g. "/blogs/"
    concurrency: int
    output_dir: Path
    request_delay: float
    timeout: int


@dataclass
class PageResult:
    """Result of scraping a single page."""
    url: str
    title: str = ""
    language: str = ""
    content: str = ""
    scraped_at: str = ""
    status: str = "pending"             # pending | success | failed | skipped
    error: str = ""


# ---------------------------------------------------------------------------
# SECTION 1: Robots.txt Parsing
# ---------------------------------------------------------------------------

def load_robots_txt(base_url: str) -> RobotFileParser:
    """
    Fetch and parse robots.txt from the target domain.

    Returns a RobotFileParser that can check if a URL is allowed.
    If robots.txt cannot be fetched, returns a permissive parser
    (allows everything).
    """
    robots_url = urljoin(base_url, "/robots.txt")
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        log.info(f"Loaded robots.txt from {robots_url}")
    except Exception as exc:
        log.warning(f"Could not fetch robots.txt ({exc}), assuming no restrictions.")
    return rp


def get_sitemap_urls_from_robots(base_url: str) -> list[str]:
    """
    Extract Sitemap: directive URLs from robots.txt.

    Returns a list of sitemap URLs found in robots.txt.
    Falls back to the conventional /sitemap.xml if none declared.
    """
    robots_url = urljoin(base_url, "/robots.txt")
    sitemap_urls: list[str] = []
    try:
        with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=15) as client:
            resp = client.get(robots_url)
            resp.raise_for_status()
            for line in resp.text.splitlines():
                if line.strip().lower().startswith("sitemap:"):
                    url = line.split(":", 1)[1].strip()
                    sitemap_urls.append(url)
                    log.debug(f"Found sitemap in robots.txt: {url}")
    except Exception as exc:
        log.warning(f"Could not read robots.txt for sitemaps ({exc}).")

    if not sitemap_urls:
        fallback = urljoin(base_url, "/sitemap.xml")
        log.info(f"No sitemap declared in robots.txt, trying fallback: {fallback}")
        sitemap_urls.append(fallback)

    return sitemap_urls


# ---------------------------------------------------------------------------
# SECTION 2: Sitemap Discovery (recursive)
# ---------------------------------------------------------------------------

def fetch_sitemap_xml(url: str) -> Optional[bytes]:
    """
    Synchronously fetch a sitemap XML (or sitemap index) URL.

    Returns raw bytes on success, None on failure.
    Handles gzip-compressed sitemaps transparently via httpx.
    """
    try:
        with httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=30,
            follow_redirects=True,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        log.warning(f"Failed to fetch sitemap {url}: {exc}")
        return None


def parse_sitemap(xml_bytes: bytes, source_url: str) -> tuple[list[str], list[str]]:
    """
    Parse a sitemap XML document.

    Returns:
        (page_urls, child_sitemap_urls)
        - page_urls: <url><loc>...</loc></url> entries (actual pages)
        - child_sitemap_urls: <sitemap><loc>...</loc></sitemap> entries
          (for sitemap index files)
    """
    page_urls: list[str] = []
    child_sitemaps: list[str] = []
    SM_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"

    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as exc:
        log.warning(f"XML parse error for {source_url}: {exc}")
        return page_urls, child_sitemaps

    tag = root.tag  # e.g. "{http://...}sitemapindex" or "{http://...}urlset"

    if "sitemapindex" in tag:
        # This is a sitemap index — collect child sitemap URLs
        for sitemap_elem in root.iter(f"{{{SM_NS}}}sitemap"):
            loc = sitemap_elem.find(f"{{{SM_NS}}}loc")
            if loc is not None and loc.text:
                child_sitemaps.append(loc.text.strip())
    else:
        # This is a regular URL set — collect page URLs
        for url_elem in root.iter(f"{{{SM_NS}}}url"):
            loc = url_elem.find(f"{{{SM_NS}}}loc")
            if loc is not None and loc.text:
                page_urls.append(loc.text.strip())

    return page_urls, child_sitemaps


def discover_all_urls(base_url: str) -> list[str]:
    """
    Recursively discover ALL page URLs from the domain's sitemap(s).

    Flow:
      robots.txt -> sitemap URL(s) -> if sitemap index -> recurse into
      child sitemaps -> collect all <url><loc> entries.

    Returns a deduplicated list of page URLs.
    """
    log.info("=== Phase 1: Sitemap Discovery ===")
    seed_sitemaps = get_sitemap_urls_from_robots(base_url)
    visited_sitemaps: set[str] = set()
    all_page_urls: list[str] = []

    # BFS queue of sitemap URLs to process
    queue = list(seed_sitemaps)

    while queue:
        sitemap_url = queue.pop(0)
        if sitemap_url in visited_sitemaps:
            continue
        visited_sitemaps.add(sitemap_url)

        log.info(f"Fetching sitemap: {sitemap_url}")
        xml_bytes = fetch_sitemap_xml(sitemap_url)
        if xml_bytes is None:
            continue

        page_urls, child_sitemaps = parse_sitemap(xml_bytes, sitemap_url)
        log.info(
            f"  -> {len(page_urls)} pages, {len(child_sitemaps)} child sitemaps"
        )

        all_page_urls.extend(page_urls)
        for child in child_sitemaps:
            if child not in visited_sitemaps:
                queue.append(child)

    # Fallback to HTML crawling if sitemap found nothing
    if not all_page_urls:
        log.info("No URLs found in sitemap, attempting basic HTML crawling on base URL...")
        try:
            with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True) as client:
                resp = client.get(base_url)
                if resp.status_code == 200:
                    tree = etree.HTML(resp.content)
                    if tree is not None:
                        for a in tree.xpath("//a[@href]"):
                            href = a.get("href").strip()
                            # abaikan link anchor/hash dan link eksternal
                            if href.startswith("/") and not href.startswith("//"):
                                all_page_urls.append(base_url.rstrip("/") + href)
                            elif href.startswith(base_url):
                                all_page_urls.append(href)
        except Exception as exc:
            log.warning(f"Fallback HTML crawling failed: {exc}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_urls = []
    for url in all_page_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    log.info(f"Discovered {len(unique_urls)} unique URLs total.")
    return unique_urls


# ---------------------------------------------------------------------------
# SECTION 3: URL Filtering
# ---------------------------------------------------------------------------

def is_same_domain(url: str, base_url: str) -> bool:
    """Return True if url belongs to the same domain as base_url."""
    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)
    return parsed_url.netloc == parsed_base.netloc


def has_excluded_lang_prefix(url_path: str, lang_filter: str) -> bool:
    """
    Return True if this URL path starts with a language prefix that is
    NOT the desired language (i.e., should be excluded).

    Example: lang_filter="en", path="/de/blog/post" -> True (exclude)
             lang_filter="en", path="/blog/post"    -> False (keep)
    """
    # Match paths like /xx/ or /xx-XX/ or /xx_XX/ at the start
    match = re.match(r"^/([a-z]{2}(?:[_-][a-zA-Z]{2,4})?)/", url_path)
    if match:
        detected_lang = match.group(1).split("-")[0].split("_")[0]
        # Exclude if prefix does NOT match desired lang
        return detected_lang != lang_filter
    return False  # No lang prefix -> keep (treat as target language)


def filter_urls(
    urls: list[str],
    config: ScrapeConfig,
    robot_parser: RobotFileParser,
) -> list[str]:
    """
    Filter the discovered URL list based on:
      1. Same domain check
      2. robots.txt Disallow rules
      3. Language prefix filter (--lang-filter)
      4. Path substring filter (--path-filter)

    Returns the filtered list.
    """
    log.info("=== Phase 2: URL Filtering ===")
    kept = []
    stats = {"domain": 0, "robots": 0, "lang": 0, "path": 0}

    for url in urls:
        parsed = urlparse(url)

        # 1) Same domain
        if not is_same_domain(url, config.base_url):
            stats["domain"] += 1
            continue

        # 2) robots.txt
        if not robot_parser.can_fetch(USER_AGENT, url):
            stats["robots"] += 1
            log.debug(f"Blocked by robots.txt: {url}")
            continue

        # 3) Language filter
        if config.lang_filter:
            if has_excluded_lang_prefix(parsed.path, config.lang_filter):
                stats["lang"] += 1
                continue

        # 4) Path filter (substring match)
        if config.path_filter:
            if config.path_filter not in parsed.path:
                stats["path"] += 1
                continue

        kept.append(url)

    log.info(
        f"Filtered: {stats['domain']} off-domain, "
        f"{stats['robots']} robots-blocked, "
        f"{stats['lang']} lang-excluded, "
        f"{stats['path']} path-excluded. "
        f"Remaining: {len(kept)} URLs."
    )
    return kept


# ---------------------------------------------------------------------------
# SECTION 4: Checkpoint / Resume
# ---------------------------------------------------------------------------

def load_progress(output_dir: Path) -> dict:
    """
    Load scrape progress from progress.json in the output directory.

    Structure:
    {
        "completed": {"url": "success" | "failed"}
    }

    On resume, URLs already in 'completed' are skipped.
    """
    progress_path = output_dir / PROGRESS_FILE
    if progress_path.exists():
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            completed = len(data.get("completed", {}))
            log.info(
                f"Resuming from checkpoint: {completed} already processed."
            )
            return data
        except Exception as exc:
            log.warning(f"Could not load progress.json ({exc}), starting fresh.")
    return {"completed": {}}


def save_progress(output_dir: Path, progress: dict) -> None:
    """Persist progress dict to progress.json (atomic write via temp file)."""
    progress_path = output_dir / PROGRESS_FILE
    tmp_path = progress_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
        tmp_path.replace(progress_path)
    except Exception as exc:
        log.error(f"Failed to save progress: {exc}")


# ---------------------------------------------------------------------------
# SECTION 5: HTTP Client with Retry Logic
# ---------------------------------------------------------------------------

def _fetch_sync(url: str, config: ScrapeConfig) -> str:
    """
    Synchronous HTTP GET with manual exponential backoff retry.

    Retries on:
      - httpx.TimeoutException
      - HTTP status codes in RETRYABLE_STATUS_CODES (429, 5xx)
    Maximum 3 attempts, waiting 2->4->8 seconds between retries.

    Returns the response HTML text on success.
    Raises the last exception if all attempts fail.
    """
    last_exc: Exception = Exception("Unknown error")

    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=config.timeout,
        follow_redirects=True,
    ) as client:
        for attempt in range(1, 4):  # attempts 1, 2, 3
            try:
                resp = client.get(url)

                # Retryable HTTP status
                if resp.status_code in RETRYABLE_STATUS_CODES:
                    wait = 2 ** attempt
                    log.debug(
                        f"Status {resp.status_code} for {url}, "
                        f"retry {attempt}/3 in {wait}s"
                    )
                    last_exc = httpx.HTTPStatusError(
                        f"Retryable status {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                    if attempt < 3:
                        time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.text

            except httpx.TimeoutException as exc:
                wait = 2 ** attempt
                log.debug(f"Timeout for {url}, retry {attempt}/3 in {wait}s")
                last_exc = exc
                if attempt < 3:
                    time.sleep(wait)

            except httpx.HTTPStatusError:
                raise   # Non-retryable 4xx errors bubble up immediately

    raise last_exc


# ---------------------------------------------------------------------------
# SECTION 6: Content Extraction
# ---------------------------------------------------------------------------

def extract_content(html: str, url: str) -> PageResult:
    """
    Extract clean article content from raw HTML using trafilatura.

    trafilatura automatically removes:
      - Navigation bars, footers, sidebars
      - Cookie banners, advertisements
      - Boilerplate/duplicate content

    Returns a PageResult with title, language, and cleaned Markdown content.
    """
    result = PageResult(url=url)
    result.scraped_at = datetime.now(timezone.utc).isoformat()

    # Extract metadata (title, language, author, etc.)
    metadata = trafilatura.extract_metadata(html, default_url=url)
    if metadata:
        result.title = metadata.title or ""
        result.language = metadata.language or ""

    # Extract main content as Markdown
    extracted = trafilatura.extract(
        html,
        url=url,
        include_tables=True,
        include_links=False,
        include_images=False,
        output_format="markdown",   # output directly as Markdown
        favor_precision=False,      # favor recall for knowledge base use
        deduplicate=True,
    )

    if extracted:
        result.content = extracted
        result.status = "success"
    else:
        # Fallback: try to grab at least the page title from lxml
        try:
            tree = etree.HTML(html.encode("utf-8"))
            if tree is not None:
                title_nodes = tree.xpath("//title/text()")
                if title_nodes and not result.title:
                    result.title = title_nodes[0].strip()
        except Exception:
            pass
        result.status = "failed"
        result.error = "trafilatura returned no content"

    return result


# ---------------------------------------------------------------------------
# SECTION 7: Output Writers
# ---------------------------------------------------------------------------

def url_to_filename(url: str) -> str:
    """
    Convert a URL to a safe filename slug (without extension).

    Example:
        https://example.com/blog/my-post/ -> example_com_blog_my-post
    """
    parsed = urlparse(url)
    slug = (parsed.netloc + parsed.path).strip("/")
    slug = re.sub(r"[^\w\-]", "_", slug)   # replace non-word chars with _
    slug = re.sub(r"_+", "_", slug)         # collapse multiple underscores
    slug = slug.strip("_")
    return slug[:200]  # cap at 200 chars to avoid OS filename limits


def write_markdown(result: PageResult, output_dir: Path) -> Path:
    """
    Write a PageResult to an individual Markdown file with YAML frontmatter.

    File format:
        ---
        url: https://...
        title: "Page Title"
        language: en
        scraped_at: 2024-01-01T00:00:00+00:00
        ---

        # Page Title

        <cleaned article content in Markdown>
    """
    filename = url_to_filename(result.url) + ".md"
    filepath = output_dir / "pages" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Safe YAML title: escape double quotes
    safe_title = result.title.replace('"', "'")

    frontmatter = "\n".join([
        "---",
        f"url: {result.url}",
        f'title: "{safe_title}"',
        f"language: {result.language}",
        f"scraped_at: {result.scraped_at}",
        "---",
        "",
    ])

    body_parts = []
    if result.title:
        body_parts.append(f"# {result.title}\n")
    body_parts.append(result.content)

    filepath.write_text(frontmatter + "\n".join(body_parts), encoding="utf-8")
    return filepath


def append_jsonl(result: PageResult, output_dir: Path) -> None:
    """
    Append a single JSON record to combined.jsonl.

    Each line is a self-contained JSON object suitable for direct import
    into a vector database or RAG pipeline (LangChain, LlamaIndex, etc.).

    Format: {"url": ..., "title": ..., "language": ..., "scraped_at": ..., "content": ...}
    """
    jsonl_path = output_dir / COMBINED_JSONL
    record = {
        "url": result.url,
        "title": result.title,
        "language": result.language,
        "scraped_at": result.scraped_at,
        "content": result.content,
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# SECTION 8: Scraping Orchestrator
# ---------------------------------------------------------------------------

async def scrape_single_url(
    url: str,
    config: ScrapeConfig,
    semaphore: asyncio.Semaphore,
    progress: dict,
) -> PageResult:
    """
    Scrape one URL: acquire semaphore -> fetch HTML -> extract -> write outputs.

    The semaphore limits how many pages are fetched concurrently.
    HTTP fetching runs in a thread pool executor (via run_in_executor) because
    our retry logic uses synchronous httpx + time.sleep for clean backoff.
    """
    async with semaphore:
        loop = asyncio.get_event_loop()

        # Run blocking HTTP fetch in thread pool
        try:
            html = await loop.run_in_executor(None, _fetch_sync, url, config)
        except Exception as exc:
            result = PageResult(
                url=url,
                status="failed",
                error=str(exc),
                scraped_at=datetime.now(timezone.utc).isoformat(),
            )
            progress["completed"][url] = "failed"
            # Small delay even on failure to avoid hammering the server
            await asyncio.sleep(config.request_delay)
            return result

        # Extract content (CPU-bound, but fast enough to run inline)
        result = extract_content(html, url)

        # Persist output files only for successful extractions
        if result.status == "success":
            write_markdown(result, config.output_dir)
            append_jsonl(result, config.output_dir)
            progress["completed"][url] = "success"
        else:
            progress["completed"][url] = "failed"

        # Polite delay between requests
        await asyncio.sleep(config.request_delay)
        return result


async def run_scraper(
    urls: list[str],
    config: ScrapeConfig,
    progress: dict,
) -> list[PageResult]:
    """
    Main async scraping loop with bounded concurrency and tqdm progress bar.

    Processes URLs in batches of 100 to limit memory usage for very large
    sitemaps. Saves a checkpoint to progress.json every 20 completions so
    the run can be safely resumed after an interruption (Ctrl+C, crash, etc.).

    Returns the list of all PageResult objects from this run.
    """
    log.info(f"=== Phase 3: Scraping {len(urls)} URLs ===")
    log.info(f"Concurrency: {config.concurrency}, Delay: {config.request_delay}s/request")

    semaphore = asyncio.Semaphore(config.concurrency)
    results: list[PageResult] = []
    completed_count = 0
    CHECKPOINT_INTERVAL = 20    # save progress every N completions
    BATCH_SIZE = 100            # process in batches to control memory

    pbar = tqdm(
        total=len(urls),
        desc="Scraping",
        unit="page",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )

    for batch_start in range(0, len(urls), BATCH_SIZE):
        batch = urls[batch_start : batch_start + BATCH_SIZE]

        # Create all tasks for this batch simultaneously
        tasks = [
            scrape_single_url(url, config, semaphore, progress)
            for url in batch
        ]

        # Process as tasks complete (not in submission order)
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed_count += 1

            # Update progress bar with last URL and status
            status_icon = "+" if result.status == "success" else "-"
            short_url = result.url[-55:] if len(result.url) > 55 else result.url
            pbar.set_postfix_str(f"[{status_icon}] {short_url}")
            pbar.update(1)

            # Periodic checkpoint save
            if completed_count % CHECKPOINT_INTERVAL == 0:
                save_progress(config.output_dir, progress)

    pbar.close()
    save_progress(config.output_dir, progress)  # final checkpoint
    return results


# ---------------------------------------------------------------------------
# SECTION 9: Summary Report
# ---------------------------------------------------------------------------

def print_summary(results: list[PageResult], skipped_count: int) -> None:
    """Print a formatted summary of the completed scraping run."""
    success = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    total_processed = len(results)

    border = "=" * 60
    log.info("")
    log.info(border)
    log.info("  MaxyCrawl - Scraping Complete")
    log.info(border)
    log.info(f"  Total processed this run : {total_processed}")
    log.info(f"  [+] Succeeded            : {success}")
    log.info(f"  [-] Failed               : {failed}")
    log.info(f"  [~] Skipped (checkpoint) : {skipped_count}")
    log.info(border)

    if failed > 0:
        log.info("  Failed URLs (first 10):")
        shown = 0
        for r in results:
            if r.status == "failed":
                log.info(f"    - {r.url}")
                if r.error:
                    log.info(f"      Error: {r.error}")
                shown += 1
                if shown >= 10:
                    if failed > 10:
                        log.info(f"    ... and {failed - 10} more. Check progress.json for full list.")
                    break


# ---------------------------------------------------------------------------
# SECTION 10: CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="scraper.py",
        description="MaxyCrawl - AI Knowledge Base Web Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Scrape entire site:
    python scraper.py --url https://example.com

  Scrape only English blog posts with 8 parallel workers:
    python scraper.py --url https://example.com --lang-filter en \\
        --path-filter /blogs/ --concurrency 8

  Custom output directory:
    python scraper.py --url https://example.com --output ./my-knowledge-base

  Resume an interrupted run (just run the same command again):
    python scraper.py --url https://example.com --output ./output
        """,
    )

    parser.add_argument(
        "--url",
        required=True,
        metavar="URL",
        help="Base URL of the target website (e.g. https://example.com)",
    )
    parser.add_argument(
        "--lang-filter",
        metavar="LANG",
        default=None,
        help=(
            "Keep only URLs that are NOT prefixed with a foreign language code. "
            "E.g. 'en' excludes /de/, /zh-hans/, /fr/, etc. "
            "URLs without any language prefix are always kept."
        ),
    )
    parser.add_argument(
        "--path-filter",
        metavar="PATH",
        default=None,
        help="Keep only URLs whose path contains this substring (e.g. /blogs/)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        metavar="N",
        help=f"Number of parallel HTTP requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        metavar="SECS",
        help=f"Delay in seconds between requests per worker (default: {DEFAULT_REQUEST_DELAY})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        metavar="SECS",
        help=f"HTTP request timeout in seconds (default: {DEFAULT_REQUEST_TIMEOUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=f"Output directory path (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )

    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    """Main async orchestrator: runs all scraping phases in sequence."""

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Debug logging enabled.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pages").mkdir(parents=True, exist_ok=True)

    config = ScrapeConfig(
        base_url=args.url.rstrip("/"),
        lang_filter=args.lang_filter,
        path_filter=args.path_filter,
        concurrency=args.concurrency,
        output_dir=output_dir,
        request_delay=args.delay,
        timeout=args.timeout,
    )

    log.info(f"MaxyCrawl starting")
    log.info(f"Target    : {config.base_url}")
    log.info(f"Lang filter: {config.lang_filter or '(none)'}")
    log.info(f"Path filter: {config.path_filter or '(none)'}")
    log.info(f"Output dir : {output_dir.resolve()}")

    # --- Phase 1: Discover all URLs from sitemaps ---
    all_urls = discover_all_urls(config.base_url)

    # --- Phase 2: Filter URLs ---
    robot_parser = load_robots_txt(config.base_url)
    filtered_urls = filter_urls(all_urls, config, robot_parser)

    if not filtered_urls:
        log.warning("No URLs to scrape after filtering. Check your filters or sitemap.")
        return

    # --- Checkpoint: skip URLs already processed in a previous run ---
    progress = load_progress(output_dir)
    already_done = set(progress.get("completed", {}).keys())
    pending_urls = [u for u in filtered_urls if u not in already_done]
    skipped_count = len(filtered_urls) - len(pending_urls)

    if skipped_count > 0:
        log.info(f"Skipping {skipped_count} already-processed URLs (checkpoint resume).")

    if not pending_urls:
        log.info("All URLs already processed. Nothing to do.")
        print_summary([], skipped_count)
        return

    log.info(f"Will scrape {len(pending_urls)} pending URLs.")

    # --- Phase 3: Scrape ---
    results = await run_scraper(pending_urls, config, progress)

    # --- Summary ---
    print_summary(results, skipped_count)
    log.info(f"Markdown files : {(output_dir / 'pages').resolve()}")
    log.info(f"Combined JSONL : {(output_dir / COMBINED_JSONL).resolve()}")
    log.info(f"Progress file  : {(output_dir / PROGRESS_FILE).resolve()}")


def main() -> None:
    """Synchronous entry point - bootstraps the asyncio event loop."""
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        log.info(
            "\nInterrupted by user (Ctrl+C). "
            "Progress has been saved - run the same command again to resume."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
