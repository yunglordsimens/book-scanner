#!/usr/bin/env python3
"""
Book Scanner — Turn scanned PDFs into readable ebooks.
Single-file Flask backend with incremental OCR processing.
"""

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
import pytesseract
from ebooklib import epub
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB upload limit

DATA_DIR = Path(os.environ.get("BOOKSCANNER_DATA", Path(__file__).parent / "data"))
UPLOADS_DIR = DATA_DIR / "uploads"
BOOKS_DIR = DATA_DIR / "books"
EXPORTS_DIR = DATA_DIR / "exports"

for d in [UPLOADS_DIR, BOOKS_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# In-memory job tracking (persisted to disk as JSON)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers — persistence
# ---------------------------------------------------------------------------

def _book_meta_path(book_id: str) -> Path:
    return BOOKS_DIR / book_id / "meta.json"


def _load_book(book_id: str) -> dict | None:
    p = _book_meta_path(book_id)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _save_book(book: dict):
    d = BOOKS_DIR / book["id"]
    d.mkdir(parents=True, exist_ok=True)
    (_book_meta_path(book["id"])).write_text(json.dumps(book, indent=2))


def _all_books() -> list[dict]:
    books = []
    if BOOKS_DIR.exists():
        for d in sorted(BOOKS_DIR.iterdir()):
            meta = d / "meta.json"
            if meta.exists():
                books.append(json.loads(meta.read_text()))
    return books


def _page_text_path(book_id: str, pdf_hash: str, page_num: int) -> Path:
    return BOOKS_DIR / book_id / "pages" / f"{pdf_hash}_p{page_num:06d}.txt"


def _save_page_text(book_id: str, pdf_hash: str, page_num: int, text: str, method: str):
    p = _page_text_path(book_id, pdf_hash, page_num)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Store metadata alongside
    meta = {
        "pdf_hash": pdf_hash,
        "page_num": page_num,
        "method": method,  # "text_layer" or "ocr"
        "char_count": len(text),
    }
    p.write_text(text)
    p.with_suffix(".meta.json").write_text(json.dumps(meta))


def _file_hash(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Helpers — PDF processing
# ---------------------------------------------------------------------------

MIN_TEXT_CHARS = 30  # Pages with fewer chars get OCR'd


def _extract_text_from_page(doc: fitz.Document, page_num: int) -> tuple[str, str]:
    """Extract text from a single PDF page.
    Returns (text, method) where method is 'text_layer' or 'ocr'.
    """
    page = doc[page_num]
    text = page.get_text().strip()
    if len(text) >= MIN_TEXT_CHARS:
        return text, "text_layer"
    # Need OCR — render page to image
    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    pix = None  # free memory
    ocr_text = pytesseract.image_to_string(img, lang="eng")
    img.close()
    return ocr_text.strip(), "ocr"


def _process_book_job(job_id: str, book_id: str):
    """Background worker: process all pages of all PDFs in a book."""
    book = _load_book(book_id)
    if not book:
        _update_job(job_id, status="error", error="Book not found")
        return

    # Collect all (pdf_path, pdf_hash, page_count) tuples
    work_items = []
    for pdf_info in book.get("pdfs", []):
        pdf_path = Path(pdf_info["path"])
        if not pdf_path.exists():
            _update_job(job_id, status="error", error=f"PDF not found: {pdf_path.name}")
            return
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        doc.close()
        work_items.append((pdf_path, pdf_info["hash"], page_count))

    total_pages = sum(pc for _, _, pc in work_items)
    _update_job(job_id, status="processing", total=total_pages, processed=0, errors=[])

    processed = 0
    errors = []
    for pdf_path, pdf_hash, page_count in work_items:
        doc = fitz.open(str(pdf_path))
        for page_num in range(page_count):
            # Check if already done (resume support)
            existing = _page_text_path(book_id, pdf_hash, page_num)
            if existing.exists():
                processed += 1
                _update_job(job_id, processed=processed)
                continue
            try:
                text, method = _extract_text_from_page(doc, page_num)
                _save_page_text(book_id, pdf_hash, page_num, text, method)
            except Exception as e:
                err_msg = f"Page {page_num + 1} of {pdf_path.name}: {e}"
                errors.append(err_msg)
                # Save empty page so we don't re-process, but mark as error
                _save_page_text(book_id, pdf_hash, page_num, f"[OCR ERROR: {e}]", "error")
            processed += 1
            _update_job(job_id, processed=processed, errors=errors)
        doc.close()

    # Update book status
    book = _load_book(book_id)
    book["status"] = "processed"
    book["processed_pages"] = processed
    book["total_pages"] = total_pages
    _save_book(book)

    final_status = "done" if not errors else "done_with_errors"
    _update_job(job_id, status=final_status)


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return _jobs.get(job_id, {}).copy() if job_id in _jobs else None


# ---------------------------------------------------------------------------
# Helpers — Export
# ---------------------------------------------------------------------------

def _collect_book_text(book_id: str) -> list[dict]:
    """Collect all page texts in order. Returns list of {pdf_name, page_num, text, method}."""
    book = _load_book(book_id)
    if not book:
        return []
    pages = []
    for pdf_info in book.get("pdfs", []):
        pdf_hash = pdf_info["hash"]
        page_num = 0
        while True:
            p = _page_text_path(book_id, pdf_hash, page_num)
            if not p.exists():
                break
            meta_path = p.with_suffix(".meta.json")
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            pages.append({
                "pdf_name": pdf_info["name"],
                "page_num": page_num,
                "text": p.read_text(),
                "method": meta.get("method", "unknown"),
            })
            page_num += 1
    return pages


def _export_markdown(book_id: str) -> Path:
    book = _load_book(book_id)
    pages = _collect_book_text(book_id)
    out_path = EXPORTS_DIR / f"{book['title']}.md"
    with open(out_path, "w") as f:
        f.write(f"# {book['title']}\n\n")
        current_pdf = None
        for pg in pages:
            if pg["pdf_name"] != current_pdf:
                current_pdf = pg["pdf_name"]
                f.write(f"\n## Source: {current_pdf}\n\n")
            f.write(f"<!-- Page {pg['page_num'] + 1} ({pg['method']}) -->\n")
            f.write(pg["text"])
            f.write("\n\n---\n\n")
    return out_path


def _export_txt(book_id: str) -> Path:
    book = _load_book(book_id)
    pages = _collect_book_text(book_id)
    out_path = EXPORTS_DIR / f"{book['title']}.txt"
    with open(out_path, "w") as f:
        for pg in pages:
            f.write(pg["text"])
            f.write("\n\n")
    return out_path


def _export_epub(book_id: str) -> Path:
    book = _load_book(book_id)
    pages = _collect_book_text(book_id)
    out_path = EXPORTS_DIR / f"{book['title']}.epub"

    eb = epub.EpubBook()
    eb.set_identifier(book["id"])
    eb.set_title(book["title"])
    eb.set_language("en")
    eb.add_author("Book Scanner")

    # CSS for readability
    style = epub.EpubItem(
        uid="style",
        file_name="style/default.css",
        media_type="text/css",
        content=b"""
body { font-family: Georgia, serif; line-height: 1.6; margin: 1em; }
p { text-indent: 1.5em; margin: 0.3em 0; }
h1, h2 { text-indent: 0; }
.page-break { page-break-before: always; }
.source-info { color: #999; font-size: 0.8em; font-style: italic; }
""",
    )
    eb.add_item(style)

    # Group pages into chapters (one per PDF source, or one big chapter)
    chapters = []
    current_pdf = None
    current_html_parts = []
    chapter_idx = 0

    def _flush_chapter(pdf_name):
        nonlocal chapter_idx, current_html_parts
        if not current_html_parts:
            return
        chapter_idx += 1
        ch = epub.EpubHtml(
            title=pdf_name or f"Chapter {chapter_idx}",
            file_name=f"chapter_{chapter_idx:03d}.xhtml",
            lang="en",
        )
        html = f"<h2>{pdf_name or f'Chapter {chapter_idx}'}</h2>\n"
        html += "\n".join(current_html_parts)
        ch.content = html.encode("utf-8")
        ch.add_item(style)
        eb.add_item(ch)
        chapters.append(ch)
        current_html_parts = []

    for pg in pages:
        if pg["pdf_name"] != current_pdf:
            _flush_chapter(current_pdf)
            current_pdf = pg["pdf_name"]

        # Convert text to HTML paragraphs
        text = pg["text"]
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        page_html = f'<div class="page-break">\n'
        page_html += f'<p class="source-info">Page {pg["page_num"] + 1}</p>\n'
        for para in paragraphs:
            # Escape HTML
            para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            page_html += f"<p>{para}</p>\n"
        page_html += "</div>\n"
        current_html_parts.append(page_html)

    _flush_chapter(current_pdf)

    # Table of contents and spine
    eb.toc = chapters
    eb.add_item(epub.EpubNcx())
    eb.add_item(epub.EpubNav())
    eb.spine = ["nav"] + chapters

    epub.write_epub(str(out_path), eb)
    return out_path


def _export_searchable_pdf(book_id: str) -> Path | None:
    """Create a searchable PDF by overlaying OCR text on original pages."""
    book = _load_book(book_id)
    if not book:
        return None

    out_path = EXPORTS_DIR / f"{book['title']}_searchable.pdf"
    
    # For a simple searchable PDF, we merge original PDFs 
    # (PyMuPDF preserves existing text layers)
    # For pages that were OCR'd, we insert an invisible text layer
    writer = fitz.open()  # new empty PDF
    
    for pdf_info in book.get("pdfs", []):
        pdf_path = Path(pdf_info["path"])
        if not pdf_path.exists():
            continue
        src = fitz.open(str(pdf_path))
        for page_num in range(len(src)):
            # Copy original page
            writer.insert_pdf(src, from_page=page_num, to_page=page_num)
            
            # Check if this page was OCR'd — if so, add invisible text overlay
            text_path = _page_text_path(book_id, pdf_info["hash"], page_num)
            meta_path = text_path.with_suffix(".meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta.get("method") == "ocr" and text_path.exists():
                    ocr_text = text_path.read_text()
                    if ocr_text.strip():
                        # Insert invisible text on the copied page
                        out_page = writer[-1]
                        # Add as invisible text (tiny font, transparent)
                        rect = out_page.rect
                        # Insert a text block with very small invisible text
                        tw = fitz.TextWriter(rect)
                        try:
                            font = fitz.Font("helv")
                            # Place text in a small area — it makes the page searchable
                            tw.append((rect.x0 + 1, rect.y1 - 1), ocr_text[:5000], font=font, fontsize=1)
                            tw.write_text(out_page, color=(1, 1, 1), render_mode=3)  # invisible
                        except Exception:
                            pass  # Non-critical — original page is still there
        src.close()
    
    writer.save(str(out_path), deflate=True, garbage=4)
    writer.close()
    return out_path


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    """Upload one or more PDF files."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    uploaded = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        # Save to uploads dir with unique name
        file_id = uuid.uuid4().hex[:8]
        safe_name = f.filename.replace("/", "_").replace("\\", "_")
        dest = UPLOADS_DIR / f"{file_id}_{safe_name}"
        f.save(str(dest))
        file_hash = _file_hash(dest)
        uploaded.append({
            "id": file_id,
            "name": safe_name,
            "path": str(dest),
            "hash": file_hash,
            "size": dest.stat().st_size,
        })
    return jsonify({"uploaded": uploaded})


@app.route("/api/books", methods=["GET"])
def list_books():
    return jsonify({"books": _all_books()})


@app.route("/api/books", methods=["POST"])
def create_book():
    """Create a new book from uploaded PDFs."""
    data = request.get_json()
    title = data.get("title", "Untitled Book")
    pdf_files = data.get("pdfs", [])  # list of {name, path, hash}

    book_id = uuid.uuid4().hex[:12]
    book = {
        "id": book_id,
        "title": title,
        "status": "created",
        "created_at": time.time(),
        "pdfs": pdf_files,
        "processed_pages": 0,
        "total_pages": 0,
    }
    _save_book(book)
    return jsonify({"book": book})


@app.route("/api/books/<book_id>/add-pdfs", methods=["POST"])
def add_pdfs_to_book(book_id: str):
    """Add more PDFs to an existing book."""
    book = _load_book(book_id)
    if not book:
        return jsonify({"error": "Book not found"}), 404
    data = request.get_json()
    new_pdfs = data.get("pdfs", [])
    # Check for duplicates by hash
    existing_hashes = {p["hash"] for p in book["pdfs"]}
    added = []
    for pdf in new_pdfs:
        if pdf["hash"] not in existing_hashes:
            book["pdfs"].append(pdf)
            existing_hashes.add(pdf["hash"])
            added.append(pdf)
    book["status"] = "created"  # reset so it can be re-processed
    _save_book(book)
    return jsonify({"book": book, "added": added})


@app.route("/api/books/<book_id>/process", methods=["POST"])
def process_book(book_id: str):
    """Start processing a book (extract text / OCR)."""
    book = _load_book(book_id)
    if not book:
        return jsonify({"error": "Book not found"}), 404

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "book_id": book_id,
            "status": "starting",
            "total": 0,
            "processed": 0,
            "errors": [],
        }

    thread = threading.Thread(target=_process_book_job, args=(job_id, book_id), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"job": job})


@app.route("/api/books/<book_id>/preview", methods=["GET"])
def preview_book(book_id: str):
    """Get paginated preview of extracted text."""
    book = _load_book(book_id)
    if not book:
        return jsonify({"error": "Book not found"}), 404

    page = int(request.args.get("page", 0))
    per_page = int(request.args.get("per_page", 20))

    all_pages = _collect_book_text(book_id)
    total = len(all_pages)
    start = page * per_page
    end = min(start + per_page, total)

    return jsonify({
        "book": book,
        "pages": all_pages[start:end],
        "total_pages": total,
        "page": page,
        "per_page": per_page,
    })


@app.route("/api/books/<book_id>/export/<fmt>", methods=["GET"])
def export_book(book_id: str, fmt: str):
    book = _load_book(book_id)
    if not book:
        return jsonify({"error": "Book not found"}), 404

    try:
        if fmt == "epub":
            path = _export_epub(book_id)
            return send_file(str(path), as_attachment=True, download_name=f"{book['title']}.epub")
        elif fmt == "markdown":
            path = _export_markdown(book_id)
            return send_file(str(path), as_attachment=True, download_name=f"{book['title']}.md")
        elif fmt == "txt":
            path = _export_txt(book_id)
            return send_file(str(path), as_attachment=True, download_name=f"{book['title']}.txt")
        elif fmt == "pdf":
            path = _export_searchable_pdf(book_id)
            if path:
                return send_file(str(path), as_attachment=True, download_name=f"{book['title']}_searchable.pdf")
            return jsonify({"error": "PDF export failed"}), 500
        else:
            return jsonify({"error": f"Unknown format: {fmt}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    print(f"Book Scanner running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
