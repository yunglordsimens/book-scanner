# 📖 Book Scanner

Turn large, imperfect scanned book PDFs into readable electronic books.

## Features

- Upload one or multiple PDF files
- Combine PDFs into a single book or keep separate
- Extracts existing text layers first (fast, no OCR needed)
- OCR only pages that need it (via Tesseract)
- Incremental processing with progress tracking
- Resume interrupted jobs
- Export to **EPUB**, **Markdown**, **TXT**, and **searchable PDF**

## Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open http://localhost:5001

**Requires:** Python 3.10+, [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

## Deploy to Render

1. Fork/push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect this repo
4. Render auto-detects `render.yaml` — click Deploy

## License

MIT
