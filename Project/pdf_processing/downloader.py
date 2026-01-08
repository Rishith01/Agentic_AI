# pdf_processing/downloader.py
import os
import requests

PDF_DIR = "data/pdfs"

os.makedirs(PDF_DIR, exist_ok=True)

def get_pdf_path(paper_id: str) -> str:
    return os.path.join(PDF_DIR, f"{paper_id}.pdf")


def download_pdf(paper):
    """
    Downloads PDF if not already present.
    Returns local PDF path.
    """
    pdf_path = get_pdf_path(paper.source_id)

    if os.path.exists(pdf_path):
        return pdf_path

    if not hasattr(paper, "pdf_url") or not paper.pdf_url:
        raise RuntimeError("Paper has no PDF URL")

    try:
        resp = requests.get(paper.pdf_url, timeout=10)
        resp.raise_for_status()

        with open(pdf_path, "wb") as f:
            f.write(resp.content)

        return pdf_path

    except Exception as e:
        raise RuntimeError(f"PDF download failed: {e}")
