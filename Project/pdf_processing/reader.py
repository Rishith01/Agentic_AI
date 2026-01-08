# pdf_processing/reader.py
import fitz  # PyMuPDF

def read_pdf_text(pdf_path: str, max_pages=10) -> str:
    """
    Extract text from first N pages.
    """
    doc = fitz.open(pdf_path)
    text_chunks = []

    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text_chunks.append(page.get_text())

    doc.close()
    return "\n".join(text_chunks)
