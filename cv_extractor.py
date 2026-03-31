"""
CV text extraction from PDF / DOCX files.
Reuses the same logic as the main project, kept standalone.
"""
import PyPDF2
import docx
from io import BytesIO


def extract_text(file_content: bytes, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return _extract_pdf(file_content)
    if ext in ("docx", "doc"):
        return _extract_docx(file_content)
    raise ValueError(f"Desteklenmeyen dosya formatı: .{ext}  (PDF veya DOCX yükleyin)")


def _extract_pdf(data: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(data))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages).strip()


def _extract_docx(data: bytes) -> str:
    doc = docx.Document(BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs).strip()
