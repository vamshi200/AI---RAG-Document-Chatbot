import os
import re
import tempfile
import fitz
import pytesseract
import docx2txt

from PIL import Image
from langchain_core.documents import Document


def looks_like_garbage(text: str) -> bool:
    if not text or len(text.strip()) < 40:
        return True

    weird_chars = len(re.findall(r"[^a-zA-Z0-9\s,.\-:/@()]", text))
    ratio = weird_chars / max(len(text), 1)

    useful_words = [
        "name", "date", "document", "invoice", "passport", "bank",
        "statement", "flight", "ticket", "resume", "education",
        "experience", "university", "policy", "report", "amount"
    ]
    has_useful_words = any(word in text.lower() for word in useful_words)

    return ratio > 0.25 and not has_useful_words


def extract_text_normally_from_pdf(pdf_path: str):
    documents = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        documents.append(
            Document(
                page_content=text,
                metadata={"source": pdf_path, "page": page_num + 1}
            )
        )

    doc.close()
    return documents


def extract_text_with_ocr(pdf_path: str):
    documents = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img)

        documents.append(
            Document(
                page_content=text,
                metadata={"source": pdf_path, "page": page_num + 1}
            )
        )

    doc.close()
    return documents


def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    try:
        documents = extract_text_normally_from_pdf(temp_path)
        combined_text = " ".join(doc.page_content for doc in documents[:2])

        if looks_like_garbage(combined_text):
            documents = extract_text_with_ocr(temp_path)

        return documents

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_docx(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    try:
        text = docx2txt.process(temp_path)
        return [
            Document(
                page_content=text,
                metadata={"source": uploaded_file.name, "page": 1}
            )
        ]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_document(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return load_pdf(uploaded_file)

    if filename.endswith(".docx"):
        return load_docx(uploaded_file)

    raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")