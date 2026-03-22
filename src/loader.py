from pathlib import Path

import fitz
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader


def extract_text_from_pdf_with_ocr(file_path: str):
    docs = []
    pdf = fitz.open(file_path)

    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img_path = Path(file_path).with_suffix(f".page_{page_num + 1}.png")
        pix.save(str(img_path))

        image = Image.open(img_path)
        ocr_text = pytesseract.image_to_string(image)

        docs.append(
            Document(
                page_content=ocr_text,
                metadata={"source": file_path, "page": page_num + 1, "ocr": True},
            )
        )

        try:
            img_path.unlink(missing_ok=True)
        except Exception:
            pass

    pdf.close()
    return docs


def load_document(file_path: str, extension: str):
    extension = extension.lower()

    if extension == ".pdf":
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            combined_text = "\n".join(doc.page_content for doc in docs).strip()

            # Use normal PDF text whenever it has enough readable content.
            if combined_text and len(combined_text) >= 100:
                return docs

            return extract_text_from_pdf_with_ocr(file_path)
        except Exception:
            return extract_text_from_pdf_with_ocr(file_path)

    if extension == ".docx":
        loader = Docx2txtLoader(file_path)
        return loader.load()

    raise ValueError(f"Unsupported file type: {extension}")