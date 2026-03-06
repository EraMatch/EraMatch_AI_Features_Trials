import pypdf
import io
from fastapi import HTTPException

def get_pdf_page_count(file_content: bytes) -> int:
    """Returns the number of pages in a PDF."""
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_content))
        return len(pdf_reader.pages)
    except:
        return 0

def extract_text_from_pdf(file_content: bytes, max_pages: int = 10, force: bool = False) -> str:
    """Extracts text from a PDF file content, with optional page limit validation."""
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_content))
        num_pages = len(pdf_reader.pages)
        
        if num_pages > max_pages and not force:
            raise HTTPException(
                status_code=400, 
                detail=f"Large file detected ({num_pages} pages). Would you like to process this in the background?"
            )
        
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        
        return text.strip()
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error parsing PDF: {str(e)}")

def count_tokens(text: str) -> int:
    """Simple token count heuristic (words * 1.3)."""
    return int(len(text.split()) * 1.3)
