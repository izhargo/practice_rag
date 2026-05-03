from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFHandler:
    def chunk_pdf(self, file_name: str, chunk_size: int = 1000, overlap_size: int = 200) -> list[str]:
        reader = PdfReader(file_name)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
        )
        return splitter.split_text(full_text)

