from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFHandler:
    def chunk_pdf(self, file_name: str, chunk_size: int = 1000, overlap_size: int = 200) -> list[str]:
        reader = PdfReader(file_name)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        full_text = self._clean_text(full_text)

        splitter = RecursiveCharacterTextSplitter( # tries to avoid a split in mid-sentence
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
        )
        return splitter.split_text(full_text)

    @staticmethod
    def _clean_text(text: str) -> str:
        import re
        text = text.replace("\t", " ")          # tabs → space
        text = re.sub(r"[^\S\n]+", " ", text)   # collapse whitespace runs (preserve newlines)
        text = re.sub(r"\n{3,}", "\n\n", text)  # collapse 3+ newlines to double
        text = re.sub(r"-\n", "", text)         # rejoin hyphenated line breaks
        return text.strip()


if __name__ == '__main__':
    filename = "/Users/yizhar/projects/practice_rag/data/Understanding_Climate_Change.pdf"
    chunks = PDFHandler().chunk_pdf(filename)
    print("end")

