from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.core.config import Settings
from app.core.exceptions import UnsupportedDocumentError


_MARKDOWN_HEADERS = [
    ("#", "title"),
    ("##", "section"),
    ("###", "subsection"),
]

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def load_documents(paths: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        elif suffix == ".docx":
            loaded = Docx2txtLoader(str(path)).load()
        elif suffix in {".txt", ".md"}:
            try:
                loaded = TextLoader(str(path), encoding="utf-8").load()
            except UnicodeDecodeError:
                loaded = TextLoader(str(path), autodetect_encoding=True).load()
        else:
            raise UnsupportedDocumentError(
                f"Format non pris en charge : {suffix or 'sans extension'}"
            )

        for document in loaded:
            document.metadata["source"] = path.name
        documents.extend(loaded)
    return documents


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    character_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MARKDOWN_HEADERS,
        strip_headers=False,
    )

    expanded: list[Document] = []
    for document in documents:
        is_markdown = Path(document.metadata.get("source", "")).suffix.lower() == ".md"
        if is_markdown and "\n#" in document.page_content:
            for piece in header_splitter.split_text(document.page_content):
                expanded.append(
                    Document(
                        page_content=piece.page_content,
                        metadata=dict(document.metadata) | dict(piece.metadata),
                    )
                )
        else:
            expanded.append(document)

    chunks = character_splitter.split_documents(expanded)
    return [chunk for chunk in chunks if chunk.page_content.strip()]
