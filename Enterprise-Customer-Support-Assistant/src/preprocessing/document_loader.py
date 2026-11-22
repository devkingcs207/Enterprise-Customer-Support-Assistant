
"""
PDF document loader using PyMuPDF.
No LangChain dependency required.
"""

from pathlib import Path
from dataclasses import dataclass, field

import pymupdf


@dataclass
class Document:
    """
    Simple document structure compatible with
    the project's ingestion pipeline.
    """

    page_content: str
    metadata: dict = field(default_factory=dict)


def load_pdf_documents(folder_path="documents"):

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {folder.resolve()}"
        )

    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {folder.resolve()}")
        return []

    documents = []

    for pdf_path in pdf_files:

        try:

            with pymupdf.open(pdf_path) as pdf:

                for page_number, page in enumerate(pdf, start=1):

                    text = page.get_text("text").strip()

                    if not text:
                        continue

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_number
                            }
                        )
                    )

            print(f"Loaded: {pdf_path.name}")

        except Exception as error:

            print(
                f"Could not read {pdf_path.name}: {error}"
            )

    print(f"\nTotal pages loaded: {len(documents)}")

    return documents
