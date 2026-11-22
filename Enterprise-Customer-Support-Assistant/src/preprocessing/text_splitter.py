
"""
Pure Python text splitter.
No LangChain dependency required.
Splits text around policy section headings where possible.
"""

import re

from src.preprocessing.document_loader import Document


def split_into_sections(text):
    """
    Split text at numbered policy headings such as:
    1. PRODUCT WARRANTY
    2. PRODUCT REPLACEMENT
    """

    heading_pattern = re.compile(
        r"(?m)^\s*\d+\.\s+[A-Z][A-Z\s/&()-]+\s*$"
    )

    matches = list(heading_pattern.finditer(text))

    if not matches:
        return [text.strip()]

    sections = []

    # Preserve text before the first heading, if any.
    if matches[0].start() > 0:
        intro = text[:matches[0].start()].strip()
        if intro:
            sections.append(intro)

    for index, match in enumerate(matches):
        section_start = match.start()

        if index + 1 < len(matches):
            section_end = matches[index + 1].start()
        else:
            section_end = len(text)

        section = text[section_start:section_end].strip()

        if section:
            sections.append(section)

    return sections


def split_text_into_chunks(
    text,
    chunk_size=500,
    chunk_overlap=100
):
    """Split text into overlapping chunks."""

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        if end < text_length:
            boundary = text.rfind(
                " ",
                start + int(chunk_size * 0.7),
                end
            )

            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end >= text_length:
            break

        start = max(start + 1, end - chunk_overlap)

    return chunks


def split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100
):
    """
    Split documents into policy-section-aware text chunks.

    Preserves source and page metadata.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative "
            "and smaller than chunk_size."
        )

    chunks = []

    for document in documents:

        text = document.page_content.strip()

        if not text:
            continue

        sections = split_into_sections(text)

        for section in sections:

            section_chunks = split_text_into_chunks(
                section,
                chunk_size,
                chunk_overlap
            )

            for chunk_text in section_chunks:

                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=dict(document.metadata)
                    )
                )

    print(f"Original documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    return chunks
