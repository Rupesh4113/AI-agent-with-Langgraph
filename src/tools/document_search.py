"""
Document Search Tool.
Searches local knowledge documents (markdown, text) using term-frequency scoring.
Returns relevant excerpts, document metadata, and similarity scores.
"""
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List
from src.config import settings

def _tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alpha-numeric words."""
    return [word.lower() for word in re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text)]

def _compute_relevance(query_tokens: List[str], chunk_tokens: List[str]) -> float:
    """Computes a normalized overlap score between query and document chunk."""
    if not query_tokens or not chunk_tokens:
        return 0.0
    query_set = set(query_tokens)
    overlap = sum(1 for token in chunk_tokens if token in query_set)
    # Cosine-like normalization
    score = overlap / math.sqrt(len(query_set) * max(len(chunk_tokens), 1))
    return min(round(score, 3), 1.0)

def search_documents(
    query: str,
    docs_dir: Path = settings.SAMPLE_DOCS_DIR,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Searches local documents for relevant passages matching the query.
    
    Args:
        query: Research question or keywords.
        docs_dir: Directory containing text/markdown files.
        top_k: Maximum number of passage matches to return.
        
    Returns:
        List of structured passage matches with title, excerpt, and score.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    target_dir = Path(docs_dir)
    if not target_dir.exists():
        return []

    matches: List[Dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    # Scan text and markdown files
    for file_path in target_dir.glob("*.*"):
        if file_path.suffix.lower() not in [".txt", ".md", ".json", ".rst"]:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Split into meaningful paragraphs/sections
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if len(p.strip()) > 30]
        if not paragraphs:
            paragraphs = [content.strip()]

        for p_idx, paragraph in enumerate(paragraphs):
            p_tokens = _tokenize(paragraph)
            score = _compute_relevance(query_tokens, p_tokens)
            if score > 0.05:
                # Extract clean title from document header or filename
                lines = paragraph.splitlines()
                first_line = lines[0].lstrip("# ").strip()
                title = first_line if first_line else file_path.stem.replace("_", " ").title()

                matches.append({
                    "source_title": f"{file_path.name} (Section {p_idx + 1}): {title[:60]}",
                    "source_url": f"file://local/{file_path.name}",
                    "retrieval_timestamp": timestamp,
                    "excerpt": paragraph[:600],
                    "source_type": "document",
                    "relevance_score": score
                })

    # Sort descending by relevance score
    matches.sort(key=lambda x: x["relevance_score"], reverse=True)
    return matches[:top_k]
