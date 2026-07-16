"""Extract short keyword phrases from long scene descriptions for Pexels search.

Standalone utility — does not depend on any model or API.
"""
import re

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "that", "which", "who",
    "whom", "what", "this", "that", "these", "those", "it", "its",
}


def extract_keywords(text, max_words=5):
    """Turn a long scene/world description into ~5 search keywords for Pexels."""
    if not text:
        return ""

    text = re.sub(r"^(World:|Characters\s*—).*?(?=\.\s|$)", "", text)
    text = re.sub(r"[^A-Za-z0-9\s-]", " ", text)

    words = re.findall(r"[A-Za-z][A-Za-z-]+", text)
    words = [w.lower() for w in words if w.lower() not in STOP_WORDS and len(w) > 2]

    seen = set()
    unique = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    return " ".join(unique[:max_words])
