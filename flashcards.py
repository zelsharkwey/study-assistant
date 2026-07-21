"""
flashcards.py
-------------
Generates study flashcards (Question / Answer pairs) from the
lecture text, without any external AI model.

The approach, in plain English:
1. Find "candidate concepts" in two ways:
   a. Headings - short standalone lines in the original PDF text
      (e.g. "Neural Networks", "Chapter 2: Data Structures").
   b. Keywords - important words/phrases found using TF-IDF over the
      whole document (this highlights terms that matter a lot in
      THIS lecture specifically, not generic English words).
2. For each concept, find the best matching sentence in the text to
   use as the answer/definition (the sentence that mentions the
   concept and is the most informative).
3. Build flashcards in the format:
       Term: <concept>
       Definition: <sentence that explains it>
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from text_utils import clean_text, split_into_sentences, split_into_lines, ensure_nltk_data


def _looks_like_heading(line):
    """
    Heuristic check: does this line look like a heading/title rather
    than a normal sentence?

    Headings are usually:
    - Short (a handful of words).
    - Do NOT end with a period (sentences usually do).
    - Mostly made of letters (not just numbers or symbols).
    """
    words = line.split()

    if not (1 <= len(words) <= 6):
        return False

    if line.endswith((".", ",", ";")):
        return False

    if not re.search(r"[A-Za-z]", line):
        return False

    # Skip lines that are clearly page numbers, bullet symbols, etc.
    if re.fullmatch(r"[\d\W]+", line):
        return False

    return True


def _extract_headings(raw_text, max_headings=10):
    """
    Scan the ORIGINAL text (before cleaning) line by line and pull out
    lines that look like headings or standalone key terms.
    """
    lines = split_into_lines(raw_text)
    headings = []

    for line in lines:
        if _looks_like_heading(line):
            headings.append(line.strip())

    # Remove duplicates while keeping the first-seen order
    seen = set()
    unique_headings = []
    for heading in headings:
        key = heading.lower()
        if key not in seen:
            seen.add(key)
            unique_headings.append(heading)

    return unique_headings[:max_headings]


def _extract_keywords(cleaned_text, max_keywords=15):
    """
    Use TF-IDF over single words and two-word phrases (bigrams) to
    find the most important keywords/phrases in the lecture.
    """
    ensure_nltk_data()
    stop_words = stopwords.words("english")

    # ngram_range=(1, 2) means we look at both single words ("gradient")
    # and two-word phrases ("gradient descent")
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        max_features=200,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    except ValueError:
        # Happens if the text is too short/empty for TF-IDF
        return []

    scores = tfidf_matrix.toarray()[0]
    terms = vectorizer.get_feature_names_out()

    # Pair each term with its score, then sort highest first
    term_scores = list(zip(terms, scores))
    term_scores.sort(key=lambda pair: pair[1], reverse=True)

    keywords = [term for term, score in term_scores[:max_keywords] if score > 0]
    return keywords


def _find_best_sentence(concept, sentences, used_sentences):
    """
    Find the sentence that best explains a given concept:
    - Must contain the concept text (case-insensitive).
    - Prefer longer, more detailed sentences (likely to be definitions).
    - Avoid reusing a sentence that's already been used for another card.
    """
    concept_lower = concept.lower()
    matching_sentences = [
        sentence for sentence in sentences
        if concept_lower in sentence.lower() and sentence not in used_sentences
    ]

    if not matching_sentences:
        return None

    # Prefer the sentence with the most words (usually more explanatory)
    best_sentence = max(matching_sentences, key=lambda s: len(s.split()))
    return best_sentence


def generate_flashcards(raw_text, min_cards=10):
    """
    Generate a list of flashcards from the lecture text.

    Returns
    -------
    list[dict]
        Each dict has the shape: {"term": str, "definition": str}
    """
    cleaned = clean_text(raw_text)
    sentences = split_into_sentences(cleaned)

    if not sentences:
        return []

    headings = _extract_headings(raw_text)
    keywords = _extract_keywords(cleaned)

    # Combine headings first (usually higher quality concepts), then
    # keywords, removing duplicates along the way.
    all_concepts = []
    seen = set()
    for concept in headings + keywords:
        key = concept.lower().strip()
        if key and key not in seen and len(key) > 2:
            seen.add(key)
            all_concepts.append(concept.strip())

    flashcards = []
    used_sentences = set()

    for concept in all_concepts:
        answer = _find_best_sentence(concept, sentences, used_sentences)
        if answer is None:
            continue

        used_sentences.add(answer)
        flashcards.append({
            "term": concept,
            "definition": answer,
        })

        # Stop once we have a generous number of flashcards so the
        # student isn't overwhelmed.
        if len(flashcards) >= max(min_cards, 15):
            break

    return flashcards
