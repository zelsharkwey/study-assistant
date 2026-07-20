"""
text_utils.py
-------------
Shared text-cleaning and sentence-splitting helpers used by both the
summarizer and the flashcard generator. Keeping these in one place
avoids duplicating logic.
"""

import re
import nltk


def ensure_nltk_data():
    """
    Make sure the NLTK data files we need (sentence tokenizer and
    stopwords list) are downloaded. This runs once and is safe to
    call every time the app starts - it will skip downloading if the
    data already exists.
    """
    required_packages = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]

    for lookup_path, package_name in required_packages:
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            nltk.download(package_name, quiet=True)


def _is_short_standalone_line(line, next_line):
    """
    Heuristic: is this line likely a HEADING rather than a wrapped
    continuation of a paragraph? Headings tend to be short and are
    NOT immediately followed by a lowercase continuation on the next
    line (a wrapped sentence line often starts lowercase, or the
    heading is visually separated by a blank line already).
    """
    words = line.split()
    if not (1 <= len(words) <= 6):
        return False
    if line.endswith((".", ",", ";", "-")):
        return False
    return True


def clean_text(raw_text):
    """
    Do some light cleanup on text extracted from a PDF and reconstruct
    proper paragraphs / sentences out of PDF line breaks.

    PDFs store text as a series of short lines that wrap at the page
    width - a single sentence is usually split across several lines
    with NO punctuation in between. If we naively added a period after
    every line, wrapped sentences would get chopped into meaningless
    fragments. Short standalone lines (like headings, e.g. "Neural
    Networks") are the exception - those really should end their own
    "sentence" so they don't glue onto the next paragraph.

    Strategy:
    1. Split the text into blank-line-separated blocks (paragraphs).
    2. If a block is just ONE short line with no ending punctuation,
       treat it as a heading and give it its own period.
    3. Otherwise, join all the lines in the block with spaces (they
       are wrapped pieces of the same paragraph) and make sure the
       whole paragraph ends with a period.
    """
    # Remove page-number-only lines up front
    raw_lines = raw_text.split("\n")
    filtered_lines = [
        line for line in raw_lines
        if not re.fullmatch(r"\s*\d{1,4}\s*", line)
    ]
    text_with_breaks = "\n".join(filtered_lines)

    # Split into paragraphs on one-or-more blank lines
    paragraphs = re.split(r"\n\s*\n", text_with_breaks)

    fixed_paragraphs = []
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        if not lines:
            continue

        if len(lines) == 1 and _is_short_standalone_line(lines[0], None):
            # Looks like a heading - keep it as its own short sentence
            sentence = lines[0]
        else:
            # Wrapped paragraph text - join lines back into one sentence/paragraph
            sentence = " ".join(lines)

        if not re.search(r"[.!?:;]$", sentence):
            sentence += "."

        fixed_paragraphs.append(sentence)

    text = " ".join(fixed_paragraphs)

    # Collapse multiple spaces/tabs into one space
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def split_into_sentences(text):
    """
    Split a block of text into a list of individual sentences using
    NLTK's sentence tokenizer.
    """
    ensure_nltk_data()
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)

    # Filter out very short "sentences" (junk like "Fig. 2" leftovers)
    clean_sentences = [
        sentence.strip() for sentence in sentences
        if len(sentence.strip().split()) >= 4
    ]

    return clean_sentences


def split_into_lines(raw_text):
    """
    Split the ORIGINAL (uncleaned) text into lines. This is useful for
    detecting headings, since headings are usually short lines on
    their own - information that gets lost once we join everything
    into paragraphs.
    """
    lines = [line.strip() for line in raw_text.split("\n")]
    return [line for line in lines if line]
