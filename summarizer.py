"""
summarizer.py
-------------
Generates an "extractive" summary of the lecture text.

"Extractive" means we do NOT generate new sentences (like ChatGPT
would). Instead, we pick the most important EXISTING sentences from
the original text using TF-IDF scoring. This is simple, fast, runs
fully offline, and needs no API keys or heavy AI models.

How TF-IDF scoring works here (in plain English):
1. Treat every sentence as its own mini "document".
2. TF-IDF gives a high score to words that are frequent in a sentence
   but NOT overly common across all sentences (so it favors
   meaningful/topic words over generic ones).
3. We add up the TF-IDF scores of the words in each sentence to get
   an overall "importance score" for that sentence.
4. We keep the top-scoring sentences and put them back in their
   original order, so the summary still reads naturally.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from text_utils import clean_text, split_into_sentences, ensure_nltk_data


def generate_summary(raw_text, min_sentences=5, max_sentences=10):
    """
    Generate a bullet-point extractive summary of the given text.

    Parameters
    ----------
    raw_text : str
        The full text extracted from the PDF.
    min_sentences, max_sentences : int
        We aim to return somewhere between these two numbers of
        bullet points, depending on how long the lecture is.

    Returns
    -------
    list[str]
        A list of sentences to display as bullet points.
    """
    ensure_nltk_data()

    cleaned = clean_text(raw_text)
    sentences = split_into_sentences(cleaned)

    if len(sentences) == 0:
        return ["Not enough readable text was found to build a summary."]

    # If the lecture is very short, just return everything we have
    if len(sentences) <= min_sentences:
        return sentences

    # Decide how many sentences to keep: enough to cover the content,
    # but capped at max_sentences so the summary stays "concise".
    num_sentences_to_keep = min(max_sentences, max(min_sentences, len(sentences) // 8))
    num_sentences_to_keep = min(num_sentences_to_keep, len(sentences))

    # Build a TF-IDF matrix where each row = one sentence
    stop_words = stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Score each sentence by summing its TF-IDF weights across all words
    sentence_scores = tfidf_matrix.sum(axis=1).A1  # .A1 flattens to a 1D array

    # Get the indices of the top N sentences, sorted by score (highest first)
    top_indices = sentence_scores.argsort()[::-1][:num_sentences_to_keep]

    # Put the selected sentences BACK in their original reading order
    top_indices_in_order = sorted(top_indices)

    summary_sentences = [sentences[i] for i in top_indices_in_order]

    return summary_sentences
