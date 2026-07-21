"""
app.py
------
Main Streamlit application for "AI Study Assistant".

Run this file with:
    streamlit run app.py

What it does:
1. Lets the student upload a PDF lecture file.
2. Extracts the text from the PDF (pdf_utils.py).
3. Generates a short bullet-point summary (summarizer.py).
4. Generates study flashcards (flashcards.py).
5. Displays everything in a clean, simple interface.

No API keys, no internet connection, and no external AI services are
used - everything runs locally with classic NLP / machine learning
techniques (TF-IDF + scikit-learn + NLTK).
"""

import streamlit as st

from pdf_utils import extract_text_from_pdf
from summarizer import generate_summary
from flashcards import generate_flashcards


def inject_custom_css():
    """Inject custom CSS for enhanced dark mode styling."""
    st.markdown(
        """
        <style>
        /* Main app background */
        .stApp {
            background-color: #0E1117;
        }
        /* Card styling for expanders (flashcards) */
        .streamlit-expanderHeader {
            background-color: #1E1E2E !important;
            border: 1px solid #333 !important;
            border-radius: 8px !important;
            color: #E0E0E0 !important;
        }
        .streamlit-expanderContent {
            background-color: #161622 !important;
            border: 1px solid #333 !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px !important;
            color: #B0B0B0 !important;
        }
        /* Info/Success/Warning boxes */
        .stAlert {
            background-color: #1E1E2E !important;
            border: 1px solid #333 !important;
            color: #E0E0E0 !important;
        }
        /* File uploader */
        .stFileUploader {
            background-color: #1E1E2E !important;
            border: 1px dashed #4FC3F7 !important;
            border-radius: 8px !important;
            padding: 10px;
        }
        /* Buttons */
        .stButton button {
            background-color: #4FC3F7 !important;
            color: #0E1117 !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
        }
        .stButton button:hover {
            background-color: #29B6F6 !important;
        }
        /* Text input / text area */
        .stTextArea textarea {
            background-color: #1E1E2E !important;
            color: #E0E0E0 !important;
            border: 1px solid #333 !important;
        }
        /* Headers */
        h1, h2, h3 {
            color: #4FC3F7 !important;
        }
        /* Spinner */
        .stSpinner > div {
            border-top-color: #4FC3F7 !important;
        }
        /* Success message */
        .st-emotion-cache-1dp5vir {
            background-color: #1E1E2E !important;
            color: #4FC3F7 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Display the app title and a short description."""
    st.set_page_config(page_title="AI Study Assistant", page_icon="📚", layout="centered")
    inject_custom_css()
    st.title("📚 AI Study Assistant")
    st.write(
        "Upload a PDF lecture and get an instant **summary** and a set of "
        "**flashcards** to help you study — all generated locally, no API keys needed."
    )


def render_upload_section():
    """
    Display the PDF upload widget.

    Returns
    -------
    The uploaded file object, or None if nothing has been uploaded yet.
    """
    st.header("1️⃣ Upload your lecture PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    return uploaded_file


def render_summary_section(summary_sentences):
    """Display the summary as a bulleted list."""
    st.header("2️⃣ Lecture Summary")
    if not summary_sentences:
        st.info("No summary could be generated for this document.")
        return

    for sentence in summary_sentences:
        st.markdown(f"- {sentence}")


def render_flashcards_section(flashcards):
    """Display flashcards as expandable Term/Definition cards."""
    st.header("3️⃣ Flashcards")

    if not flashcards:
        st.info("No flashcards could be generated for this document.")
        return

    st.write(f"Generated **{len(flashcards)} flashcards**. Click each term to reveal its definition.")

    for index, card in enumerate(flashcards, start=1):
        with st.expander(f"Card {index}: {card['term']}"):
            st.markdown(f"**Definition:** {card['definition']}")


def main():
    render_header()
    uploaded_file = render_upload_section()

    if uploaded_file is None:
        st.info("👆 Upload a PDF to get started.")
        return

    # Step 1: Extract text from the PDF, handling errors gracefully
    with st.spinner("Extracting text from PDF..."):
        try:
            raw_text = extract_text_from_pdf(uploaded_file)
        except ValueError as error:
            st.error(f"⚠️ {error}")
            return

    st.success("Text extracted successfully!")

    # Step 2: Generate the summary
    with st.spinner("Generating summary..."):
        summary_sentences = generate_summary(raw_text, min_sentences=5, max_sentences=10)

    render_summary_section(summary_sentences)

    # Step 3: Generate flashcards
    with st.spinner("Generating flashcards..."):
        flashcards = generate_flashcards(raw_text, min_cards=10)

    render_flashcards_section(flashcards)

    # Optional: let the student see the raw extracted text for debugging/study
    with st.expander("📄 Show full extracted text"):
        st.text_area("Extracted text", raw_text, height=300)


if __name__ == "__main__":
    main()
