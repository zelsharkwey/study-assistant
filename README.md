# 📚 AI Study Assistant

A simple, fully local Streamlit app that turns a PDF lecture into:

1. A concise bullet-point **summary**.
2. A set of **flashcards** (Question / Answer) for studying.

No API keys. No internet connection required after setup. No LangChain,
vector databases, OpenAI, or Ollama — just Python, Streamlit, PyMuPDF,
NLTK, and scikit-learn.

## Project structure

```
ai_study_assistant/
├── app.py              # Streamlit UI - the file you run
├── pdf_utils.py         # Extracts text from the uploaded PDF
├── text_utils.py        # Shared text-cleaning + sentence-splitting helpers
├── summarizer.py         # TF-IDF based extractive summary
├── flashcards.py         # Concept/keyword detection + flashcard generation
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## How it works (high level)

- **PDF extraction**: `pdf_utils.py` uses PyMuPDF (`fitz`) to pull raw text
  out of the uploaded PDF, page by page.
- **Cleaning**: `text_utils.py` reconstructs proper paragraphs from the
  wrapped lines PDFs produce, and splits everything into sentences with
  NLTK's sentence tokenizer.
- **Summary**: `summarizer.py` scores every sentence with TF-IDF
  (`scikit-learn`'s `TfidfVectorizer`) and keeps the highest-scoring
  sentences (5–10), in their original order, as bullet points.
- **Flashcards**: `flashcards.py` finds "concepts" two ways — short
  heading-like lines in the PDF (e.g. "Neural Networks") and important
  keywords/phrases found via TF-IDF over the whole document — then
  matches each concept to the most informative sentence containing it
  to use as the answer.

Everything runs with classic NLP/ML techniques on your own machine —
no calls to any external AI service.

## Setup instructions

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The first time you run the app, it will automatically download the small
NLTK data files it needs (`punkt`, `punkt_tab`, `stopwords`). This
requires an internet connection once; after that, everything runs
offline.

### 3. Run the app

```bash
streamlit run app.py
```

Streamlit will open the app in your browser (usually at
`http://localhost:8501`).

### 4. Use it

1. Upload a PDF lecture file.
2. Wait a few seconds while the text is extracted and processed.
3. Read the generated summary bullet points.
4. Click each flashcard to reveal its answer.

## Notes & limitations

- Works best on PDFs with real, selectable text (not scanned images).
  Scanned/image-only PDFs would need OCR, which isn't included here.
- Flashcard quality depends on how the lecture is formatted — PDFs with
  clear headings tend to produce better flashcards than plain, unbroken
  paragraphs.
- This uses simple, transparent statistical methods (TF-IDF), not a
  large language model, so summaries are extractive (made of real
  sentences from the text) rather than freely rewritten.

## Future improvements

- **OCR support** for scanned PDFs (e.g. using `pytesseract`).
- **Difficulty levels** for flashcards (basic recall vs. deeper "why"
  questions).
- **Export options**: download flashcards as CSV/Anki deck, or the
  summary as a `.txt`/`.pdf` file.
- **Multi-file support**: upload a whole folder of lecture PDFs and get
  one combined study set.
- **Better concept detection** using part-of-speech tagging to pick out
  noun phrases more accurately than the current heading/keyword
  heuristics.
- **Quiz mode**: turn flashcards into a self-testing quiz with
  score tracking, right inside Streamlit.
- **Local embeddings-based semantic search** (still no external APIs)
  to let students ask questions about the lecture content.
