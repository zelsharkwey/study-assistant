"""
pdf_utils.py
------------
Beginner-friendly helper functions for working with PDF files.

This module is responsible for ONE job only: taking an uploaded PDF
and turning it into plain text that the rest of the app can use.
"""

import fitz  # This is the import name for PyMuPDF


def extract_text_from_pdf(uploaded_file):
    """
    Extract all text from an uploaded PDF file.

    Parameters
    ----------
    uploaded_file : a file-like object coming from Streamlit's
        st.file_uploader (it behaves like a normal Python file).

    Returns
    -------
    str
        All the text found in the PDF, with pages separated by
        a newline. Returns an empty string if nothing could be read.

    Raises
    ------
    ValueError
        If the file is not a valid / readable PDF, we raise a clear
        error so the Streamlit app can show a friendly message
        instead of crashing.
    """
    try:
        # Read the raw bytes from the uploaded file
        pdf_bytes = uploaded_file.read()

        # Open the PDF directly from memory (no need to save to disk)
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        extracted_pages = []
        for page in pdf_document:
            # get_text("text") returns plain readable text for the page
            page_text = page.get_text("text")
            if page_text:
                extracted_pages.append(page_text)

        pdf_document.close()

        full_text = "\n".join(extracted_pages)

        if not full_text.strip():
            # The PDF opened fine but had no extractable text
            # (this can happen with scanned/image-only PDFs)
            raise ValueError(
                "No readable text was found in this PDF. "
                "It might be a scanned document (images only)."
            )

        return full_text

    except ValueError:
        # Re-raise our own, already-friendly error message
        raise
    except Exception as error:
        # Catch anything unexpected (corrupted file, wrong format, etc.)
        raise ValueError(
            f"Could not read this PDF file. Details: {error}"
        )
