# NDA Clause Comparison Tool
A Streamlit-powered tool to compare clauses in NDAs (Non-Disclosure Agreements) by identifying semantic differences, deviations, and missing clauses. This tool leverages NLP and generative AI to perform clause comparisons with an emphasis on legally significant differences.

## Features
PDF and DOCX Support: Upload NDA documents in PDF or DOCX format.
Automated Clause Extraction: Extracts the introduction, clauses, and signatures from both a BRP NDA Template and an uploaded NDA.
Semantic Clause Comparison: Uses Sentence-BERT to find the best matching clauses and measures similarity.
Deviation Analysis: Identifies critical legal deviations using Google’s Gemini AI.
Detailed DOCX Report: Generates a DOCX document highlighting differences, including sections for further human review and key suggestions.
Interactive UI: Streamlit-based UI for easy file uploads, clause review, and download of comparison results.

## Usage
Upload Files:

Upload the BRP NDA Template and the NDA for Comparison in either PDF or DOCX format.
Extract and Display Text:

The app will display extracted text for verification.
Review Comparison Results:

The tool compares each clause and highlights deviations.
Mark clauses for further human review if needed.
Download DOCX Report:

After reviewing, download the generated DOCX report, which includes detailed deviations and suggestions.
Key Functions
load_pdf_with_fitz: Loads text from PDF files and handles common formatting issues like hyphenation.
extract_intro_clauses_signatures: Extracts the introduction, clauses, and signatures from NDA text.
find_best_match: Finds the best matching clause using Sentence-BERT embeddings.
analyze_with_gemini: Uses Google’s Gemini model to analyze semantic deviations between clauses.
generate_highlighted_docx: Generates a DOCX file with deviations and review suggestions, highlighted for easy review.

## Dependencies and Models
Sentence-BERT: Pretrained model for semantic similarity using all-MiniLM-L6-v2.
Google Gemini API: Used for deviation analysis, emphasizing legally significant differences.
## Troubleshooting
Missing API Key: Ensure your .env file has the correct GENAI_API_KEY.
Quota Limits: If you encounter "API Quota exhausted" messages, consider using a different API key or adjusting usage.
Missing NLTK Data: If NLTK fails to tokenize, run nltk.download('punkt').

## Future Enhancements
Persistent Checkbox States: Use st.session_state in Streamlit for persistent clause review across interactions.
Adjustable Similarity Threshold: Allow users to set their preferred similarity threshold.
Multi-Document Comparison: Support for batch comparisons with multiple NDA documents.

## License
This project is licensed under Balanced Rock Power.
