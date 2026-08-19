# AI Resume Analyzer

AI Resume Analyzer is an ATS-style resume analysis tool built with Python and Streamlit.

It compares a resume with a job description and provides:

- ATS Match Score
- Content Similarity
- Skill Match Analysis
- Missing Job Skills
- Resume Section Check
- Job Keyword Analysis
- Resume Improvement Suggestions

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- TF-IDF
- NLP
- PyPDF
- python-docx

## Features

- Upload PDF or DOCX resume
- Paste any job description
- Analyze resume-job compatibility
- Detect matched and missing skills
- Check important resume sections
- Generate ATS-style recommendations
- Clean responsive interface

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py