import re
from collections import Counter

import streamlit as st
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(124,58,237,0.16), transparent 25%),
                radial-gradient(circle at 90% 10%, rgba(6,182,212,0.12), transparent 25%),
                #07111f;
        }

        .main-title {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0;
        }

        .subtitle {
            color: #94a3b8;
            font-size: 1.05rem;
            margin-bottom: 25px;
        }

        .score-card {
            padding: 25px;
            border-radius: 18px;
            background: rgba(17, 24, 39, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.16);
            text-align: center;
            min-height: 150px;
        }

        .score-number {
            font-size: 3.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #a78bfa, #22d3ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .small-muted {
            color: #94a3b8;
            font-size: 0.9rem;
        }

        .section-card {
            padding: 20px;
            border-radius: 15px;
            background: rgba(17, 24, 39, 0.60);
            border: 1px solid rgba(148, 163, 184, 0.14);
            margin-bottom: 15px;
        }

        .skill-chip {
            display: inline-block;
            padding: 6px 11px;
            margin: 4px;
            border-radius: 999px;
            background: rgba(124, 58, 237, 0.15);
            color: #c4b5fd;
            border: 1px solid rgba(167,139,250,0.20);
            font-size: 0.82rem;
        }

        .missing-chip {
            display: inline-block;
            padding: 6px 11px;
            margin: 4px;
            border-radius: 999px;
            background: rgba(239,68,68,0.10);
            color: #fca5a5;
            border: 1px solid rgba(239,68,68,0.18);
            font-size: 0.82rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SKILL DATABASE
# =========================================================

SKILLS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "c",
    "c++",
    "c#",
    "html",
    "css",
    "react",
    "node.js",
    "node",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "git",
    "github",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "linux",
    "api",
    "rest api",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "generative ai",
    "llm",
    "large language models",
    "data science",
    "data analysis",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "nlp",
    "natural language processing",
    "computer vision",
    "ai agents",
    "automation",
    "n8n",
    "prompt engineering",
    "data structures",
    "algorithms",
    "dsa",
    "oop",
    "object oriented programming",
    "problem solving",
    "communication",
    "leadership",
    "teamwork",
    "ui/ux",
    "figma",
    "streamlit",
]


REQUIRED_SECTIONS = {
    "Summary / Objective": [
        "summary",
        "objective",
        "profile",
        "professional summary",
    ],
    "Skills": [
        "skills",
        "technical skills",
        "technologies",
        "tech stack",
    ],
    "Education": [
        "education",
        "academic",
        "university",
        "college",
    ],
    "Projects": [
        "projects",
        "project",
    ],
    "Experience": [
        "experience",
        "internship",
        "employment",
        "work experience",
    ],
}


# =========================================================
# FILE EXTRACTION
# =========================================================

def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_docx_text(file):
    document = Document(file)
    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


def extract_text(file):
    filename = file.name.lower()

    if filename.endswith(".pdf"):
        return extract_pdf_text(file)

    if filename.endswith(".docx"):
        return extract_docx_text(file)

    return ""


# =========================================================
# TEXT PROCESSING
# =========================================================

def normalize_text(text):
    text = text.lower()
    text = text.replace("node js", "node.js")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    return re.findall(r"\b[a-zA-Z][a-zA-Z+#.\-/]*\b", text.lower())


# =========================================================
# SIMILARITY SCORE
# =========================================================

def calculate_similarity(resume_text, job_description):
    documents = [
        normalize_text(resume_text),
        normalize_text(job_description),
    ]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        )

        matrix = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:2],
        )[0][0]

        return round(similarity * 100)

    except ValueError:
        return 0


# =========================================================
# SKILL MATCHING
# =========================================================

def detect_skills(text):
    clean_text = normalize_text(text)
    detected = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, clean_text):
            detected.append(skill)

    return sorted(set(detected))


def compare_skills(resume_text, job_description):
    resume_skills = set(detect_skills(resume_text))
    job_skills = set(detect_skills(job_description))

    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)

    if not job_skills:
        skill_score = 0

    else:
        skill_score = round(
            len(matched) / len(job_skills) * 100
        )

    return (
        sorted(resume_skills),
        sorted(job_skills),
        matched,
        missing,
        skill_score,
    )


# =========================================================
# SECTION CHECK
# =========================================================

def check_resume_sections(resume_text):
    text = normalize_text(resume_text)

    results = {}

    for section_name, keywords in REQUIRED_SECTIONS.items():
        found = any(
            keyword in text
            for keyword in keywords
        )

        results[section_name] = found

    return results


# =========================================================
# KEYWORD ANALYSIS
# =========================================================

def important_job_keywords(job_description, limit=20):
    words = tokenize(job_description)

    stop_words = {
        "the",
        "and",
        "with",
        "for",
        "you",
        "your",
        "are",
        "our",
        "this",
        "that",
        "will",
        "have",
        "has",
        "from",
        "into",
        "job",
        "role",
        "candidate",
        "experience",
        "knowledge",
        "skills",
        "looking",
        "work",
        "working",
        "plus",
    }

    filtered_words = [
        word
        for word in words
        if len(word) > 2
        and word not in stop_words
    ]

    counts = Counter(filtered_words)

    return [
        word
        for word, _ in counts.most_common(limit)
    ]


def keyword_coverage(resume_text, job_description):
    resume = normalize_text(resume_text)

    keywords = important_job_keywords(
        job_description
    )

    matched = []
    missing = []

    for keyword in keywords:
        if keyword in resume:
            matched.append(keyword)
        else:
            missing.append(keyword)

    if not keywords:
        coverage = 0
    else:
        coverage = round(
            len(matched) / len(keywords) * 100
        )

    return matched, missing, coverage


# =========================================================
# ATS SCORE
# =========================================================

def calculate_ats_score(
    similarity_score,
    skill_score,
    keyword_score,
    section_results,
):
    section_score = round(
        sum(section_results.values())
        / len(section_results)
        * 100
    )

    final_score = (
        similarity_score * 0.35
        + skill_score * 0.35
        + keyword_score * 0.20
        + section_score * 0.10
    )

    return round(final_score), section_score


# =========================================================
# RECOMMENDATIONS
# =========================================================

def generate_recommendations(
    ats_score,
    missing_skills,
    section_results,
    resume_text,
):
    recommendations = []

    if ats_score < 50:
        recommendations.append(
            "Tailor your resume more closely to the specific job description."
        )

    elif ats_score < 70:
        recommendations.append(
            "Your resume has a reasonable match, but stronger job-specific wording could improve it."
        )

    else:
        recommendations.append(
            "Your resume is well aligned with this job description."
        )

    if missing_skills:
        recommendations.append(
            "Consider highlighting relevant experience with these job skills: "
            + ", ".join(missing_skills[:8])
            + "."
        )

    for section, found in section_results.items():
        if not found:
            recommendations.append(
                f"Consider adding or clearly labeling a {section} section."
            )

    word_count = len(resume_text.split())

    if word_count < 250:
        recommendations.append(
            "Your resume may be too short. Add measurable project or experience details where relevant."
        )

    elif word_count > 900:
        recommendations.append(
            "Your resume may be too long. Keep content concise and prioritize relevant achievements."
        )

    if "%" not in resume_text and not re.search(
        r"\b\d+\+?\b",
        resume_text,
    ):
        recommendations.append(
            "Where truthful, quantify achievements using numbers, percentages or measurable outcomes."
        )

    return recommendations


# =========================================================
# SCORE LABEL
# =========================================================

def get_score_label(score):
    if score >= 80:
        return "Excellent Match", "success"

    if score >= 65:
        return "Good Match", "success"

    if score >= 50:
        return "Moderate Match", "warning"

    return "Needs Improvement", "error"


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📄 AI Resume Analyzer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        ATS-style resume analysis for job matching, skills,
        keywords and improvement suggestions.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Analyzer")

    st.write(
        "Upload a resume and paste a job description "
        "to analyze their compatibility."
    )

    st.info(
        "Supported resume formats: PDF and DOCX"
    )

    st.caption(
        "This tool provides an estimated ATS-style score. "
        "It does not represent an actual employer ATS system."
    )


# =========================================================
# INPUT SECTION
# =========================================================

left, right = st.columns(2)

with left:
    st.subheader("1️⃣ Upload Resume")

    uploaded_file = st.file_uploader(
        "Choose your resume",
        type=["pdf", "docx"],
    )


with right:
    st.subheader("2️⃣ Job Description")

    job_description = st.text_area(
        "Paste the complete job description",
        height=220,
        placeholder=(
            "Example: We are looking for an AI/ML Intern "
            "with Python, Machine Learning, Git..."
        ),
    )


# =========================================================
# ANALYZE
# =========================================================

st.write("")

analyze_button = st.button(
    "🚀 Analyze Resume",
    use_container_width=True,
    type="primary",
)


if analyze_button:

    if uploaded_file is None:
        st.error(
            "Please upload a PDF or DOCX resume."
        )

    elif not job_description.strip():
        st.error(
            "Please paste the job description."
        )

    else:

        with st.spinner(
            "Analyzing resume..."
        ):

            try:
                resume_text = extract_text(
                    uploaded_file
                )

            except Exception as error:
                st.error(
                    f"Could not read the resume: {error}"
                )
                st.stop()

            if not resume_text.strip():
                st.error(
                    "No readable text was found in the uploaded resume."
                )
                st.stop()

            similarity_score = calculate_similarity(
                resume_text,
                job_description,
            )

            (
                resume_skills,
                job_skills,
                matched_skills,
                missing_skills,
                skill_score,
            ) = compare_skills(
                resume_text,
                job_description,
            )

            section_results = check_resume_sections(
                resume_text
            )

            (
                matched_keywords,
                missing_keywords,
                keyword_score,
            ) = keyword_coverage(
                resume_text,
                job_description,
            )

            ats_score, section_score = calculate_ats_score(
                similarity_score,
                skill_score,
                keyword_score,
                section_results,
            )

            recommendations = generate_recommendations(
                ats_score,
                missing_skills,
                section_results,
                resume_text,
            )


        # =================================================
        # MAIN RESULT
        # =================================================

        st.divider()

        st.subheader("📊 Resume Analysis")

        score_label, score_type = get_score_label(
            ats_score
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-number">
                        {ats_score}%
                    </div>
                    <strong>ATS Match Score</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.metric(
                "Content Similarity",
                f"{similarity_score}%",
            )

        with col3:
            st.metric(
                "Skill Match",
                f"{skill_score}%",
            )

        with col4:
            st.metric(
                "Resume Sections",
                f"{section_score}%",
            )


        if score_type == "success":
            st.success(
                f"✅ {score_label}"
            )

        elif score_type == "warning":
            st.warning(
                f"⚠️ {score_label}"
            )

        else:
            st.error(
                f"❌ {score_label}"
            )


        # =================================================
        # SKILL ANALYSIS
        # =================================================

        st.subheader(
            "🎯 Skill Match Analysis"
        )

        skill_col1, skill_col2 = st.columns(2)

        with skill_col1:

            st.markdown(
                "#### ✅ Matched Skills"
            )

            if matched_skills:
                matched_html = "".join(
                    f'<span class="skill-chip">{skill.title()}</span>'
                    for skill in matched_skills
                )

                st.markdown(
                    matched_html,
                    unsafe_allow_html=True,
                )

            else:
                st.info(
                    "No predefined job skills were matched."
                )


        with skill_col2:

            st.markdown(
                "#### ❌ Missing Job Skills"
            )

            if missing_skills:
                missing_html = "".join(
                    f'<span class="missing-chip">{skill.title()}</span>'
                    for skill in missing_skills
                )

                st.markdown(
                    missing_html,
                    unsafe_allow_html=True,
                )

            else:
                st.success(
                    "No predefined missing job skills detected."
                )


        # =================================================
        # SECTION CHECK
        # =================================================

        st.subheader(
            "📋 Resume Section Check"
        )

        section_columns = st.columns(
            len(section_results)
        )

        for column, (
            section_name,
            found,
        ) in zip(
            section_columns,
            section_results.items(),
        ):

            with column:

                if found:
                    st.success(
                        f"✅ {section_name}"
                    )

                else:
                    st.warning(
                        f"⚠️ {section_name}"
                    )


        # =================================================
        # KEYWORD ANALYSIS
        # =================================================

        st.subheader(
            "🔎 Job Keyword Analysis"
        )

        keyword_left, keyword_right = st.columns(2)

        with keyword_left:
            st.markdown(
                "#### Keywords Found"
            )

            if matched_keywords:
                st.write(
                    ", ".join(
                        matched_keywords
                    )
                )

            else:
                st.write(
                    "No major keywords detected."
                )


        with keyword_right:
            st.markdown(
                "#### Keywords To Review"
            )

            if missing_keywords:
                st.write(
                    ", ".join(
                        missing_keywords[:15]
                    )
                )

            else:
                st.write(
                    "Strong keyword coverage."
                )


        # =================================================
        # RECOMMENDATIONS
        # =================================================

        st.subheader(
            "💡 Resume Improvement Suggestions"
        )

        for index, recommendation in enumerate(
            recommendations,
            start=1,
        ):
            st.write(
                f"{index}. {recommendation}"
            )


        # =================================================
        # DETECTED RESUME SKILLS
        # =================================================

        with st.expander(
            "🛠️ Skills Detected In Resume"
        ):

            if resume_skills:
                st.write(
                    ", ".join(
                        skill.title()
                        for skill in resume_skills
                    )
                )

            else:
                st.write(
                    "No predefined technical skills detected."
                )


        # =================================================
        # RESUME TEXT
        # =================================================

        with st.expander(
            "📄 View Extracted Resume Text"
        ):

            st.text(
                resume_text[:12000]
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Resume Analyzer V2 • Python • Streamlit • NLP • TF-IDF • Scikit-learn"
)