import json
import html
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.loader import load_document
from src.splitter import split_document
from src.vectorstore import create_vectorstore
from src.chain import (
    answer_question,
    detect_document_type,
    get_suggested_questions,
    summarize_document_for_home,
)

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
PHOTO_PATH = ASSETS_DIR / "photo.jpg"
FEEDBACK_DIR = BASE_DIR / "feedback"
TEMP_DIR = BASE_DIR / "temp"

FEEDBACK_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Document Intelligence Assistant | Vamshi Kardhanoori",
    page_icon="📄",
    layout="wide",
)


def init_session():
    defaults = {
        "vectorstore": None,
        "documents": None,
        "full_text": "",
        "doc_type": "",
        "doc_summary": "",
        "suggested_questions": [],
        "uploaded_file_name": "",
        "answer": "",
        "context": "",
        "selected_question": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clean_text_for_display(text: str) -> str:
    if not text:
        return ""

    text = str(text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)
    text = text.replace("\uFFFD", "")
    text = text.replace("\u0000", "")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)

    return text.strip()


def safe_html_text(text: str) -> str:
    cleaned = clean_text_for_display(text)
    escaped = html.escape(cleaned)
    escaped = escaped.replace("\n", "<br>")
    return escaped


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(35, 95, 180, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(40, 165, 220, 0.10), transparent 24%),
                linear-gradient(135deg, #020617 0%, #03112f 48%, #020617 100%);
            color: #eaf2ff;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }

        .main-shell {
            margin-top: 0.4rem;
        }

        .hero-card,
        .section-card,
        .info-card,
        .demo-card,
        .answer-card,
        .feedback-card,
        .tech-card,
        .contact-card,
        .note-banner,
        .photo-card,
        .metric-box,
        .usage-note-card {
            background: linear-gradient(180deg, rgba(8,16,36,0.95), rgba(5,11,27,0.98));
            border: 1px solid rgba(120, 160, 255, 0.16);
            border-radius: 24px;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
        }

        .hero-card {
            padding: 34px;
            min-height: 100%;
        }

        .section-card,
        .demo-card,
        .answer-card,
        .feedback-card,
        .contact-card,
        .photo-card,
        .usage-note-card {
            padding: 28px;
            margin-bottom: 22px;
        }

        .info-card,
        .tech-card {
            padding: 22px;
            height: 100%;
            margin-bottom: 18px;
        }

        .note-banner {
            padding: 18px 22px;
            margin-bottom: 18px;
        }

        .eyebrow {
            font-size: 0.84rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #78d4ff;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .hero-name {
            font-size: 2rem;
            color: #8ad8ff;
            font-weight: 850;
            margin-bottom: 10px;
            line-height: 1.2;
        }

        .hero-role {
            font-size: 1.06rem;
            color: #dce9ff;
            font-weight: 700;
            margin-bottom: 12px;
            line-height: 1.8;
        }

        .hero-title {
            font-size: 3rem;
            line-height: 1.08;
            color: #f8fbff;
            font-weight: 860;
            margin-top: 16px;
            margin-bottom: 18px;
            letter-spacing: -0.02em;
        }

        .open-to-work {
            font-size: 1rem;
            color: #ffffff;
            font-weight: 800;
            line-height: 1.8;
            margin-top: 8px;
            margin-bottom: 18px;
            padding: 12px 14px;
            border-radius: 14px;
            background: rgba(20, 58, 112, 0.38);
            border: 1px solid rgba(110, 203, 255, 0.16);
        }

        .hero-copy {
            color: #d9e8ff;
            font-size: 1rem;
            line-height: 1.95;
            margin-bottom: 16px;
        }

        .section-title {
            font-size: 2.1rem;
            font-weight: 850;
            color: #f8fbff;
            margin-bottom: 12px;
            line-height: 1.2;
        }

        .sub-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: #f3f8ff;
            margin-bottom: 10px;
            line-height: 1.5;
        }

        .copy {
            color: #d8e7ff;
            font-size: 1rem;
            line-height: 1.95;
            margin-bottom: 14px;
        }

        .small-copy {
            color: #d8e7ff;
            font-size: 0.98rem;
            line-height: 1.9;
        }

        .usage-note-title {
            color: #ffffff;
            font-size: 1.12rem;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .usage-note-text {
            color: #dce9ff;
            font-size: 0.98rem;
            line-height: 1.9;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 18px;
        }

        .pill {
            padding: 9px 14px;
            border-radius: 999px;
            background: rgba(22, 60, 120, 0.22);
            border: 1px solid rgba(110, 203, 255, 0.20);
            color: #e5f4ff;
            font-size: 0.9rem;
            font-weight: 700;
        }

        .contact-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 14px;
            margin-top: 10px;
        }

        .contact-item {
            padding: 16px 18px;
            border-radius: 16px;
            background: rgba(10, 24, 54, 0.85);
            border: 1px solid rgba(120, 160, 255, 0.14);
        }

        .contact-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #7dd8ff;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .contact-value a {
            color: #eef6ff !important;
            font-size: 1rem;
            line-height: 1.8;
            font-weight: 700;
            text-decoration: none;
        }

        .contact-value a:hover {
            color: #8ad8ff !important;
        }

        .photo-image-wrap {
            margin-bottom: 16px;
        }

        .photo-caption {
            color: #dbe9ff;
            font-size: 0.98rem;
            line-height: 1.9;
            margin-top: 16px;
        }

        .metric-box {
            padding: 20px;
            height: 100%;
        }

        .metric-title {
            font-size: 0.88rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #78d4ff;
            font-weight: 800;
            margin-bottom: 10px;
        }

        .metric-text {
            color: #e8f1ff;
            line-height: 1.9;
            font-size: 0.97rem;
        }

        .workflow-step {
            padding: 16px 18px;
            border-radius: 16px;
            background: rgba(11, 24, 52, 0.84);
            border: 1px solid rgba(120, 160, 255, 0.14);
            margin-bottom: 12px;
            color: #dceaff;
            line-height: 1.85;
            font-size: 0.98rem;
        }

        .note-box {
            padding: 16px 18px;
            border-radius: 16px;
            background: rgba(18, 42, 84, 0.72);
            border: 1px solid rgba(110, 203, 255, 0.16);
            color: #eff7ff;
            font-weight: 700;
            margin-top: 6px;
            line-height: 1.85;
        }

        .training-note {
            color: #ffffff;
            font-size: 1rem;
            line-height: 1.85;
            font-weight: 800;
        }

        .uploaded-box {
            padding: 14px 16px;
            background: rgba(17, 34, 68, 0.78);
            border: 1px solid rgba(120, 160, 255, 0.16);
            border-radius: 14px;
            margin-top: 12px;
            color: #dceeff;
            font-weight: 700;
            line-height: 1.7;
        }

        .answer-box {
            padding: 18px;
            border-radius: 16px;
            background: rgba(11, 24, 52, 0.84);
            border: 1px solid rgba(120, 160, 255, 0.14);
            color: #eef6ff;
            line-height: 1.95;
            margin-top: 8px;
            word-break: break-word;
        }

        .context-box {
            padding: 18px;
            border-radius: 16px;
            background: rgba(8, 17, 37, 0.90);
            border: 1px solid rgba(120, 160, 255, 0.12);
            color: #d7e6ff;
            line-height: 1.85;
            white-space: normal;
            font-size: 0.96rem;
            margin-top: 8px;
            word-break: break-word;
        }

        .footer-line {
            margin-top: 18px;
            color: #b9cbe7;
            font-size: 0.95rem;
            text-align: center;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background: rgba(8, 16, 36, 0.72);
            padding: 8px;
            border-radius: 18px;
            border: 1px solid rgba(120, 160, 255, 0.14);
            margin-bottom: 24px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 54px;
            border-radius: 14px;
            padding-left: 20px;
            padding-right: 20px;
            color: #dce9ff;
            font-weight: 800;
            background: transparent;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(180deg, rgba(18,32,68,0.98), rgba(8,18,40,0.98)) !important;
            color: #ffffff !important;
        }

        .stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            border-radius: 14px;
            padding: 0.82rem 1rem;
            background: linear-gradient(180deg, rgba(18,32,68,0.98), rgba(8,18,40,0.98));
            color: #ebf5ff;
            border: 1px solid rgba(122, 178, 255, 0.22);
            font-weight: 800;
        }

        .stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            border-color: rgba(110, 203, 255, 0.45);
        }

        label, .stSelectbox label, .stTextInput label, .stTextArea label, .stFileUploader label {
            color: #e7f1ff !important;
            font-weight: 700 !important;
        }

        .stTextInput > div > div > input,
        .stTextArea textarea,
        .stSelectbox > div > div {
            border-radius: 14px !important;
        }

        .stAlert {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def save_feedback(name: str, feedback_type: str, rating: str, message: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = FEEDBACK_DIR / f"feedback_{timestamp}.json"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "name": name.strip(),
        "feedback_type": feedback_type,
        "rating": rating,
        "message": message.strip(),
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def process_uploaded_file(uploaded_file):
    file_path = TEMP_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    extension = file_path.suffix.lower()

    documents = load_document(str(file_path), extension)
    chunks = split_document(documents)
    vectorstore = create_vectorstore(chunks)

    full_text = "\n".join([clean_text_for_display(doc.page_content) for doc in documents])
    doc_type = detect_document_type(full_text)
    doc_summary = summarize_document_for_home(doc_type, full_text)
    suggested_questions = get_suggested_questions(doc_type)

    st.session_state.documents = documents
    st.session_state.vectorstore = vectorstore
    st.session_state.full_text = full_text
    st.session_state.doc_type = doc_type
    st.session_state.doc_summary = clean_text_for_display(doc_summary)
    st.session_state.suggested_questions = suggested_questions
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.answer = ""
    st.session_state.context = ""
    st.session_state.selected_question = ""


def render_home_page():
    left, right = st.columns([1.18, 0.82], gap="large")

    with left:
        st.markdown(
            """
            <div class="hero-card">
                <div class="eyebrow">Gen AI Project Portfolio</div>
                <div class="hero-name">Vamshi Kardhanoori</div>
                <div class="hero-role">Graduate Student in Information Science and Technology • Gen AI Engineer • LLM, NLP, and Applied AI Systems Builder</div>
                <div class="open-to-work">The code speaks for itself, but I’m still happy to say: yes, I’m open to work.</div>
                <div class="hero-title">Document Intelligence<br>Assistant</div>
                <div class="hero-copy">
                    This project portfolio presents a recruiter facing document intelligence assistant built to accept uploaded files, identify document type, retrieve semantically relevant context, and return grounded answers based on the uploaded content rather than generic responses.
                </div>
                <div class="hero-copy">
                    It is designed to demonstrate practical AI engineering through modular document ingestion, semantic chunking, embedding generation, vector retrieval, document aware answering logic, and a professional Streamlit interface suitable for project demonstrations, technical discussions, and recruiter review.
                </div>
                <div class="hero-copy">
                    My goal with this build was to create a strong end to end Gen AI project that shows technical depth, real system design, portfolio quality presentation, and clear business relevance for document intelligence use cases.
                </div>
                <div class="pill-row">
                    <div class="pill">RAG Workflow</div>
                    <div class="pill">Document Question Answering</div>
                    <div class="pill">Semantic Search</div>
                    <div class="pill">NLP Logic</div>
                    <div class="pill">FAISS Retrieval</div>
                    <div class="pill">Streamlit Deployment</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="contact-card">
                <div class="sub-title">Professional Contact</div>
                <div class="contact-grid">
                    <div class="contact-item">
                        <div class="contact-label">LinkedIn</div>
                        <div class="contact-value">
                            <a href="https://www.linkedin.com/in/vamshi-kardhanoori/" target="_blank">View LinkedIn Profile</a>
                        </div>
                    </div>
                    <div class="contact-item">
                        <div class="contact-label">GitHub</div>
                        <div class="contact-value">
                            <a href="https://github.com/vamshi200" target="_blank">View GitHub Portfolio</a>
                        </div>
                    </div>
                    <div class="contact-item">
                        <div class="contact-label">Email</div>
                        <div class="contact-value">
                            <a href="mailto:vamshikardhanoori@gmail.com">Send Email</a>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="photo-card">', unsafe_allow_html=True)

        if PHOTO_PATH.exists():
            st.markdown('<div class="photo-image-wrap">', unsafe_allow_html=True)
            st.image(str(PHOTO_PATH), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Add your image as assets/photo.jpg to display it here.")

        st.markdown(
            """
            <div class="photo-caption">
                This portfolio highlights an applied AI document assistant built for realistic document understanding, semantic retrieval, grounded question answering, and practical extension into enterprise document workflows.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    a, b, c = st.columns(3, gap="large")

    with a:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-title">Project Focus</div>
                <div class="metric-text">
                    Build a portfolio ready AI application that turns uploaded documents into searchable knowledge and grounded answers for practical real world use.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-title">Core Capabilities</div>
                <div class="metric-text">
                    Document loading, semantic chunking, vector embeddings, contextual retrieval, document type detection, guided question answering, and structured feedback.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-title">Recruiter Value</div>
                <div class="metric-text">
                    Demonstrates RAG architecture, NLP driven document understanding, modular engineering, deployable product thinking, and strong applied AI presentation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_details_page():
    st.markdown(
        """
        <div class="section-card">
            <div class="eyebrow">Project Details</div>
            <div class="section-title">Detailed Project Explanation</div>
            <div class="copy">
                Document Intelligence Assistant is an applied AI system built to transform static uploaded documents into interactive and queryable knowledge. Instead of forcing users to manually read through long files to extract a few important details, the application allows them to upload a document, process its content, identify the likely document category, retrieve the most relevant semantic context, and receive grounded answers based on the uploaded file itself.
            </div>
            <div class="copy">
                At a high level, the project addresses a common real world problem: information inside documents is often valuable, but difficult to access quickly. Resumes, passports, bank statements, identity records, onboarding documents, and many other file types usually contain information that a user or business needs immediately. This project reduces that effort by turning the document into an intelligent question answerable interface.
            </div>
            <div class="copy">
                Architecturally, the project is designed as a modular retrieval driven AI pipeline. The uploaded document first goes through a loading stage where text is extracted from PDF or DOCX format. The extracted content is then split into overlapping chunks so important meaning is not lost across boundaries. Those chunks are converted into embeddings using a Hugging Face sentence transformer model and indexed in FAISS for fast semantic similarity search. When the user asks a question, the system retrieves the chunks most relevant to that question and combines the retrieved context with document aware logic to return a grounded answer.
            </div>
            <div class="copy">
                From a recruiter perspective, this project is important because it demonstrates much more than a basic chatbot. It shows understanding of document ingestion, retrieval augmented architectures, embeddings, vector search, semantic matching, modular code organization, front end presentation, user feedback capture, and the broader product thinking required to turn an AI workflow into a usable application.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(
            """
            <div class="info-card">
                <div class="sub-title">RAG in This Project</div>
                <div class="small-copy">
                    Retrieval Augmented Generation is the foundation of this application. In a normal AI interaction, a system may generate an answer from broad model memory, which can be generic, incomplete, or inaccurate when the question depends on a specific uploaded file. In this project, the answer is grounded through retrieval first.
                    <br><br>
                    The document is parsed into text, split into manageable chunks, embedded into semantic vectors, and stored inside FAISS. When a question is asked, the application performs similarity search to identify the chunks whose meaning is closest to the user query. Those retrieved chunks act as the contextual basis for the response.
                    <br><br>
                    This design improves relevance, trust, and explainability. It is especially important in document based workflows because users expect answers to come from the actual file they uploaded, not from unsupported general assumptions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="info-card">
                <div class="sub-title">NLP in This Project</div>
                <div class="small-copy">
                    Natural Language Processing supports the document understanding layer. In this application, NLP oriented techniques are used for text normalization, document type detection, pattern recognition, question intent matching, and targeted extraction of useful fields such as names, dates, email addresses, company names, university references, passport details, and banking related information.
                    <br><br>
                    This means the system does not behave like a plain text search tool. It also interprets how to answer based on the type of uploaded document and the kind of question being asked. That makes the assistant more structured, more useful, and more realistic for practical workflows.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="info-card">
            <div class="sub-title">LLM Role in This Project</div>
            <div class="small-copy">
                Large Language Models are the broader architectural inspiration behind this system. Even though this version is intentionally lightweight and cost conscious, the design pattern used here is aligned with how production Gen AI assistants are structured in enterprise environments.
                <br><br>
                The application separates retrieval from answer generation, which is one of the most important ideas in modern LLM system design. This makes the project future ready. A stronger hosted or local model could later be integrated to provide richer reasoning, more natural responses, and more sophisticated summarization, while still using the same retrieval pipeline already built in this portfolio project.
                <br><br>
                In that sense, this project is not just a demo. It is a strong architectural base for future AI copilots, document assistants, enterprise search tools, and knowledge retrieval systems.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="sub-title">Detailed Uses of This Project</div>
            <div class="copy">
                This project can be used as a personal document assistant for files such as resumes, passports, licenses, bank statements, and bills. It can also evolve into a business facing assistant for HR documents, onboarding material, internal manuals, compliance files, contracts, customer records, and policy search workflows.
            </div>
            <div class="copy">
                For recruiters, the value of this project is that it proves the ability to move from AI concept to working product. It shows how machine learning ideas such as embeddings and retrieval can be connected with real application engineering, front end interaction, modular architecture, and user focused design.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="sub-title">End to End Workflow</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    workflow_steps = [
        "1. The user uploads a supported PDF or DOCX document into the interface.",
        "2. The loader module extracts text content from the uploaded file.",
        "3. The system inspects the extracted content and identifies the likely document type.",
        "4. The text is split into overlapping chunks so long documents remain searchable while preserving nearby context.",
        "5. A Hugging Face sentence transformer model converts each chunk into a dense semantic embedding.",
        "6. FAISS stores those embeddings to support efficient similarity based retrieval.",
        "7. Based on the detected document type, the system suggests useful starter questions for the user.",
        "8. When the user asks a question, the application performs semantic search against the vector index.",
        "9. The most relevant chunks are retrieved and passed into the answer layer.",
        "10. The answer layer combines retrieved context with document aware extraction logic to generate a grounded response.",
        "11. The result is shown in the interface together with the retrieved context so the answer is more transparent and explainable.",
        "12. The reviewer can then leave structured feedback to help improve the application further.",
    ]

    for step in workflow_steps:
        st.markdown(f'<div class="workflow-step">{step}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="note-box">
            This project is designed to be portfolio ready, interview ready, and extension ready. It prioritizes grounded retrieval, modular engineering, explainability, and realistic product design over flashy but unreliable output.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="sub-title">Tech Stack Explained in Detail</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tech_items = [
        (
            "Python",
            "Python is the core programming language used to build the entire application. It manages document ingestion, preprocessing, vector pipeline execution, response generation, session state handling, feedback logging, and the orchestration of all backend components."
        ),
        (
            "Streamlit",
            "Streamlit powers the complete user interface and presentation layer. It enables the application to become an interactive product experience with tabs, structured project explanation, file upload, live demo capability, answer rendering, and feedback collection."
        ),
        (
            "PyPDFLoader",
            "PyPDFLoader is used to extract readable text from uploaded PDF documents. This is a key part of the ingestion stage because PDF files are one of the most common document formats in business and personal workflows."
        ),
        (
            "Docx2txtLoader",
            "Docx2txtLoader enables the application to support DOCX uploads. This broadens usability and shows that the system is not restricted to a single document format."
        ),
        (
            "RecursiveCharacterTextSplitter",
            "This text splitter divides long documents into overlapping chunks. Chunking is critical in retrieval based systems because embeddings work best on smaller context windows and overlap preserves continuity."
        ),
        (
            "Hugging Face Embeddings",
            "Hugging Face embeddings are used to convert text chunks into dense semantic vectors. These vectors capture meaning, not just exact wording."
        ),
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            "This sentence transformer model is lightweight, efficient, and well suited for semantic similarity search without requiring a paid API."
        ),
        (
            "FAISS",
            "FAISS serves as the vector retrieval engine. Once embeddings are created, FAISS indexes them so the application can quickly search across document chunks and identify the parts of the document most relevant to a user query."
        ),
        (
            "Custom NLP and Rule Based Logic",
            "The chain layer combines retrieval with custom logic for document type detection, text normalization, field extraction, question intent recognition, and document specific answering."
        ),
        (
            "JSON Feedback Logging",
            "The application stores submitted user feedback as JSON files. This adds a lightweight but useful feedback system that can later support iterative improvement."
        ),
        (
            "Modular Source Architecture",
            "The codebase is organized into dedicated source modules such as loader, splitter, vectorstore, and chain. This improves readability, debugging, maintainability, and interview readiness."
        ),
    ]

    col1, col2 = st.columns(2, gap="large")
    for i, (name, desc) in enumerate(tech_items):
        with col1 if i % 2 == 0 else col2:
            st.markdown(
                f"""
                <div class="tech-card">
                    <div class="sub-title">{name}</div>
                    <div class="small-copy">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_demo_page():
    st.markdown(
        """
        <div class="demo-card">
            <div class="eyebrow">Live Demo</div>
            <div class="section-title">Try the Document Intelligence Workflow</div>
            <div class="copy">
                Upload a supported document, let the system detect the document type, review suggested questions, and test grounded answers generated from retrieved context.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="usage-note-card">
            <div class="usage-note-title">User Note</div>
            <div class="usage-note-text">
                This project is designed as a hands on implementation of RAG and LLM based document understanding workflows.
                <br><br>
                At this stage, the application is optimized for a limited set of document types, including resumes, passports, driving licenses, bank statements, and bills.
                <br><br>
                For the best results, please upload clear and readable files. Blurry, low quality, or heavily distorted documents may reduce extraction accuracy and answer quality.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF or DOCX file",
        type=["pdf", "docx"],
        help="Supported formats: PDF and DOCX",
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.uploaded_file_name:
            with st.spinner("Processing document..."):
                process_uploaded_file(uploaded_file)

    if st.session_state.uploaded_file_name:
        st.markdown(
            f'<div class="uploaded-box">Uploaded document: {safe_html_text(st.session_state.uploaded_file_name)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="uploaded-box">Detected document type: {safe_html_text(st.session_state.doc_type)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="uploaded-box">Document summary: {safe_html_text(st.session_state.doc_summary)}</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.suggested_questions:
            suggested = st.selectbox(
                "Choose a suggested question",
                options=["Select a suggested question"] + st.session_state.suggested_questions,
                index=0,
            )
            if suggested != "Select a suggested question":
                st.session_state.selected_question = suggested

    question = st.text_input(
        "Ask a question about the uploaded document",
        value=st.session_state.selected_question,
        placeholder="Example: What is this document about?",
    )

    ask = st.button("Generate Answer")

    if ask:
        if st.session_state.vectorstore is None:
            st.warning("Please upload a document first.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Generating grounded answer..."):
                    answer, context = answer_question(
                        question=question.strip(),
                        vectorstore=st.session_state.vectorstore,
                        full_text=st.session_state.full_text,
                        document_type=st.session_state.doc_type,
                    )
                    st.session_state.answer = clean_text_for_display(answer)
                    st.session_state.context = clean_text_for_display(context)
            except Exception as e:
                st.error(f"Answer generation failed: {clean_text_for_display(str(e))}")

    if st.session_state.answer:
        st.markdown(
            f"""
            <div class="answer-card">
                <div class="sub-title">Grounded Answer</div>
                <div class="answer-box">{safe_html_text(st.session_state.answer)}</div>
                <div class="sub-title" style="margin-top:20px;">Retrieved Context</div>
                <div class="context-box">{safe_html_text(st.session_state.context)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="note-banner">
            <div class="training-note">This project is still being trained. Please ignore any moments of confusion.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feedback-card">
            <div class="eyebrow">Feedback</div>
            <div class="section-title">Viewer Feedback</div>
            <div class="copy">
                Please select a feedback category first and then enter your comments. This helps organize responses more clearly and makes feedback more useful for future improvement.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("feedback_form", clear_on_submit=True):
        c1, c2 = st.columns(2)

        with c1:
            name = st.text_input("Your name")
            feedback_type = st.selectbox(
                "Feedback category",
                [
                    "Select feedback category",
                    "UI and Design",
                    "Project Clarity",
                    "Answer Quality",
                    "Feature Suggestion",
                    "Bug Report",
                    "Recruiter Impression",
                    "General Feedback",
                ],
            )

        with c2:
            rating = st.selectbox(
                "Overall rating",
                [
                    "Select rating",
                    "Excellent",
                    "Good",
                    "Average",
                    "Needs Improvement",
                ],
            )

        message = st.text_area(
            "Your feedback",
            height=160,
            placeholder="Write your feedback here...",
        )

        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            if feedback_type == "Select feedback category":
                st.warning("Please select a feedback category.")
            elif rating == "Select rating":
                st.warning("Please select an overall rating.")
            elif not message.strip():
                st.warning("Please enter your feedback before submitting.")
            else:
                save_feedback(name, feedback_type, rating, message)
                st.success("Thank you. Your feedback has been submitted.")


def render_footer():
    st.markdown(
        """
        <div class="footer-line">
            Built by Vamshi Kardhanoori • Document Intelligence Assistant • Gen AI Project Portfolio
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    init_session()
    inject_css()

    st.markdown('<div class="main-shell">', unsafe_allow_html=True)

    tabs = st.tabs(
        [
            "About Me and Project",
            "Project Details",
            "Live Demo and Feedback",
        ]
    )

    with tabs[0]:
        render_home_page()

    with tabs[1]:
        render_details_page()

    with tabs[2]:
        render_demo_page()

    render_footer()
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()