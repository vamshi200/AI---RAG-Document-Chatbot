import os
from pathlib import Path

import streamlit as st

from src.loader import load_document
from src.splitter import split_document
from src.vectorstore import create_vectorstore
from src.qa_chain import (
    detect_document_type,
    get_suggested_questions,
    answer_question,
    summarize_document_for_home,
)

st.set_page_config(
    page_title="Vamshi Kardhanoori | Document Intelligence Assistant",
    page_icon="📄",
    layout="wide",
)

PHOTO_PATH = Path("assets/photo.jpg")


def init_state():
    defaults = {
        "page": "Project Home",
        "vectorstore": None,
        "chunks": [],
        "document_name": "",
        "document_type": "",
        "document_text": "",
        "suggested_questions": [],
        "last_answer": "",
        "last_context": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_page(page_name: str):
    st.session_state.page = page_name


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(0, 94, 255, 0.14), transparent 30%),
                radial-gradient(circle at bottom right, rgba(0, 178, 255, 0.10), transparent 28%),
                linear-gradient(135deg, #030712 0%, #071426 100%);
            color: #f8fafc;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        .hero-card, .section-card, .answer-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 18px 40px rgba(0,0,0,0.25);
        }

        .hero-label {
            color: #7dd3fc;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-weight: 700;
            font-size: 0.78rem;
            margin-bottom: 10px;
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 8px;
        }

        .hero-role {
            color: #7dd3fc;
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 18px;
        }

        .hero-text {
            color: #dbe4ee;
            font-size: 1rem;
            line-height: 1.85;
            margin-bottom: 16px;
        }

        .section-title {
            color: #f8fafc;
            font-weight: 800;
            font-size: 1.15rem;
            margin-bottom: 12px;
        }

        .section-body {
            color: #dbe4ee;
            font-size: 0.98rem;
            line-height: 1.8;
        }

        .tag {
            display: inline-block;
            background: rgba(59,130,246,0.14);
            color: #dbeafe;
            border: 1px solid rgba(147,197,253,0.20);
            padding: 8px 14px;
            border-radius: 999px;
            margin: 6px 8px 0 0;
            font-size: 0.92rem;
            font-weight: 600;
        }

        .project-subtitle {
            color: #7dd3fc;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 14px;
        }

        .project-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 12px;
        }

        .feedback-note {
            background: rgba(125, 211, 252, 0.08);
            border: 1px solid rgba(125, 211, 252, 0.18);
            border-radius: 18px;
            padding: 18px;
            color: #e5e7eb;
            line-height: 1.8;
        }

        .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.10);
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            color: #f8fafc;
            font-weight: 600;
        }

        .stButton > button:hover {
            border-color: rgba(125,211,252,0.30);
            color: #7dd3fc;
        }

        .stTextInput input, .stTextArea textarea {
            border-radius: 14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("## Navigation")

        if st.button("Project Home", use_container_width=True):
            set_page("Project Home")
        if st.button("Project Details", use_container_width=True):
            set_page("Project Details")
        if st.button("Live Demo", use_container_width=True):
            set_page("Live Demo")

        st.markdown("### Project Builder")
        st.markdown("**Vamshi Kardhanoori**")
        st.markdown("Gen AI Engineer")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/vamshi-kardhanoori/)")
        st.markdown("[GitHub](https://github.com/vamshi200)")
        st.markdown("[vamshikardhanoori@gmail.com](mailto:vamshikardhanoori@gmail.com)")

        if PHOTO_PATH.exists():
            st.image(str(PHOTO_PATH), use_container_width=True)


def render_project_home():
    st.markdown('<div class="project-subtitle">Document Intelligence Project Portfolio</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.3], gap="large")

    with left:
        if PHOTO_PATH.exists():
            st.image(str(PHOTO_PATH), use_container_width=True)
        else:
            st.warning("Add your image to assets/photo.jpg")

    with right:
        st.markdown(
            """
            <div class="hero-card">
                <div class="hero-label">Project Builder</div>
                <div class="hero-title">Vamshi Kardhanoori</div>
                <div class="hero-role">Gen AI Engineer</div>

                <div class="hero-text">
                    The code speaks for itself, but I’m still happy to say: yes, I’m open to work.
                </div>

                <div class="hero-text">
                    This project portfolio presents a practical document intelligence assistant built to classify document types,
                    retrieve relevant context, and answer grounded questions from uploaded content.
                    The focus is on applied AI engineering, product thinking, and real world implementation quality.
                </div>

                <div class="hero-text">
                    I am a graduate student at Missouri University of Science and Technology, pursuing a Master’s in
                    Information Science and Technology. My academic background and technical work are centered around
                    Machine Learning, Generative AI, LLM systems, NLP, and document understanding workflows.
                </div>

                <div class="hero-text">
                    My career path is aligned with Gen AI Engineering, AI application development, LLM systems,
                    production oriented machine learning, and intelligent user facing products.
                </div>

                <div class="hero-text" style="color:#7dd3fc; font-weight:700; margin-top:8px;">
                    Project details are available in the next section.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Education</div>
                <div class="section-body">
                    Missouri University of Science and Technology<br>
                    Master’s in Information Science and Technology
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Core Interests</div>
                <div class="section-body">
                    <span class="tag">Machine Learning</span>
                    <span class="tag">Generative AI</span>
                    <span class="tag">LLMs</span>
                    <span class="tag">NLP</span>
                    <span class="tag">RAG</span>
                    <span class="tag">Document Intelligence</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Contact</div>
                <div class="section-body">
                    LinkedIn<br>
                    linkedin.com/in/vamshi-kardhanoori/<br><br>
                    GitHub<br>
                    github.com/vamshi200<br><br>
                    Email<br>
                    vamshikardhanoori@gmail.com
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_project_details():
    st.markdown('<div class="project-subtitle">Project Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<div class="project-title">Document Intelligence Assistant</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Project Overview</div>
            <div class="section-body">
                This project is a document intelligence assistant that allows users to upload files and ask grounded
                questions based on document content. It is designed to support workflows involving resumes, passports,
                driving licenses, bank statements, and other structured or semi structured business and personal documents.
                <br><br>
                The application combines document parsing, semantic chunking, embedding generation, vector retrieval,
                document type detection, question intent handling, and answer generation into a single interactive workflow.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">RAG in this Project</div>
                <div class="section-body">
                    Retrieval Augmented Generation is used so the system answers from uploaded document content rather than
                    relying only on general model behavior. The document is split into chunks, embedded into vectors,
                    stored in FAISS, and searched at question time to retrieve the most relevant context before answering.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">NLP in this Project</div>
                <div class="section-body">
                    NLP helps the system understand question intent, detect entities like names, dates, companies,
                    universities, nationalities, email addresses, and other fields, and produce more structured answers
                    depending on the document type.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Tech Stack Explained</div>
            <div class="section-body">
                <strong>Python</strong><br>
                Python is used as the core application language for orchestration, data flow, answer handling, and document processing.
                <br><br>

                <strong>Streamlit</strong><br>
                Streamlit provides the interface for navigation, file upload, project presentation, question answering, and feedback capture.
                <br><br>

                <strong>LangChain Style Flow</strong><br>
                The application follows a retrieval pipeline architecture where documents are loaded, split, indexed, retrieved, and answered in sequence.
                <br><br>

                <strong>Sentence Transformers</strong><br>
                Semantic embeddings are generated so the app can compare text by meaning rather than only exact keyword matches.
                <br><br>

                <strong>FAISS</strong><br>
                FAISS acts as the vector store used for similarity search on document chunks.
                <br><br>

                <strong>Document Loaders</strong><br>
                Document loaders extract usable text from uploaded PDF and DOCX files.
                <br><br>

                <strong>Chunking Strategy</strong><br>
                Large documents are split into overlapping text chunks so important context is preserved during retrieval.
                <br><br>

                <strong>Answering Layer</strong><br>
                The answering logic uses detected document type, retrieved context, and question pattern analysis to provide grounded responses.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">End to End Workflow</div>
            <div class="section-body">
                1. User uploads a document.<br>
                2. Text is extracted from the file.<br>
                3. Document type is detected.<br>
                4. The content is split into chunks.<br>
                5. Embeddings are created.<br>
                6. Chunks are indexed in FAISS.<br>
                7. Suggested questions are shown based on the detected type.<br>
                8. User asks a question.<br>
                9. Relevant chunks are retrieved.<br>
                10. The answer is generated from the retrieved content.<br>
                11. Retrieved context is displayed for transparency.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
        <div class="feedback-note">
            <strong>This project is still being trained. Please ignore any moments of confusion.</strong>
            <br><br>
            Feedback is welcome. If you notice any issue, inconsistency, or improvement opportunity, please share it below.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    with st.form("feedback_form", clear_on_submit=True):
        feedback_name = st.text_input("Name")
        feedback_text = st.text_area("Feedback", height=150)
        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            os.makedirs("feedback", exist_ok=True)
            feedback_path = Path("feedback") / "viewer_feedback.txt"
            with open(feedback_path, "a", encoding="utf-8") as f:
                f.write(f"Name: {feedback_name.strip() or 'Anonymous'}\n")
                f.write(f"Feedback: {feedback_text.strip()}\n")
                f.write("=" * 80 + "\n")
            st.success("Feedback submitted successfully.")


def process_uploaded_document(uploaded_file):
    file_extension = Path(uploaded_file.name).suffix.lower()

    temp_dir = Path("data")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / uploaded_file.name

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    documents = load_document(str(temp_file_path), file_extension)
    full_text = "\n".join([doc.page_content for doc in documents]).strip()

    chunks = split_document(documents)
    vectorstore = create_vectorstore(chunks)
    doc_type = detect_document_type(full_text)
    suggested = get_suggested_questions(doc_type)

    st.session_state.vectorstore = vectorstore
    st.session_state.chunks = chunks
    st.session_state.document_name = uploaded_file.name
    st.session_state.document_type = doc_type
    st.session_state.document_text = full_text
    st.session_state.suggested_questions = suggested
    st.session_state.last_answer = ""
    st.session_state.last_context = ""


def render_live_demo():
    st.markdown('<div class="project-subtitle">Interactive Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="project-title">Upload a document and ask grounded questions</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx"],
        help="Supported formats: PDF and DOCX",
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.document_name:
            process_uploaded_document(uploaded_file)

        st.success("Document processed successfully.")
        st.info(f"Detected document type: {st.session_state.document_type}")

        if st.session_state.document_text:
            st.markdown("## Suggested questions")
            for q in st.session_state.suggested_questions:
                st.markdown(f"• {q}")

            question = st.text_input(
                "Ask a question about the uploaded document",
                placeholder="Type your question here",
            )

            if question:
                answer, context = answer_question(
                    question=question,
                    vectorstore=st.session_state.vectorstore,
                    full_text=st.session_state.document_text,
                    document_type=st.session_state.document_type,
                )
                st.session_state.last_answer = answer
                st.session_state.last_context = context

            if st.session_state.last_answer:
                st.markdown("## Answer")
                st.markdown(
                    f'<div class="answer-card">{st.session_state.last_answer}</div>',
                    unsafe_allow_html=True,
                )

            if st.session_state.last_context:
                with st.expander("Retrieved Context"):
                    st.write(st.session_state.last_context)

            with st.expander("Document summary"):
                summary = summarize_document_for_home(
                    st.session_state.document_type,
                    st.session_state.document_text,
                )
                st.write(summary)


def main():
    init_state()
    inject_css()
    render_sidebar()

    if st.session_state.page == "Project Home":
        render_project_home()
    elif st.session_state.page == "Project Details":
        render_project_details()
    else:
        render_live_demo()


if __name__ == "__main__":
    main()