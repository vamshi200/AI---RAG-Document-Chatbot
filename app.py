import os
from pathlib import Path

import streamlit as st

from src.loader import load_document
from src.splitter import split_document
from src.vectorstore import create_vectorstore
from src.qa_chain import detect_document_type, get_suggested_questions, answer_question


PHOTO_PATH = Path("assets/photo.jpg")
FEEDBACK_DIR = Path("feedback")
FEEDBACK_FILE = FEEDBACK_DIR / "viewer_feedback.txt"


st.set_page_config(
    page_title="Vamshi Kardhanoori | Document Intelligence Assistant",
    page_icon="📄",
    layout="wide",
)


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(0, 119, 255, 0.14), transparent 25%),
                radial-gradient(circle at bottom right, rgba(0, 180, 255, 0.08), transparent 25%),
                linear-gradient(135deg, #030712 0%, #07111f 45%, #020617 100%);
            color: #f8fafc;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        .block-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.25);
        }

        .small-label {
            color: #7dd3fc;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .main-title {
            font-size: 3rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 6px;
        }

        .role-line {
            color: #67e8f9;
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 14px;
        }

        .muted-text {
            color: #dbe4ee;
            line-height: 1.9;
            font-size: 1rem;
        }

        .section-heading {
            color: #f8fafc;
            font-weight: 800;
            font-size: 1.15rem;
            margin-bottom: 8px;
        }

        .tag {
            display: inline-block;
            padding: 8px 14px;
            margin: 6px 8px 0 0;
            border-radius: 999px;
            background: rgba(59,130,246,0.14);
            border: 1px solid rgba(96,165,250,0.22);
            color: #dbeafe;
            font-size: 0.92rem;
            font-weight: 600;
        }

        .notice {
            background: rgba(125, 211, 252, 0.08);
            border: 1px solid rgba(125, 211, 252, 0.18);
            border-radius: 18px;
            padding: 18px;
            color: #e5e7eb;
            line-height: 1.8;
        }

        .answer-box {
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 18px;
            color: #f8fafc;
            line-height: 1.8;
        }

        .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            color: white;
            font-weight: 600;
        }

        .stButton > button:hover {
            border-color: rgba(125,211,252,0.4);
            color: #7dd3fc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def render_sidebar():
    with st.sidebar:
        st.markdown("## Navigation")

        if st.button("Project Home", use_container_width=True):
            set_page("Project Home")
        if st.button("Project Details", use_container_width=True):
            set_page("Project Details")
        if st.button("Live Demo", use_container_width=True):
            set_page("Live Demo")

        st.markdown("---")
        st.markdown("### Project Builder")
        st.markdown("**Vamshi Kardhanoori**")
        st.caption("Gen AI Engineer")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/vamshi-kardhanoori/)")
        st.markdown("[GitHub](https://github.com/vamshi200)")
        st.markdown("[vamshikardhanoori@gmail.com](mailto:vamshikardhanoori@gmail.com)")

        st.markdown("---")
        if PHOTO_PATH.exists():
            st.image(str(PHOTO_PATH), use_container_width=True)


def render_home():
    st.markdown('<div class="small-label">Document Intelligence Project Portfolio</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        if PHOTO_PATH.exists():
            st.image(str(PHOTO_PATH), use_container_width=True)
        else:
            st.warning("Add your profile image to assets/photo.jpg")

    with right:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-label">Project Builder</div>', unsafe_allow_html=True)
        st.markdown('<div class="main-title">Vamshi Kardhanoori</div>', unsafe_allow_html=True)
        st.markdown('<div class="role-line">Gen AI Engineer</div>', unsafe_allow_html=True)

        st.write("The code speaks for itself, but I’m still happy to say: yes, I’m open to work.")

        st.write(
            "This project portfolio showcases a practical document intelligence assistant built to "
            "classify uploaded documents, retrieve relevant context, and answer grounded questions. "
            "The goal is to present applied AI engineering through a real working system rather than a static profile."
        )

        st.write(
            "I am a graduate student at Missouri University of Science and Technology, pursuing a Master’s in "
            "Information Science and Technology. My academic and technical path is centered around Machine Learning, "
            "Generative AI, LLM systems, NLP, and intelligent document understanding."
        )

        st.write(
            "My career path is aligned with Gen AI Engineering, LLM applications, AI product development, "
            "production oriented software systems, and real world business use cases."
        )

        st.info("Project details are available in the next section.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Education</div>', unsafe_allow_html=True)
        st.write("Missouri University of Science and Technology")
        st.write("Master’s in Information Science and Technology")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Interests</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <span class="tag">Machine Learning</span>
            <span class="tag">Generative AI</span>
            <span class="tag">LLMs</span>
            <span class="tag">NLP</span>
            <span class="tag">RAG</span>
            <span class="tag">Document Intelligence</span>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Contact</div>', unsafe_allow_html=True)
        st.write("LinkedIn")
        st.write("linkedin.com/in/vamshi-kardhanoori/")
        st.write("GitHub")
        st.write("github.com/vamshi200")
        st.write("Email")
        st.write("vamshikardhanoori@gmail.com")
        st.markdown("</div>", unsafe_allow_html=True)


def render_project_details():
    st.markdown('<div class="small-label">Project Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title" style="font-size:2.4rem;">Document Intelligence Assistant</div>', unsafe_allow_html=True)

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Project Overview</div>', unsafe_allow_html=True)
    st.write(
        "This project is a document intelligence assistant that allows users to upload files and ask grounded "
        "questions based on document content. It is designed for practical workflows involving resumes, passports, "
        "driving licenses, bank statements, and other personal or business documents."
    )
    st.write(
        "The system combines document loading, text extraction, semantic chunking, embedding generation, "
        "vector retrieval, document type detection, question intent handling, and answer generation into a single interactive application."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">RAG in this Project</div>', unsafe_allow_html=True)
        st.write(
            "Retrieval Augmented Generation is used so answers come from the uploaded document instead of depending only on "
            "general model behavior. The document is split into chunks, transformed into embeddings, stored in FAISS, "
            "and searched at question time to retrieve the most relevant context."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">NLP in this Project</div>', unsafe_allow_html=True)
        st.write(
            "NLP techniques help the system understand question intent and identify entities such as names, dates, "
            "passport numbers, universities, companies, emails, nationalities, and other document specific fields."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Tech Stack Explained</div>', unsafe_allow_html=True)

    st.write("**Python**")
    st.write("Python is the application backbone and is used for document processing, retrieval logic, data flow, and answer generation.")

    st.write("**Streamlit**")
    st.write("Streamlit powers the user interface. It makes it possible to build an interactive app with navigation, file upload, answer display, project explanation pages, and feedback collection.")

    st.write("**LangChain Style Workflow**")
    st.write("The project follows a retrieval pipeline structure where documents are loaded, split, indexed, and queried in sequence. This keeps the architecture modular and easier to extend.")

    st.write("**Sentence Transformers**")
    st.write("Semantic embeddings are created with a sentence transformer model so text chunks can be compared by meaning, not only by exact keyword matching.")

    st.write("**FAISS**")
    st.write("FAISS is used as the vector database layer for fast similarity search over embedded chunks. When a user asks a question, the system searches FAISS to find the chunks most likely to contain the answer.")

    st.write("**Document Loaders**")
    st.write("The loader supports uploaded PDF and DOCX files and converts them into usable text for downstream retrieval.")

    st.write("**Chunking Strategy**")
    st.write("Large files are split into overlapping chunks so important context is preserved and long documents can still be searched effectively.")

    st.write("**Answering Layer**")
    st.write("The answering layer combines detected document type, retrieved context, and question pattern matching to produce grounded outputs.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">End to End Workflow</div>', unsafe_allow_html=True)
    st.write("1. User uploads a document.")
    st.write("2. The file is parsed and text is extracted.")
    st.write("3. The document type is detected.")
    st.write("4. The content is split into chunks.")
    st.write("5. Embeddings are generated for the chunks.")
    st.write("6. The chunks are stored in FAISS.")
    st.write("7. Suggested questions are generated based on document type.")
    st.write("8. The user asks a question.")
    st.write("9. Relevant chunks are retrieved.")
    st.write("10. The system returns a grounded answer with retrieved context.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown(
        """
        <div class="notice">
            <strong>This project is still being trained. Please ignore any moments of confusion.</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Viewer Feedback</div>', unsafe_allow_html=True)

    with st.form("feedback_form", clear_on_submit=True):
        viewer_name = st.text_input("Name")
        viewer_feedback = st.text_area("Feedback", height=140)
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            FEEDBACK_DIR.mkdir(exist_ok=True)
            with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                f.write(f"Name: {viewer_name.strip() or 'Anonymous'}\n")
                f.write(f"Feedback: {viewer_feedback.strip()}\n")
                f.write("-" * 80 + "\n")
            st.success("Feedback submitted successfully.")

    st.markdown("</div>", unsafe_allow_html=True)


def process_uploaded_document(uploaded_file):
    temp_dir = Path("data")
    temp_dir.mkdir(exist_ok=True)

    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    extension = file_path.suffix.lower()
    documents = load_document(str(file_path), extension)

    full_text = "\n".join(doc.page_content for doc in documents).strip()
    chunks = split_document(documents)
    vectorstore = create_vectorstore(chunks)
    document_type = detect_document_type(full_text)
    suggested_questions = get_suggested_questions(document_type)

    st.session_state.vectorstore = vectorstore
    st.session_state.chunks = chunks
    st.session_state.document_name = uploaded_file.name
    st.session_state.document_type = document_type
    st.session_state.document_text = full_text
    st.session_state.suggested_questions = suggested_questions
    st.session_state.last_answer = ""
    st.session_state.last_context = ""


def render_live_demo():
    st.markdown('<div class="small-label">Interactive Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title" style="font-size:2.3rem;">Live Document Demo</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx"],
        help="Supported file types: PDF and DOCX",
    )

    if uploaded_file is None:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">What this demo does</div>', unsafe_allow_html=True)
        st.write("Upload a supported file and the app will detect the document type, suggest useful questions, retrieve relevant context, and generate grounded answers.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if uploaded_file.name != st.session_state.document_name:
        process_uploaded_document(uploaded_file)

    st.success("Document processed successfully.")
    st.info(f"Detected document type: {st.session_state.document_type}")

    if st.session_state.suggested_questions:
        st.markdown("## Suggested questions")
        for q in st.session_state.suggested_questions:
            st.write(f"• {q}")

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
            f'<div class="answer-box">{st.session_state.last_answer}</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.last_context:
        with st.expander("Retrieved Context"):
            st.write(st.session_state.last_context)


def main():
    init_state()
    inject_css()
    render_sidebar()

    if st.session_state.page == "Project Home":
        render_home()
    elif st.session_state.page == "Project Details":
        render_project_details()
    elif st.session_state.page == "Live Demo":
        render_live_demo()
    else:
        render_home()


if __name__ == "__main__":
    main()