import csv
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.loader import load_document
from src.splitter import split_documents
from src.vectorstore import create_vectorstore
from src.qa_chain import (
    get_llm,
    answer_question,
    detect_document_type,
    get_suggested_questions,
    get_full_text,
    extract_known_fields,
)

st.set_page_config(
    page_title="Vamshi Kardhanoori | Document Intelligence Assistant",
    page_icon="📄",
    layout="wide",
)

PHOTO_PATH = Path("assets/photo.jpg")
FEEDBACK_DIR = Path("feedback")
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.csv"

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "llm" not in st.session_state:
    st.session_state.llm = get_llm()

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = None

if "doc_type" not in st.session_state:
    st.session_state.doc_type = None

if "known_fields" not in st.session_state:
    st.session_state.known_fields = {}

if "page" not in st.session_state:
    st.session_state.page = "home"

st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 28%),
            radial-gradient(circle at top right, rgba(59, 130, 246, 0.08), transparent 30%),
            linear-gradient(180deg, #040814 0%, #050816 100%);
        color: #f8fafc;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.2rem;
        max-width: 1180px;
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 14px;
        padding: 0.72rem 1rem;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        color: #ffffff;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }

    div[data-testid="stButton"] > button:hover {
        border-color: rgba(96, 165, 250, 0.8);
        background: linear-gradient(180deg, #12203a 0%, #0f172a 100%);
        color: #ffffff;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(11, 18, 32, 0.96), rgba(17, 24, 39, 0.96));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 1.7rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 34px rgba(0,0,0,0.24);
    }

    .section-card {
        background: linear-gradient(180deg, rgba(11,18,32,0.98), rgba(10,14,28,0.98));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 1.25rem;
        height: 100%;
        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
    }

    .title-main {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.35rem;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }

    .subtitle-main {
        font-size: 1.05rem;
        color: #93c5fd;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }

    .body-text {
        font-size: 0.98rem;
        color: #dbe4f0;
        line-height: 1.82;
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.65rem;
        letter-spacing: -0.01em;
    }

    .chip {
        display: inline-block;
        padding: 0.46rem 0.88rem;
        margin: 0.18rem 0.35rem 0.18rem 0;
        border-radius: 999px;
        background: rgba(19, 32, 58, 0.95);
        color: #e2e8f0;
        font-size: 0.88rem;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .project-title {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }

    .project-subtitle {
        color: #d7e1ee;
        font-size: 1rem;
        line-height: 1.78;
    }

    .feature-card {
        background: linear-gradient(180deg, rgba(11,18,32,0.98), rgba(10,14,28,0.98));
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1rem;
        border-radius: 18px;
        height: 100%;
        box-shadow: 0 8px 22px rgba(0,0,0,0.16);
    }

    .feature-title {
        font-size: 1rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.35rem;
    }

    .feature-text {
        font-size: 0.95rem;
        color: #dbe4f0;
        line-height: 1.68;
    }

    .link-grid {
        display: grid;
        gap: 0.65rem;
        margin-top: 0.65rem;
    }

    .contact-pill {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.8rem 0.95rem;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(255,255,255,0.08);
        color: #e2e8f0;
        text-decoration: none !important;
        font-size: 0.94rem;
        line-height: 1.5;
    }

    .contact-pill:hover {
        border-color: rgba(96,165,250,0.75);
        color: #ffffff;
        text-decoration: none !important;
    }

    .contact-icon {
        width: 30px;
        height: 30px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(59,130,246,0.12);
        color: #93c5fd;
        font-size: 0.95rem;
        font-weight: 800;
        flex-shrink: 0;
    }

    .answer-box {
        background: linear-gradient(180deg, rgba(11,18,32,0.98), rgba(10,14,28,0.98));
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1rem;
        border-radius: 16px;
        color: #f8fafc;
    }

    .feedback-note {
        color: #dbeafe;
        font-size: 0.98rem;
        margin-bottom: 0.6rem;
    }

    .photo-frame {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 28px rgba(0,0,0,0.20);
    }

    .mini-label {
        color: #8fbaf7;
        font-size: 0.83rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.55rem;
    }

    .soft-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin-top: 0.9rem;
        margin-bottom: 0.9rem;
    }

    .hover-tip {
        position: relative;
        display: inline-block;
        cursor: default;
        border-bottom: 1px dashed rgba(147,197,253,0.35);
    }

    .hover-tip .tooltip-text {
        visibility: hidden;
        opacity: 0;
        width: max-content;
        max-width: 260px;
        background: rgba(15, 23, 42, 0.97);
        color: #e2e8f0;
        text-align: left;
        border-radius: 12px;
        padding: 0.65rem 0.8rem;
        position: absolute;
        z-index: 999;
        bottom: 130%;
        left: 50%;
        transform: translateX(-50%) translateY(6px);
        transition: opacity 0.22s ease, transform 0.22s ease;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 10px 26px rgba(0,0,0,0.26);
        font-size: 0.86rem;
        line-height: 1.55;
        pointer-events: none;
        white-space: normal;
    }

    .hover-tip .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -6px;
        border-width: 6px;
        border-style: solid;
        border-color: rgba(15, 23, 42, 0.97) transparent transparent transparent;
    }

    .hover-tip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%) translateY(0);
    }
</style>
""", unsafe_allow_html=True)


def save_feedback(name: str, email: str, feedback_type: str, message: str):
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = FEEDBACK_FILE.exists()

    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "name", "email", "feedback_type", "message"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            name.strip(),
            email.strip(),
            feedback_type.strip(),
            message.strip(),
        ])


nav_left, nav_mid, nav_right, _ = st.columns([1, 1, 1, 5])

with nav_left:
    if st.button("Project Home"):
        st.session_state.page = "home"

with nav_mid:
    if st.button("Project Details"):
        st.session_state.page = "details"

with nav_right:
    if st.button("Live Demo"):
        st.session_state.page = "demo"


if st.session_state.page == "home":
    left, right = st.columns([1.02, 1.68], gap="large")

    with left:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown('<div class="mini-label">Project Builder</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="link-grid">
            <a class="contact-pill" href="https://www.linkedin.com/in/vamshi-kardhanoori/" target="_blank">
                <span class="contact-icon">in</span>
                <span>LinkedIn</span>
            </a>
            <a class="contact-pill" href="https://github.com/vamshi200" target="_blank">
                <span class="contact-icon">GH</span>
                <span>GitHub</span>
            </a>
            <a class="contact-pill" href="mailto:vamshikardhanoori@gmail.com">
                <span class="contact-icon">@</span>
                <span>vamshikardhanoori@gmail.com</span>
            </a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

        if PHOTO_PATH.exists():
            st.markdown('<div class="photo-frame">', unsafe_allow_html=True)
            st.image(str(PHOTO_PATH), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Place your image at assets/photo.jpg")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="title-main">
            <span class="hover-tip">
                Vamshi Kardhanoori
                <span class="tooltip-text">Open to Gen AI, Machine Learning, NLP, and AI Engineering opportunities.</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="subtitle-main">
            <span class="hover-tip">
                Gen AI Engineer
                <span class="tooltip-text">Focused on practical AI systems, document intelligence, LLM workflows, and real world AI applications.</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            """
            <div class="body-text">
                The code speaks for itself, but I’m still happy to say: yes, I’m open to work.
                <br><br>
                This project is designed as a practical AI document understanding system. It combines retrieval based
                question answering, document type detection, structured field extraction, and safer fallback behavior
                into a single application. The goal is to make document interaction more useful, grounded, and reliable
                for real users and real files.
                <br><br>
                I built this project as part of my broader interest in
                <span class="hover-tip">Machine Learning<span class="tooltip-text">Building predictive systems and intelligent workflows from data.</span></span>,
                <span class="hover-tip">Generative AI<span class="tooltip-text">Designing AI systems that can generate, summarize, and assist with useful outputs.</span></span>,
                <span class="hover-tip">Large Language Models<span class="tooltip-text">Working with modern language models for reasoning, answering, and automation.</span></span>,
                and
                <span class="hover-tip">Natural Language Processing<span class="tooltip-text">Understanding and processing human language in documents, text, and user prompts.</span></span>.
                My work is centered on building production oriented AI applications that connect
                strong engineering with real world business use cases.
                <br><br>
                I am a graduate student at
                <span class="hover-tip">Missouri University of Science and Technology<span class="tooltip-text">Master’s in Information Science and Technology.</span></span>,
                pursuing a Master’s in Information Science and Technology. My career path is aligned with Gen AI Engineering,
                intelligent automation, LLM systems, NLP workflows, and practical AI products for business and user facing environments.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">
                <span class="hover-tip">
                    Education
                    <span class="tooltip-text">Academic foundation supporting my work in AI, data systems, and intelligent applications.</span>
                </span>
            </div>
            <div class="body-text">
                Missouri University of Science and Technology<br>
                Master’s in Information Science and Technology
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">
                <span class="hover-tip">
                    Core Interests
                    <span class="tooltip-text">The technologies and problem spaces I am actively building in.</span>
                </span>
            </div>
            <span class="chip">Machine Learning</span>
            <span class="chip">Generative AI</span>
            <span class="chip">LLMs</span>
            <span class="chip">NLP</span>
            <span class="chip">RAG Systems</span>
            <span class="chip">AI Applications</span>
            <span class="chip">Document Intelligence</span>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">
                <span class="hover-tip">
                    Project Focus
                    <span class="tooltip-text">The core engineering goals behind this document intelligence project.</span>
                </span>
            </div>
            <div class="body-text">
                Grounded document Q and A<br>
                Structured extraction from real documents<br>
                Safer answer handling for unsupported queries<br>
                Real world AI application design
            </div>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.page == "details":
    st.markdown("""
    <div class="hero-card">
        <div class="project-title">Project Details</div>
        <div class="project-subtitle">
            This project is an AI powered document understanding application that can identify document types,
            extract structured information, retrieve relevant context, and answer questions in a grounded way.
            It is designed to show how modern Gen AI systems can be used for practical document intelligence workflows.
        </div>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4, gap="large")

    with f1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Document Type Detection</div>
            <div class="feature-text">
                The application first analyzes the uploaded file and determines what type of document it is.
                It can work with resumes, passports, bank statements, driving licenses, invoices, receipts,
                and general document files.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Structured Field Extraction</div>
            <div class="feature-text">
                For documents with clear structure, the system extracts specific fields such as name, date of birth,
                expiry date, place of issue, account information, or other document specific values.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Retrieval Based Question Answering</div>
            <div class="feature-text">
                The system retrieves relevant chunks from the uploaded document and uses that context to answer
                user questions instead of relying on unsupported guesswork.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Safe Fallback Behavior</div>
            <div class="feature-text">
                If the answer cannot be verified from the uploaded document, the application responds with a
                refusal style fallback rather than inventing information.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### What this project does")
    st.write(
        "This project allows a user to upload a PDF or DOCX document, converts that document into searchable text, "
        "splits the content into smaller chunks, stores those chunks in a vector database, retrieves the most relevant "
        "sections based on a user question, and then generates an answer using grounded context. For structured documents, "
        "it also uses rule based extraction logic to answer sensitive factual fields more reliably."
    )

    st.markdown("### What RAG means in this project")
    st.write(
        "RAG stands for Retrieval Augmented Generation. In simple terms, it means the model does not answer only from memory. "
        "Instead, it first retrieves the most relevant parts of the uploaded document and then uses those retrieved sections "
        "to build the answer. This matters because it improves answer grounding, reduces hallucinations, and makes the system "
        "more useful for document based interaction."
    )
    st.write(
        "In this project, RAG is used so the answer stays tied to the actual uploaded file. When a user asks a question, "
        "the system searches the document chunks, selects the most relevant pieces, and then answers from those pieces. "
        "That makes the workflow more practical for resumes, passports, bank statements, invoices, and other files where "
        "the answer must come from the uploaded content."
    )

    st.markdown("### What NLP means in this project")
    st.write(
        "NLP stands for Natural Language Processing. It is the area of AI focused on how systems read, interpret, process, "
        "and generate human language. NLP is relevant here because the project needs to understand user questions, process "
        "text extracted from documents, detect patterns such as names and dates, and produce meaningful responses."
    )
    st.write(
        "This project uses NLP ideas in several places. It uses text processing to normalize content, document analysis to "
        "detect likely document types, extraction logic for structured fields, and language generation for final answers. "
        "Even when the final answer is generated by a language model, the surrounding NLP pipeline is what makes the system usable."
    )

    st.markdown("### Complete workflow")
    st.write("Step one is document upload. The user uploads a PDF or DOCX file through the Streamlit interface.")
    st.write("Step two is document loading. The loader reads the file and extracts raw text from it.")
    st.write("Step three is text chunking. The splitter breaks long document text into smaller manageable sections so retrieval works better.")
    st.write("Step four is vector indexing. The vector store converts chunks into embeddings and stores them for similarity search.")
    st.write("Step five is retrieval. When a question is asked, the retriever finds the most relevant chunks from the uploaded document.")
    st.write("Step six is answer generation. The question and retrieved chunks are passed into the answering logic so the final answer is grounded.")
    st.write("Step seven is safety. If the requested answer cannot be confirmed confidently, the app returns a refusal style response.")

    st.markdown("### Tech stack explained in detail")
    st.write(
        "Python is the core language used to build the full application logic, including document processing, retrieval workflow, "
        "question answering, UI behavior, and field extraction rules."
    )
    st.write(
        "Streamlit is used to build the application interface. It makes it possible to create an interactive AI app quickly with file upload, "
        "buttons, text inputs, status messages, and answer panels."
    )
    st.write(
        "LangChain is used as part of the retrieval and document processing workflow. It helps organize loaders, chunking, retrievers, "
        "and vector search interactions in a cleaner way."
    )
    st.write(
        "FAISS is used as the vector database for similarity search. After the document is split into chunks and embedded, FAISS stores "
        "those vectors and returns the most relevant chunks when a question is asked."
    )
    st.write(
        "Hugging Face models are used for the local language model and embedding pipeline. This makes the project more accessible "
        "without requiring paid API usage during development."
    )
    st.write(
        "FLAN T5 is used for lightweight text generation. It helps generate concise grounded answers from retrieved context."
    )
    st.write(
        "Rule based extraction logic is added for structured documents like passports and bank statements. This is important because "
        "some fields are better handled with deterministic extraction than with pure generation."
    )
    st.write(
        "RAG architecture is what ties everything together. Retrieval gives relevant context, generation creates the answer, and structured "
        "extraction improves reliability for sensitive factual queries."
    )

    st.markdown("### Why this project matters")
    st.write(
        "Many AI document apps look impressive on the surface but fail when asked specific factual questions. This project is designed "
        "to move beyond a simple upload and chat demo by combining document type awareness, field extraction, grounded retrieval, and safer responses."
    )
    st.write("That makes it closer to a practical document intelligence assistant than a generic chatbot.")

    st.markdown("### Supported document types")
    st.markdown("""
    <span class="chip">Resume</span>
    <span class="chip">Passport</span>
    <span class="chip">Bank Statement</span>
    <span class="chip">Driving License</span>
    <span class="chip">Invoice</span>
    <span class="chip">Receipt</span>
    <span class="chip">PDF</span>
    <span class="chip">DOCX</span>
    """, unsafe_allow_html=True)

    st.markdown("### Current strengths")
    st.write(
        "The application handles resumes and passports much better than a basic general purpose document chatbot. "
        "It can correctly refuse unsupported questions, answer many structured questions, and give users suggested questions "
        "based on the detected document type."
    )

    st.markdown("### Current limitations")
    st.write(
        "The project still depends on the quality of extracted text. Blurry scans, OCR noise, poor formatting, and heavily image based documents "
        "can reduce accuracy. Very layout heavy files may require stronger OCR or document parsing methods in future versions."
    )

    st.markdown("### Future improvements")
    st.write(
        "Future improvements can include stronger OCR support, better layout aware parsing, multi document comparison, citation highlighting, "
        "improved bank statement handling, downloadable summaries, and optional hosted LLM integration for stronger general reasoning."
    )

    st.markdown("### Feedback")
    st.markdown(
        '<div class="feedback-note">This project is still being trained. Please ignore any moments of confusion.</div>',
        unsafe_allow_html=True
    )

    with st.form("feedback_form", clear_on_submit=True):
        fb_name = st.text_input("Name")
        fb_email = st.text_input("Email")
        fb_type = st.selectbox(
            "Feedback Type",
            ["General Feedback", "Bug Report", "Suggestion", "Accuracy Issue"]
        )
        fb_message = st.text_area("Your Feedback", height=140)
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            if fb_message.strip():
                save_feedback(fb_name, fb_email, fb_type, fb_message)
                st.success("Thank you. Your feedback has been recorded.")
            else:
                st.warning("Please enter feedback before submitting.")

elif st.session_state.page == "demo":
    st.markdown("""
    <div class="hero-card">
        <div class="project-title">Live Demo</div>
        <div class="project-subtitle">
            Upload a document, let the system detect the type, review the suggested questions,
            and then ask grounded questions based on the uploaded file.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx"])

    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            docs = load_document(uploaded_file)
            chunks = split_documents(docs)
            vectorstore = create_vectorstore(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.all_chunks = chunks

            full_text = get_full_text(chunks)
            st.session_state.doc_type = detect_document_type(full_text)
            st.session_state.known_fields = extract_known_fields(
                full_text,
                st.session_state.doc_type
            )

        st.success("Document processed successfully.")

    if st.session_state.doc_type:
        readable_doc_type = st.session_state.doc_type.replace("_", " ").title()
        st.info(f"Detected document type: {readable_doc_type}")

        suggestions = get_suggested_questions(
            st.session_state.doc_type,
            st.session_state.known_fields
        )

        st.markdown("### Suggested questions")
        for item in suggestions:
            st.markdown(f"• {item}")

    question = st.text_input("Ask a question about the uploaded document:")

    if question and st.session_state.vectorstore:
        with st.spinner("Searching and generating answer..."):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
            relevant_docs = retriever.invoke(question)

            response = answer_question(
                st.session_state.llm,
                relevant_docs,
                question,
                st.session_state.all_chunks
            )

        st.markdown("### Answer")
        st.markdown(f'<div class="answer-box">{response}</div>', unsafe_allow_html=True)

        with st.expander("Retrieved Context"):
            if relevant_docs:
                for i, doc in enumerate(relevant_docs, start=1):
                    st.markdown(f"Chunk {i}")
                    st.write(doc.page_content[:1500])
            else:
                st.write("No relevant context was retrieved.")