import re
from typing import Dict, List

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0.1,
    )
    return HuggingFacePipeline(pipeline=pipe)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def get_full_text(chunks) -> str:
    if not chunks:
        return ""
    return "\n".join([doc.page_content for doc in chunks if getattr(doc, "page_content", None)])


def detect_document_type(text: str) -> str:
    t = text.lower()

    if any(x in t for x in ["passport", "nationality", "date of birth", "place of issue"]):
        return "passport"

    if any(x in t for x in ["education", "experience", "skills", "university", "linkedin", "github"]):
        return "resume"

    if any(x in t for x in ["statement period", "account number", "available balance", "opening balance"]):
        return "bank_statement"

    if any(x in t for x in ["invoice", "bill to", "total amount", "invoice number"]):
        return "invoice"

    if any(x in t for x in ["driving licence", "driver license", "license no", "dl no"]):
        return "driving_license"

    return "general_document"


def extract_resume_email(text: str):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text, re.I)
    return match.group(0) if match else None


def extract_resume_links(text: str):
    links = re.findall(r'(https?://[^\s]+|www\.[^\s]+|linkedin\.com/[^\s]+|github\.com/[^\s]+)', text, re.I)
    cleaned = []
    for link in links:
        value = link[0] if isinstance(link, tuple) else link
        value = value.rstrip(".,);]")
        cleaned.append(value)
    return cleaned


def extract_resume_university(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    university_patterns = [
        r".*university.*",
        r".*institute of technology.*",
        r".*college.*",
        r".*school.*",
    ]

    found = []
    for line in lines:
        low = line.lower()
        for pattern in university_patterns:
            if re.match(pattern, low):
                found.append(line)
                break

    if found:
        return list(dict.fromkeys(found))[:5]
    return []


def extract_resume_skills(text: str):
    known_skills = [
        "Python", "SQL", "Machine Learning", "Deep Learning", "Natural Language Processing",
        "NLP", "Generative AI", "LLM", "LLMs", "RAG", "TensorFlow", "PyTorch", "Scikit-learn",
        "Pandas", "NumPy", "Matplotlib", "Streamlit", "LangChain", "FAISS", "Hugging Face",
        "AWS", "Docker", "Kubernetes", "Git", "GitHub", "Airflow", "Spark", "Tableau",
        "Power BI", "Data Science", "Data Analysis", "Statistics", "ETL", "MLOps"
    ]

    found = []
    lower_text = text.lower()

    for skill in known_skills:
        if skill.lower() in lower_text:
            found.append(skill)

    skill_section = re.search(
        r"(skills|technical skills|core competencies|technologies)(.*?)(education|experience|projects|certifications|$)",
        text,
        re.I | re.S
    )
    if skill_section:
        section_text = skill_section.group(2)
        parts = re.split(r"[,|\n•]+", section_text)
        for part in parts:
            item = part.strip(" :-\t")
            if 2 <= len(item) <= 40:
                found.append(item)

    cleaned = []
    seen = set()
    for item in found:
        item = re.sub(r"\s+", " ", item).strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            cleaned.append(item)

    return cleaned[:25]


def extract_resume_companies(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    companies = []

    known_company_patterns = [
        "broadridge", "evernorth", "massmutual", "aragen", "infosys", "tcs",
        "wipro", "accenture", "amazon", "microsoft", "google", "meta", "ibm",
        "deloitte", "cognizant", "capgemini", "oracle"
    ]

    for line in lines:
        low = line.lower()
        for company in known_company_patterns:
            if company in low:
                companies.append(line)
                break

    experience_block = re.search(
        r"(experience|work experience|professional experience)(.*?)(projects|education|skills|certifications|$)",
        text,
        re.I | re.S
    )
    if experience_block:
        block = experience_block.group(2)
        block_lines = [x.strip() for x in block.splitlines() if x.strip()]
        for line in block_lines:
            if len(line.split()) <= 8 and not any(char.isdigit() for char in line):
                if any(word[0].isupper() for word in line.split() if word):
                    companies.append(line)

    cleaned = []
    seen = set()
    for c in companies:
        c = re.sub(r"\s+", " ", c).strip("•- ")
        if c and c.lower() not in seen and len(c) <= 80:
            seen.add(c.lower())
            cleaned.append(c)

    return cleaned[:15]


def extract_passport_fields(text: str) -> Dict[str, str]:
    data = {}

    passport_match = re.search(r'\b[A-Z][0-9]{7}\b', text)
    if passport_match:
        data["passport_number"] = passport_match.group(0)

    dob_match = re.search(r'(?:date of birth|birth)\s*[:\-]?\s*([0-9]{2}/[0-9]{2}/[0-9]{4})', text, re.I)
    if dob_match:
        data["date_of_birth"] = dob_match.group(1)
    else:
        all_dates = re.findall(r'\b[0-9]{2}/[0-9]{2}/[0-9]{4}\b', text)
        if len(all_dates) >= 1:
            data["date_of_birth"] = all_dates[0]
        if len(all_dates) >= 2:
            data["expiry_date"] = all_dates[-1]

    exp_match = re.search(r'(?:expiry date|date of expiry|expiry)\s*[:\-]?\s*([0-9]{2}/[0-9]{2}/[0-9]{4})', text, re.I)
    if exp_match:
        data["expiry_date"] = exp_match.group(1)

    nat_match = re.search(r'(?:nationality)\s*[:\-]?\s*([A-Z ]{3,})', text, re.I)
    if nat_match:
        data["nationality"] = nat_match.group(1).strip()

    poi_match = re.search(r'(?:place of issue)\s*[:\-]?\s*([A-Z ]{3,})', text, re.I)
    if poi_match:
        data["place_of_issue"] = poi_match.group(1).strip()

    name_match = re.search(r'(?:surname\s*[:\-]?\s*[A-Z]+.*?\n.*?given name[s]?\s*[:\-]?\s*[A-Z ]+)', text, re.I | re.S)
    if name_match:
        block = name_match.group(0)
        surname = re.search(r'surname\s*[:\-]?\s*([A-Z]+)', block, re.I)
        given = re.search(r'given name[s]?\s*[:\-]?\s*([A-Z ]+)', block, re.I)
        if surname and given:
            data["name"] = f"{given.group(1).strip()} {surname.group(1).strip()}".strip()
    else:
        upper_lines = [line.strip() for line in text.splitlines() if line.strip()]
        probable_names = [line for line in upper_lines if line.isupper() and 2 <= len(line.split()) <= 4]
        if probable_names:
            data["name"] = probable_names[0]

    return data


def extract_known_fields(text: str, doc_type: str) -> Dict:
    text = normalize_text(text)

    if doc_type == "passport":
        return extract_passport_fields(text)

    if doc_type == "resume":
        return {
            "email": extract_resume_email(text),
            "links": extract_resume_links(text),
            "universities": extract_resume_university(text),
            "skills": extract_resume_skills(text),
            "companies": extract_resume_companies(text),
        }

    return {}


def get_suggested_questions(doc_type: str, known_fields: Dict) -> List[str]:
    if doc_type == "passport":
        return [
            "What is this document about?",
            "What passport number is mentioned?",
            "What nationality is mentioned?",
            "What date of birth is mentioned?",
            "What is the expiry date?",
            "What place of issue is mentioned?",
            "What name is mentioned?",
        ]

    if doc_type == "resume":
        return [
            "What is this document about?",
            "What university is mentioned?",
            "What skills are mentioned?",
            "What companies are mentioned?",
            "What email address is mentioned?",
        ]

    return [
        "What is this document about?",
        "Summarize this document.",
        "What important information is mentioned?",
    ]


def answer_from_known_fields(question: str, doc_type: str, known_fields: Dict):
    q = question.lower().strip()

    if doc_type == "passport":
        if "passport number" in q and known_fields.get("passport_number"):
            return known_fields["passport_number"]
        if ("nationality" in q or "citizenship" in q) and known_fields.get("nationality"):
            return known_fields["nationality"]
        if ("date of birth" in q or "birth" in q) and known_fields.get("date_of_birth"):
            return known_fields["date_of_birth"]
        if "expiry" in q and known_fields.get("expiry_date"):
            return known_fields["expiry_date"]
        if "place of issue" in q and known_fields.get("place_of_issue"):
            return known_fields["place_of_issue"]
        if "name" in q and known_fields.get("name"):
            return known_fields["name"]
        if "what is this document about" in q or "what is this document" in q:
            return (
                "This document appears to be a passport or identity document. "
                "It contains personal identity details such as name, nationality, passport number, and issue or expiry dates."
            )

    if doc_type == "resume":
        if "email" in q and known_fields.get("email"):
            return known_fields["email"]

        if ("university" in q or "college" in q or "school" in q) and known_fields.get("universities"):
            return ", ".join(known_fields["universities"][:3])

        if "skill" in q and known_fields.get("skills"):
            return ", ".join(known_fields["skills"][:12])

        if ("company" in q or "companies" in q or "organization" in q or "employer" in q) and known_fields.get("companies"):
            return ", ".join(known_fields["companies"][:10])

        if "what is this document about" in q or "what is this document" in q:
            return (
                "This document appears to be a resume. It contains professional information such as education, "
                "skills, experience, projects, and contact details."
            )

    return None


def build_context(relevant_docs) -> str:
    if not relevant_docs:
        return ""
    return "\n\n".join([doc.page_content for doc in relevant_docs if getattr(doc, "page_content", None)])


def answer_question(llm, relevant_docs, question: str, all_chunks=None) -> str:
    full_text = get_full_text(all_chunks if all_chunks else relevant_docs)
    full_text = normalize_text(full_text)

    doc_type = detect_document_type(full_text)
    known_fields = extract_known_fields(full_text, doc_type)

    direct_answer = answer_from_known_fields(question, doc_type, known_fields)
    if direct_answer:
        return direct_answer

    context = build_context(relevant_docs)

    if not context.strip():
        return "I could not verify that confidently from the uploaded document."

    prompt = f"""
You are a document question answering assistant.

Answer the user's question only from the context below.
If the answer is not clearly present in the context, say exactly:
I could not verify that confidently from the uploaded document.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = llm.invoke(prompt).strip()
    except Exception:
        return "I could not verify that confidently from the uploaded document."

    bad_signals = [
        "not mentioned in the context",
        "not provided in the context",
        "cannot be determined",
        "not enough information",
        "i do not know",
        "unknown",
    ]

    if not response:
        return "I could not verify that confidently from the uploaded document."

    if any(signal in response.lower() for signal in bad_signals):
        return "I could not verify that confidently from the uploaded document."

    return response