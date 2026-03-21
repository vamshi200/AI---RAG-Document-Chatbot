import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def get_llm():
    return {"tokenizer": tokenizer, "model": model}


def generate_text(llm, prompt, max_new_tokens=120):
    inputs = llm["tokenizer"](
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = llm["model"].generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True
        )

    response = llm["tokenizer"].decode(outputs[0], skip_special_tokens=True)
    return response.strip()


def safe_not_found():
    return "I could not verify that confidently from the uploaded document."


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def get_full_text(docs, max_docs=20):
    return "\n".join([doc.page_content for doc in docs[:max_docs]])


def detect_document_type(text: str) -> str:
    t = text.lower()

    scores = {
        "resume": 0,
        "passport": 0,
        "bank_statement": 0,
        "driving_license": 0,
        "invoice": 0,
        "receipt": 0,
        "flight_ticket": 0,
        "boarding_pass": 0,
        "travel_itinerary": 0,
        "transcript": 0,
        "contract": 0,
        "meeting_notes": 0,
        "policy_document": 0,
        "book_or_article": 0,
        "general": 0,
    }

    keyword_groups = {
        "resume": [
            "resume", "curriculum vitae", "cv", "employment history",
            "education", "skills", "technical skills", "projects",
            "professional summary", "work experience"
        ],
        "passport": [
            "passport", "passport no", "passport number", "surname",
            "given name", "given name(s)", "nationality", "nationailty",
            "place of birth", "place of issue", "date of issue", "date of expiry"
        ],
        "bank_statement": [
            "bank statement", "statement period", "account number",
            "opening balance", "closing balance", "debit", "credit",
            "transaction", "withdrawal", "deposit", "available balance"
        ],
        "driving_license": [
            "driving licence", "driving license", "licence no", "license no",
            "vehicle class", "date of issue", "valid till", "transport", "non-transport"
        ],
        "invoice": [
            "invoice", "invoice number", "bill to", "amount due",
            "subtotal", "total amount", "due date", "tax"
        ],
        "receipt": [
            "receipt", "payment received", "thank you for your purchase", "transaction id"
        ],
        "flight_ticket": [
            "flight", "pnr", "booking reference", "departure", "arrival", "e-ticket"
        ],
        "boarding_pass": [
            "boarding pass", "gate", "seat", "boarding time", "zone"
        ],
        "travel_itinerary": [
            "itinerary", "reservation", "check in", "check out", "travel plan"
        ],
        "transcript": [
            "transcript", "semester", "gpa", "credits", "course", "grade"
        ],
        "contract": [
            "agreement", "contract", "terms and conditions", "effective date",
            "confidentiality", "termination"
        ],
        "meeting_notes": [
            "meeting notes", "minutes of meeting", "agenda", "action items", "next steps"
        ],
        "policy_document": [
            "policy", "procedure", "scope", "compliance", "guidelines"
        ],
        "book_or_article": [
            "chapter", "author", "publisher", "isbn", "references", "table of contents"
        ],
    }

    for doc_type, keywords in keyword_groups.items():
        for kw in keywords:
            if kw in t:
                scores[doc_type] += 1

    best_type = max(scores, key=scores.get)
    return best_type if scores[best_type] > 0 else "general"


def is_summary_question(question: str) -> bool:
    q = question.lower().strip()
    patterns = [
        "what is this document about",
        "what is the document about",
        "summarize this document",
        "summarize the document",
        "give me a summary",
        "overview of this document",
        "tell me about this document",
        "what does this document contain",
        "what are the key points",
        "main points"
    ]
    return any(pattern in q for pattern in patterns)


def classify_question(question: str) -> str:
    q = question.lower().strip()

    unsupported_patterns = [
        "favorite movie", "favourite movie", "favorite food", "favourite food",
        "hobby", "hobbies", "girlfriend", "boyfriend", "relationship status",
        "religion", "caste", "political", "opinion", "horoscope", "zodiac",
        "how does he feel", "how does she feel", "love", "crush"
    ]
    for pattern in unsupported_patterns:
        if pattern in q:
            return "unsupported"

    if is_summary_question(q):
        return "summary"

    if "passport number" in q or "passport no" in q:
        return "passport_number"

    if "nationality" in q:
        return "nationality"

    if "date of birth" in q or "dob" in q:
        return "dob"

    if "expiry date" in q or "date of expiry" in q or "expires" in q:
        return "expiry_date"

    if "date of issue" in q or "issue date" in q:
        return "issue_date"

    if "place of birth" in q:
        return "place_of_birth"

    if "place of issue" in q:
        return "place_of_issue"

    if "surname" in q:
        return "surname"

    if "given name" in q or "given names" in q:
        return "given_name"

    if "account number" in q:
        return "account_number"

    if "opening balance" in q:
        return "opening_balance"

    if "closing balance" in q:
        return "closing_balance"

    if "available balance" in q:
        return "available_balance"

    if "license number" in q or "licence number" in q or "license no" in q or "licence no" in q:
        return "license_number"

    if "vehicle class" in q:
        return "vehicle_class"

    if "valid till" in q or "validity" in q:
        return "valid_till"

    if (
        "candidate name" in q
        or "name mentioned" in q
        or "what is the name" in q
        or q == "what name is mentioned?"
        or q == "what is the candidate name?"
    ):
        return "name"

    if "email" in q:
        return "email"

    if "phone" in q or "contact number" in q or "mobile number" in q:
        return "phone"

    if "date" in q:
        return "date"

    return "unsupported"


def extract_email(text: str):
    matches = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return matches[0] if matches else None


def extract_phone(text: str):
    matches = re.findall(r"(\+?\d[\d\-\s\(\)]{8,}\d)", text)
    return normalize_spaces(matches[0]) if matches else None


def extract_dates_ddmmyyyy(text: str):
    matches = re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text)
    unique = []
    for d in matches:
        if d not in unique:
            unique.append(d)
    return unique[:20]


def extract_years(text: str):
    matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    unique = []
    for y in matches:
        if y not in unique:
            unique.append(y)
    return unique[:10]


def normalize_passport_ocr_text(text: str) -> str:
    cleaned = text

    cleaned = re.sub(
        r"([A-Z])\s+([0-9])\s+([0-9])\s+([0-9])\s+([0-9])\s+([0-9])\s+([0-9])\s+([0-9])",
        r"\1\2\3\4\5\6\7\8",
        cleaned
    )

    cleaned = re.sub(r"\bI\s*N\s*D\s*I\s*A\s*N\b", "INDIAN", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Given Name\(s\}", "Given Name(s)", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Nationailty", "Nationality", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Passoort", "Passport", cleaned, flags=re.IGNORECASE)

    return cleaned


def extract_passport_fields(text: str):
    result = {}
    cleaned = normalize_passport_ocr_text(text)
    compact = re.sub(r"\s+", "", cleaned)

    mrz_lines = []
    for line in cleaned.splitlines():
        line2 = line.strip().replace(" ", "")
        if line2.startswith("P<") or "<" in line2:
            mrz_lines.append(line2)

    m = re.search(r"Passport No\.?\s*([A-Z][0-9]{7})", cleaned, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"(?:passport\s*(?:no|number)?)[:\s]*([A-Z][0-9]{7})", cleaned, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\b([A-Z][0-9]{7})\b", compact)
    if m:
        result["passport_number"] = m.group(1).strip()

    if "passport_number" not in result:
        m = re.search(r"\b([A-Z][0-9]{7})<", compact)
        if m:
            result["passport_number"] = m.group(1).strip()

    m = re.search(r"(?:Nationality|Nationailty)\s*.*?(INDIAN)", cleaned, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"(?:Nationality|Nationailty).*?([A-Z]{3,})", cleaned, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bINDIAN\b", compact, flags=re.IGNORECASE)
    if not m:
        for line in mrz_lines:
            if line.startswith("P<IND"):
                result["nationality"] = "INDIAN"
                break
    else:
        result["nationality"] = m.group(1).strip().upper()

    m = re.search(r"Surname\s*([A-Z]+)", cleaned, flags=re.IGNORECASE)
    if m:
        result["surname"] = m.group(1).strip().upper()

    m = re.search(r"Given Name\(s\)\s*([A-Z][A-Z]+)", cleaned, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"(?:given name\(s\)|given names|given name)[:\s]*([A-Z][A-Za-z\s]+)", cleaned, flags=re.IGNORECASE)
    if m:
        result["given_name"] = normalize_spaces(m.group(1)).strip().upper()

    if ("surname" not in result or "given_name" not in result) and mrz_lines:
        for line in mrz_lines:
            if line.startswith("P<"):
                m = re.search(r"P<[A-Z]{3}([A-Z<]+)<<([A-Z<]+)", line)
                if m:
                    surname = m.group(1).replace("<", " ").strip()
                    given = m.group(2).replace("<", " ").strip()
                    if surname and "surname" not in result:
                        result["surname"] = surname.upper()
                    if given and "given_name" not in result:
                        result["given_name"] = given.upper()

    m = re.search(r"Date of Birth\s*(\d{2}/\d{2}/\d{4})", cleaned, flags=re.IGNORECASE)
    if m:
        result["dob"] = m.group(1)

    m = re.search(r"Place of Birth\s*([A-Z][A-Za-z,\s]+)", cleaned, flags=re.IGNORECASE)
    if m:
        value = normalize_spaces(m.group(1)).strip(" ,.-")
        value = re.split(r"Place of Issue|Date of Issue|Sex", value, flags=re.IGNORECASE)[0].strip(" ,.-")
        result["place_of_birth"] = value.upper()

    m = re.search(r"Place of Issue\s*([A-Z][A-Za-z,\s]+)", cleaned, flags=re.IGNORECASE)
    if m:
        value = normalize_spaces(m.group(1)).strip(" ,.-")
        value = re.split(r"Date of Issue|Date of Expiry", value, flags=re.IGNORECASE)[0].strip(" ,.-")
        result["place_of_issue"] = value.upper()

    m = re.search(r"Date of Issue\s*(\d{2}/\d{2}/\d{4})", cleaned, flags=re.IGNORECASE)
    if m:
        result["issue_date"] = m.group(1)

    m = re.search(r"Date of Expiry\s*(\d{2}/\d{2}/\d{4})", cleaned, flags=re.IGNORECASE)
    if m:
        result["expiry_date"] = m.group(1)

    return result


def extract_bank_statement_fields(text: str):
    result = {}

    m = re.search(r"(?:account number|a/c number|acct number)[:\s]*([A-Z0-9\-Xx]+)", text, flags=re.IGNORECASE)
    if m:
        result["account_number"] = m.group(1).strip()

    m = re.search(r"(?:opening balance)[:\s₹$]*([0-9,]+(?:\.\d{1,2})?)", text, flags=re.IGNORECASE)
    if m:
        result["opening_balance"] = m.group(1)

    m = re.search(r"(?:closing balance)[:\s₹$]*([0-9,]+(?:\.\d{1,2})?)", text, flags=re.IGNORECASE)
    if m:
        result["closing_balance"] = m.group(1)

    m = re.search(r"(?:available balance|avail balance)[:\s₹$]*([0-9,]+(?:\.\d{1,2})?)", text, flags=re.IGNORECASE)
    if m:
        result["available_balance"] = m.group(1)

    dates = extract_dates_ddmmyyyy(text)
    if dates:
        result["dates"] = dates

    return result


def extract_driving_license_fields(text: str):
    result = {}

    m = re.search(r"(?:license no|licence no|license number|licence number)[:\s]*([A-Z0-9\-\/]+)", text, flags=re.IGNORECASE)
    if m:
        result["license_number"] = m.group(1).strip()

    m = re.search(r"(?:name)[:\s]*([A-Z][A-Za-z\s]+)", text, flags=re.IGNORECASE)
    if m:
        result["name"] = normalize_spaces(m.group(1)).strip()

    m = re.search(r"(?:date of birth|dob)[:\s]*(\d{2}/\d{2}/\d{4})", text, flags=re.IGNORECASE)
    if m:
        result["dob"] = m.group(1)

    m = re.search(r"(?:date of issue|issue date)[:\s]*(\d{2}/\d{2}/\d{4})", text, flags=re.IGNORECASE)
    if m:
        result["issue_date"] = m.group(1)

    m = re.search(r"(?:valid till|validity|expiry date|date of expiry)[:\s]*(\d{2}/\d{2}/\d{4})", text, flags=re.IGNORECASE)
    if m:
        result["valid_till"] = m.group(1)

    m = re.search(r"(?:vehicle class|class)[:\s]*([A-Z,\s\-]+)", text, flags=re.IGNORECASE)
    if m:
        result["vehicle_class"] = normalize_spaces(m.group(1)).strip()

    return result


def extract_known_fields(full_text: str, doc_type: str):
    if doc_type == "passport":
        return extract_passport_fields(full_text)
    if doc_type == "bank_statement":
        return extract_bank_statement_fields(full_text)
    if doc_type == "driving_license":
        return extract_driving_license_fields(full_text)
    return {}


def get_suggested_questions(doc_type: str, known_fields=None):
    known_fields = known_fields or {}

    if doc_type == "passport":
        questions = ["What is this document about?"]
        if known_fields.get("passport_number"):
            questions.append("What passport number is mentioned?")
        if known_fields.get("nationality"):
            questions.append("What nationality is mentioned?")
        if known_fields.get("dob"):
            questions.append("What date of birth is mentioned?")
        if known_fields.get("expiry_date"):
            questions.append("What is the expiry date?")
        if known_fields.get("place_of_issue"):
            questions.append("What place of issue is mentioned?")
        if known_fields.get("given_name") or known_fields.get("surname"):
            questions.append("What name is mentioned?")
        return questions

    if doc_type == "bank_statement":
        questions = ["What is this document about?"]
        if known_fields.get("account_number"):
            questions.append("What is the account number?")
        if known_fields.get("opening_balance"):
            questions.append("What is the opening balance?")
        if known_fields.get("closing_balance"):
            questions.append("What is the closing balance?")
        if known_fields.get("available_balance"):
            questions.append("What is the available balance?")
        if known_fields.get("dates"):
            questions.append("What date is mentioned?")
        return questions

    if doc_type == "driving_license":
        questions = ["What is this document about?"]
        if known_fields.get("license_number"):
            questions.append("What license number is mentioned?")
        if known_fields.get("name"):
            questions.append("What name is mentioned?")
        if known_fields.get("dob"):
            questions.append("What is the date of birth?")
        if known_fields.get("valid_till"):
            questions.append("What is the expiry date?")
        if known_fields.get("vehicle_class"):
            questions.append("What vehicle class is mentioned?")
        return questions

    if doc_type == "resume":
        return [
            "What is this document about?",
            "What university is mentioned?",
            "What skills are mentioned?",
            "What companies are mentioned?",
            "What email address is mentioned?"
        ]

    if doc_type == "invoice":
        return [
            "What is the invoice number?",
            "What is the total amount?",
            "What date is mentioned?"
        ]

    if doc_type == "receipt":
        return [
            "What amount is mentioned?",
            "What date is mentioned?"
        ]

    return [
        "What is this document about?",
        "Summarize this document",
        "What name is mentioned?",
        "What date is mentioned?"
    ]


def summarize_document(text: str, doc_type: str):
    if doc_type == "passport":
        return "This document appears to be a passport or identity document. It contains personal identity details such as name, nationality, passport number, and issue or expiry dates."
    if doc_type == "bank_statement":
        return "This document appears to be a bank statement. It contains account, balance, and transaction-related details."
    if doc_type == "driving_license":
        return "This document appears to be a driving license. It contains identity and driving authorization details."
    if doc_type == "resume":
        return "This document appears to be a resume. It contains education, work experience, projects, and technical skills."
    if doc_type == "invoice":
        return "This document appears to be an invoice. It contains billing and payment details."
    if doc_type == "receipt":
        return "This document appears to be a receipt. It contains purchase and payment details."
    return "This document appears to be a general document."


def answer_rule_based(question: str, full_text: str, doc_type: str):
    question_type = classify_question(question)

    if question_type == "unsupported":
        return safe_not_found()

    if question_type == "summary":
        return summarize_document(full_text, doc_type)

    if doc_type == "passport":
        fields = extract_passport_fields(full_text)

        if question_type == "passport_number":
            return fields.get("passport_number", safe_not_found())
        if question_type == "nationality":
            return fields.get("nationality", safe_not_found())
        if question_type == "dob":
            return fields.get("dob", safe_not_found())
        if question_type == "issue_date":
            return fields.get("issue_date", safe_not_found())
        if question_type == "expiry_date":
            return fields.get("expiry_date", safe_not_found())
        if question_type == "place_of_birth":
            return fields.get("place_of_birth", safe_not_found())
        if question_type == "place_of_issue":
            return fields.get("place_of_issue", safe_not_found())
        if question_type == "surname":
            return fields.get("surname", safe_not_found())
        if question_type == "given_name":
            return fields.get("given_name", safe_not_found())
        if question_type == "name":
            given = fields.get("given_name")
            surname = fields.get("surname")
            if given and surname:
                return f"{given} {surname}"
            return given or surname or safe_not_found()

    if doc_type == "bank_statement":
        fields = extract_bank_statement_fields(full_text)
        if question_type == "account_number":
            return fields.get("account_number", safe_not_found())
        if question_type == "opening_balance":
            return fields.get("opening_balance", safe_not_found())
        if question_type == "closing_balance":
            return fields.get("closing_balance", safe_not_found())
        if question_type == "available_balance":
            return fields.get("available_balance", safe_not_found())
        if question_type == "date":
            return ", ".join(fields.get("dates", [])[:5]) if fields.get("dates") else safe_not_found()

    if doc_type == "driving_license":
        fields = extract_driving_license_fields(full_text)
        if question_type == "license_number":
            return fields.get("license_number", safe_not_found())
        if question_type == "name":
            return fields.get("name", safe_not_found())
        if question_type == "dob":
            return fields.get("dob", safe_not_found())
        if question_type == "issue_date":
            return fields.get("issue_date", safe_not_found())
        if question_type == "expiry_date":
            return fields.get("valid_till", safe_not_found())
        if question_type == "vehicle_class":
            return fields.get("vehicle_class", safe_not_found())

    if question_type == "email":
        return extract_email(full_text) or safe_not_found()

    if question_type == "phone":
        return extract_phone(full_text) or safe_not_found()

    if question_type == "date":
        dates = extract_dates_ddmmyyyy(full_text)
        if dates:
            return ", ".join(dates[:5])
        years = extract_years(full_text)
        return ", ".join(years[:5]) if years else safe_not_found()

    return safe_not_found()


def answer_question(llm, docs, question, all_chunks=None):
    source_docs = all_chunks if all_chunks else docs
    full_text = get_full_text(source_docs)
    doc_type = detect_document_type(full_text)
    return answer_rule_based(question, full_text, doc_type)