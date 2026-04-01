from flask import Flask, render_template, request, send_file
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from waitress import serve
import smtplib
import json
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import os
import sys
import subprocess
from dotenv import load_dotenv
from urllib.parse import quote
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import time
from collections import Counter
import re
import math
from io import BytesIO
import shutil
import uuid
from html import escape, unescape
import xml.etree.ElementTree as ET

try:
    import pytesseract
    from pdf2image import convert_from_bytes
except Exception:
    pytesseract = None
    convert_from_bytes = None

colors = None
A4 = None
ParagraphStyle = None
getSampleStyleSheet = None
Paragraph = None
SimpleDocTemplate = None
Spacer = None
Table = None
TableStyle = None
FPDF = None
CV_EXPORT_IMPORT_ERROR = None
CV_EXPORT_FALLBACK_IMPORT_ERROR = None

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

# ---- App Initialization ----
app = Flask(__name__)

# Load environment variables
load_dotenv()

# ---- Configuration (Use Environment Variables for better security) ----
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "").strip()
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "").strip()
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

OCR_AVAILABLE = bool(
    pytesseract is not None
    and convert_from_bytes is not None
    and shutil.which("tesseract")
    and shutil.which("pdftoppm")
)
CV_EXPORT_AVAILABLE = False
CV_EXPORT_BOOTSTRAP_ATTEMPTED = False
CV_EXPORT_BOOTSTRAP_ERROR = None
CV_EXPORT_STORE = {}
CV_EXPORT_TTL_SECONDS = 60 * 60
JOB_ALERTS_FILE_PATH = os.getenv("JOB_ALERTS_FILE_PATH", "data/job_alerts.json").strip() or "data/job_alerts.json"
JOB_ALERTS_LOCK = threading.Lock()
JOB_ALERT_RUN_TOKEN = os.getenv("JOB_ALERT_RUN_TOKEN", "").strip()
DEFAULT_ALERT_MAX_ITEMS = 12

# Job API Configuration (Optional - Get free keys from respective platforms)
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")  # Get from https://developer.adzuna.com/
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")  # Get from https://rapidapi.com/

JOB_SEARCH_LOCATION = os.getenv("JOB_SEARCH_LOCATION", "India").strip() or "India"
JOB_SEARCH_REGION = os.getenv("JOB_SEARCH_REGION", "in-en").strip() or "in-en"

try:
    JOB_RECENCY_DAYS = max(1, int(os.getenv("JOB_RECENCY_DAYS", "7")))
except ValueError:
    JOB_RECENCY_DAYS = 7

ENABLE_LIVE_JOB_SEARCH = os.getenv("ENABLE_LIVE_JOB_SEARCH", "false").strip().lower() in {"1", "true", "yes"}
LIVE_JOB_SEARCH_AVAILABLE = DDGS is not None and ENABLE_LIVE_JOB_SEARCH

try:
    JOB_FETCH_TOTAL_TIMEOUT_SECONDS = max(2.0, float(os.getenv("JOB_FETCH_TOTAL_TIMEOUT_SECONDS", "9")))
except ValueError:
    JOB_FETCH_TOTAL_TIMEOUT_SECONDS = 9.0

try:
    JOB_FETCH_PER_REQUEST_TIMEOUT_SECONDS = max(0.75, float(os.getenv("JOB_FETCH_PER_REQUEST_TIMEOUT_SECONDS", "3")))
except ValueError:
    JOB_FETCH_PER_REQUEST_TIMEOUT_SECONDS = 3.0

try:
    JOB_QUERY_VARIANTS_PER_SOURCE = max(1, int(os.getenv("JOB_QUERY_VARIANTS_PER_SOURCE", "1")))
except ValueError:
    JOB_QUERY_VARIANTS_PER_SOURCE = 1

def load_cv_export_backends():
    global colors, A4, ParagraphStyle, getSampleStyleSheet, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    global FPDF, CV_EXPORT_IMPORT_ERROR, CV_EXPORT_FALLBACK_IMPORT_ERROR, CV_EXPORT_AVAILABLE

    try:
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.pagesizes import A4 as rl_A4
        from reportlab.lib.styles import ParagraphStyle as rl_ParagraphStyle, getSampleStyleSheet as rl_getSampleStyleSheet
        from reportlab.platypus import Paragraph as rl_Paragraph, SimpleDocTemplate as rl_SimpleDocTemplate, Spacer as rl_Spacer, Table as rl_Table, TableStyle as rl_TableStyle

        colors = rl_colors
        A4 = rl_A4
        ParagraphStyle = rl_ParagraphStyle
        getSampleStyleSheet = rl_getSampleStyleSheet
        Paragraph = rl_Paragraph
        SimpleDocTemplate = rl_SimpleDocTemplate
        Spacer = rl_Spacer
        Table = rl_Table
        TableStyle = rl_TableStyle
        CV_EXPORT_IMPORT_ERROR = None
    except Exception as error:
        colors = None
        A4 = None
        ParagraphStyle = None
        getSampleStyleSheet = None
        Paragraph = None
        SimpleDocTemplate = None
        Spacer = None
        Table = None
        TableStyle = None
        CV_EXPORT_IMPORT_ERROR = str(error)

    try:
        from fpdf import FPDF as fpdf_cls
        FPDF = fpdf_cls
        CV_EXPORT_FALLBACK_IMPORT_ERROR = None
    except Exception as error:
        FPDF = None
        CV_EXPORT_FALLBACK_IMPORT_ERROR = str(error)

    CV_EXPORT_AVAILABLE = bool(SimpleDocTemplate or FPDF)

def ensure_cv_export_backends():
    global CV_EXPORT_BOOTSTRAP_ATTEMPTED, CV_EXPORT_BOOTSTRAP_ERROR

    if CV_EXPORT_AVAILABLE:
        return

    if CV_EXPORT_BOOTSTRAP_ATTEMPTED:
        return

    CV_EXPORT_BOOTSTRAP_ATTEMPTED = True
    target_dir = "/tmp/ai_career_platform_pdf_deps"
    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--target",
        target_dir,
        "reportlab==4.2.5",
        "fpdf2==2.8.1",
    ]

    try:
        subprocess.run(
            install_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=90,
        )

        if target_dir not in sys.path:
            sys.path.insert(0, target_dir)

        load_cv_export_backends()
        if CV_EXPORT_AVAILABLE:
            CV_EXPORT_BOOTSTRAP_ERROR = None
            return

        CV_EXPORT_BOOTSTRAP_ERROR = "Runtime install completed but PDF imports are still unavailable."
    except Exception as error:
        CV_EXPORT_BOOTSTRAP_ERROR = str(error)

load_cv_export_backends()

# ---- Skill Master List ----
skill_keywords = [
    "python", "java", "c++", "c#", "javascript", "typescript",
    "html", "css", "react", "angular", "vue", "node.js",
    "express", "flask", "django", "fastapi", "spring", "sql",
    "mysql", "postgresql", "mongodb", "redis", "aws", "azure",
    "gcp", "docker", "kubernetes", "git", "linux", "rest api",
    "graphql", "machine learning", "deep learning", "nlp",
    "data science", "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "tableau", "power bi"
]

# ---- Job Roles and Required Skills ----
job_roles = {
    "Data Scientist": ["python", "machine learning", "data science", "sql", "pandas"],
    "Data Analyst": ["sql", "python", "tableau", "power bi", "excel"],
    "Frontend Developer": ["html", "css", "javascript", "react", "typescript"],
    "Backend Developer": ["python", "java", "sql", "rest api", "docker"],
    "Software Engineer": ["python", "java", "c++", "sql", "git"],
    "Full Stack Developer": ["html", "css", "javascript", "react", "python", "sql"],
    "Machine Learning Engineer": ["python", "machine learning", "deep learning", "pytorch"],
    "DevOps Engineer": ["aws", "docker", "kubernetes", "linux", "git"]
}

SECTION_PATTERNS = {
    "summary": ["professional summary", "summary", "profile", "objective"],
    "skills": ["technical skills", "skills", "core competencies"],
    "experience": ["professional experience", "work experience", "experience", "employment history"],
    "education": ["education", "academic background"],
    "projects": ["projects", "project experience"],
    "certifications": ["certifications", "certificates", "licenses"]
}

SECTION_WEIGHTS = {
    "summary": 3,
    "skills": 4,
    "experience": 5,
    "education": 4,
    "projects": 2,
    "certifications": 2
}

ACTION_VERBS = {
    "built", "created", "delivered", "designed", "developed", "drove",
    "implemented", "improved", "launched", "led", "managed", "optimized",
    "reduced", "streamlined", "automated", "analyzed"
}

IMPACT_TERMS = {
    "increased", "improved", "reduced", "optimized", "saved", "grew",
    "launched", "automated", "accelerated", "boosted"
}

GENERIC_JOB_TERMS = {
    "ability", "candidate", "communication", "company", "customer", "degree",
    "engineer", "engineering", "excellent", "experience", "familiarity",
    "knowledge", "looking", "preferred", "requirements", "responsibilities",
    "role", "skills", "strong", "team", "work", "working", "years"
}

COMMON_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves"
}

EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}")
YEAR_REGEX = re.compile(r"\b([1-9]\d?)\+?\s*(?:years?|yrs?)\b", re.IGNORECASE)
METRIC_REGEX = re.compile(
    r"(?:\b\d+(?:\.\d+)?%|\$\d[\d,]*(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:users|clients|projects|features|months|weeks|years|hours|days|percent))",
    re.IGNORECASE,
)
URL_REGEX = re.compile(r"(?:https?://|www\.)[^\s]+", re.IGNORECASE)

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def predict_selection_probability(skills_count, experience_years):
    bounded_skills = min(max(skills_count, 0), 10)
    bounded_experience = min(max(experience_years, 0), 4)
    linear_score = -2.8 + (0.42 * bounded_skills) + (0.85 * bounded_experience)
    probability = 1 / (1 + math.exp(-linear_score))
    return round(probability * 100, 2)

def send_feedback_email(receiver_email, ats_score, message, skills):
    """Sends analysis results to the user's email securely."""
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        return False

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = receiver_email
    msg['Subject'] = "Your AI Resume Analysis Results"

    body = f"""
    Hello,

    Thank you for using our AI Career Ecosystem. Here is your resume feedback:

    ATS Score: {ats_score}/100
    Status: {message}
    Extracted Skills: {', '.join(skills)}

    Keep improving your skills!
    """
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=8)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Email Error: {e}")
        return False

def send_job_alert_email(receiver_email, subscription, new_jobs):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        return False, "Job alerts email is disabled because SMTP credentials are not configured."

    if not receiver_email:
        return False, "Missing receiver email for job alert delivery."

    frequency = subscription.get("frequency", "weekly")
    location = subscription.get("location", JOB_SEARCH_LOCATION)
    skills = subscription.get("skills", [])
    min_salary = subscription.get("min_salary")
    max_salary = subscription.get("max_salary")

    salary_range_text = "Any"
    if min_salary is not None or max_salary is not None:
        min_txt = str(min_salary) if min_salary is not None else "0"
        max_txt = str(max_salary) if max_salary is not None else "Any"
        salary_range_text = f"{min_txt} - {max_txt}"

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email
    msg["Subject"] = f"Your {frequency.title()} Job Alerts ({len(new_jobs)} new matches)"

    lines = [
        "Hello,",
        "",
        "Here are your latest job alert matches from AI Career Platform:",
        f"Location: {location}",
        f"Skills: {', '.join(skills) if skills else 'General profile matching'}",
        f"Salary Range: {salary_range_text}",
        "",
    ]

    for index, job in enumerate(new_jobs[:DEFAULT_ALERT_MAX_ITEMS], start=1):
        lines.extend(
            [
                f"{index}. {job.get('title', 'Role')} at {job.get('company', 'Unknown Company')}",
                f"   Source: {job.get('source', 'Live Jobs')} | Location: {job.get('location', 'Not specified')} | Posted: {job.get('posted', 'Recent')}",
                f"   Salary: {job.get('salary', 'Not listed')}",
                f"   Link: {job.get('url', '#')}",
                "",
            ]
        )

    lines.extend(
        [
            "Tip: Continue improving your resume based on ATS suggestions to get stronger matches.",
            "",
            "Best regards,",
            "AI Career Platform",
        ]
    )

    msg.attach(MIMEText("\n".join(lines), "plain"))

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=8)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True, "Job alert email sent."
    except Exception as error:
        print(f"Job alert email error: {error}")
        return False, f"Email delivery failed: {error}"

def email_feedback_enabled():
    return bool(SENDER_EMAIL and SENDER_PASSWORD)

def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip().lower()

def clean_url(url):
    return (url or "").rstrip(".,;)")

def contains_normalized_term(normalized_text, term):
    if not normalized_text or not term:
        return False

    normalized_term = normalize_text(term)
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None

def deadline_exceeded(deadline):
    return deadline is not None and time.monotonic() >= deadline

def bounded_request_timeout(deadline):
    if deadline is None:
        return JOB_FETCH_PER_REQUEST_TIMEOUT_SECONDS

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return 0

    return max(0.5, min(JOB_FETCH_PER_REQUEST_TIMEOUT_SECONDS, remaining))

def extract_ocr_text_from_pdf_bytes(pdf_bytes):
    if not OCR_AVAILABLE:
        return [], []

    try:
        images = convert_from_bytes(pdf_bytes, dpi=230)
    except Exception as error:
        print(f"OCR Conversion Error: {error}")
        return [], []

    text_parts = []
    extracted_pages = []

    for index, image in enumerate(images, start=1):
        try:
            page_text = (pytesseract.image_to_string(image) or "").strip()
        except Exception as error:
            print(f"OCR Page Error ({index}): {error}")
            continue

        if page_text:
            text_parts.append(page_text)
            extracted_pages.append(
                {
                    "page": index,
                    "characters": len(page_text),
                    "words": len(page_text.split()),
                }
            )

    return text_parts, extracted_pages

def extract_pdf_text_details(pdf):
    stream = getattr(pdf, "stream", pdf)

    try:
        stream.seek(0)
    except Exception:
        pass

    pdf_bytes = stream.read()
    if not pdf_bytes:
        raise ValueError("The uploaded PDF appears to be empty.")

    try:
        stream.seek(0)
    except Exception:
        pass

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except PdfReadError as error:
        raise ValueError("The uploaded file could not be read as a valid PDF.") from error

    text_parts = []
    extracted_pages = []

    for index, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            text_parts.append(page_text)
            extracted_pages.append(
                {
                    "page": index,
                    "characters": len(page_text),
                    "words": len(page_text.split()),
                }
            )

    text_source = "digital"

    if not text_parts:
        ocr_text_parts, ocr_page_stats = extract_ocr_text_from_pdf_bytes(pdf_bytes)
        if ocr_text_parts:
            text_parts = ocr_text_parts
            extracted_pages = ocr_page_stats
            text_source = "ocr"
        else:
            text_source = "none"

    raw_text = "\n".join(text_parts).strip()
    normalized_text = normalize_text(raw_text)

    return {
        "raw_text": raw_text,
        "text": normalized_text,
        "page_count": len(reader.pages),
        "extractable_pages": len(extracted_pages),
        "page_stats": extracted_pages,
        "text_source": text_source,
        "ocr_used": text_source == "ocr",
        "ocr_available": OCR_AVAILABLE,
        "has_text": bool(re.search(r"[A-Za-z0-9]", raw_text)),
    }

def extract_skills_nlp(text):
    normalized_text = normalize_text(text)
    extracted = [
        skill for skill in skill_keywords
        if contains_normalized_term(normalized_text, skill)
    ]
    return sorted(set(extracted))

def extract_resume_sections(text):
    normalized_text = normalize_text(text)
    detected_sections = []

    for section, patterns in SECTION_PATTERNS.items():
        if any(contains_normalized_term(normalized_text, pattern) for pattern in patterns):
            detected_sections.append(section)

    return detected_sections

def extract_contact_signals(text):
    normalized_text = normalize_text(text)
    email = bool(EMAIL_REGEX.search(text or ""))
    phone = bool(PHONE_REGEX.search(text or ""))
    linkedin = "linkedin.com" in normalized_text
    github = "github.com" in normalized_text
    website = bool(re.search(r"https?://|www\.", normalized_text))

    return {
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
        "website": website,
    }

def extract_top_terms(text, limit=8):
    stop_words = COMMON_STOP_WORDS | GENERIC_JOB_TERMS
    counts = Counter()

    for token in re.findall(r"[a-zA-Z][a-zA-Z+#.]{2,}", normalize_text(text)):
        if token in stop_words:
            continue
        counts[token] += 1

    return [term for term, _ in counts.most_common(limit)]

def extract_target_keywords(job_desc, limit=12):
    normalized_desc = normalize_text(job_desc)
    target_skills = extract_skills_nlp(normalized_desc)
    counts = Counter()

    for token in re.findall(r"[a-zA-Z][a-zA-Z+#.]{2,}", normalized_desc):
        if token in GENERIC_JOB_TERMS or token in COMMON_STOP_WORDS:
            continue
        counts[token] += 1

    target_keywords = []
    for keyword in target_skills + [term for term, _ in counts.most_common(limit)]:
        if keyword not in target_keywords:
            target_keywords.append(keyword)

    return target_keywords[:limit]

def read_job_alert_store():
    with JOB_ALERTS_LOCK:
        if not os.path.exists(JOB_ALERTS_FILE_PATH):
            return {"subscriptions": []}

        try:
            with open(JOB_ALERTS_FILE_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict):
                return {"subscriptions": []}
            if "subscriptions" not in data or not isinstance(data["subscriptions"], list):
                data["subscriptions"] = []
            return data
        except Exception as error:
            print(f"Job alert store read error: {error}")
            return {"subscriptions": []}

def write_job_alert_store(store):
    with JOB_ALERTS_LOCK:
        os.makedirs(os.path.dirname(JOB_ALERTS_FILE_PATH) or ".", exist_ok=True)
        with open(JOB_ALERTS_FILE_PATH, "w", encoding="utf-8") as file:
            json.dump(store, file, indent=2)

def parse_int_or_none(value):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None

def extract_salary_bounds(salary_text):
    text = normalize_text(salary_text)
    values = [int(number) for number in re.findall(r"\d{2,7}", text)]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    return min(values), max(values)

def job_matches_salary_range(job, min_salary=None, max_salary=None):
    if min_salary is None and max_salary is None:
        return True

    low, high = extract_salary_bounds(job.get("salary", ""))
    if low is None and high is None:
        return False

    if min_salary is not None and high is not None and high < min_salary:
        return False
    if max_salary is not None and low is not None and low > max_salary:
        return False
    return True

def job_matches_location(job, expected_location):
    expected = normalize_text(expected_location)
    if not expected:
        return True

    job_location = normalize_text(job.get("location", ""))
    if not job_location:
        return False
    return expected in job_location or job_location in expected

def frequency_to_days(frequency):
    return 1 if frequency == "daily" else 7

def current_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def utc_now_timestamp():
    return datetime.now(timezone.utc).timestamp()

def iso_to_utc_timestamp(iso_value):
    if not iso_value:
        return 0
    try:
        return datetime.fromisoformat(iso_value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0

def save_or_update_job_alert_subscription(email, skills, location, min_salary, max_salary, frequency):
    normalized_email = (email or "").strip().lower()
    if not normalized_email:
        return False, "Email is required for job alerts."

    if frequency not in {"daily", "weekly"}:
        frequency = "weekly"

    sanitized_skills = unique_trimmed(skills or [], limit=12)
    now_iso = current_utc_iso()
    next_run_ts = utc_now_timestamp() + (frequency_to_days(frequency) * 86400)
    next_run_iso = datetime.fromtimestamp(next_run_ts, tz=timezone.utc).isoformat()

    store = read_job_alert_store()
    subscriptions = store.get("subscriptions", [])

    updated = False
    for entry in subscriptions:
        if normalize_text(entry.get("email", "")) != normalize_text(normalized_email):
            continue

        entry["skills"] = sanitized_skills
        entry["location"] = location or JOB_SEARCH_LOCATION
        entry["min_salary"] = min_salary
        entry["max_salary"] = max_salary
        entry["frequency"] = frequency
        entry["updated_at"] = now_iso
        entry["next_run_at"] = next_run_iso
        entry.setdefault("seen_job_urls", [])
        updated = True
        break

    if not updated:
        subscriptions.append(
            {
                "id": uuid.uuid4().hex,
                "email": normalized_email,
                "skills": sanitized_skills,
                "location": location or JOB_SEARCH_LOCATION,
                "min_salary": min_salary,
                "max_salary": max_salary,
                "frequency": frequency,
                "created_at": now_iso,
                "updated_at": now_iso,
                "last_sent_at": "",
                "next_run_at": next_run_iso,
                "seen_job_urls": [],
            }
        )

    store["subscriptions"] = subscriptions
    write_job_alert_store(store)
    action = "updated" if updated else "created"
    return True, f"Job alert subscription {action} successfully ({frequency})."

def build_interview_prep_assistant(analysis, job_matches, job_recommendations, max_companies=3):
    cv_payload = build_cv_payload(analysis, "", "", job_matches)
    candidate_name = cv_payload.get("name", "Candidate")
    headline = cv_payload.get("headline", "Technology Professional")
    skills = analysis.get("skills", [])
    top_skills = skills[:4] if skills else ["problem solving", "communication", "software engineering"]
    project_summary = cv_payload.get("project_summary", "relevant projects")
    experience_summary = cv_payload.get("experience_summary", "practical implementation work")

    target_companies = []
    seen = set()
    for job in job_recommendations or []:
        company = clean_company_name(job.get("company"))
        key = normalize_text(company)
        if not key or key in seen:
            continue
        seen.add(key)
        target_companies.append(
            {
                "company": company,
                "role": title_from_job_entry(job),
                "source": job.get("source", "Live Jobs"),
                "job_url": clean_url(job.get("url", "")),
            }
        )
        if len(target_companies) >= max_companies:
            break

    if not target_companies:
        fallback_role = job_matches[0][0] if job_matches else headline
        target_companies.append(
            {
                "company": "Target Company",
                "role": fallback_role,
                "source": "Resume Alignment",
                "job_url": "",
            }
        )

    prep_sets = []
    for item in target_companies:
        company = item["company"]
        role = item["role"]

        questions = [
            {
                "question": f"Tell me about yourself and why you are a fit for the {role} role at {company}.",
                "model_answer": (
                    f"I am {candidate_name}, a {headline} with strengths in {', '.join(top_skills[:3])}. "
                    f"Recently, I worked on {project_summary} and contributed through {experience_summary}. "
                    f"I am excited about this {role} role because it matches my background and growth direction."
                ),
            },
            {
                "question": f"Why do you want to join {company}?",
                "model_answer": (
                    f"I want to join {company} because of its reputation for impactful work and strong engineering standards. "
                    "I value teams that prioritize ownership, learning, and measurable outcomes. "
                    f"I believe I can contribute quickly in a {role} role while continuing to grow."
                ),
            },
            {
                "question": f"Describe a project where you used {top_skills[0]} to solve a difficult problem.",
                "model_answer": (
                    f"In one project, I used {top_skills[0]} to address a high-priority issue affecting delivery quality. "
                    "I analyzed the root cause, implemented a clear solution path, and validated results with measurable checks. "
                    "The result was improved reliability and faster execution for the team."
                ),
            },
            {
                "question": "Tell me about a time you handled ambiguity or conflicting priorities.",
                "model_answer": (
                    "I handled ambiguity by clarifying objectives early, breaking work into milestones, and communicating risks quickly. "
                    "I aligned stakeholders on priorities and delivered in phases, which reduced uncertainty and kept progress visible."
                ),
            },
            {
                "question": "How do you ensure quality before shipping your work?",
                "model_answer": (
                    "I use a quality checklist that includes requirement validation, edge-case testing, and peer review where possible. "
                    "I also monitor outcomes after release and iterate quickly on feedback."
                ),
            },
        ]

        scoring = [
            {"criterion": "Role relevance", "weight": 25, "what_good_looks_like": "Answer directly maps experience to role requirements."},
            {"criterion": "Structure and clarity", "weight": 20, "what_good_looks_like": "Response is concise, logical, and easy to follow."},
            {"criterion": "Impact evidence", "weight": 25, "what_good_looks_like": "Includes measurable outcomes or concrete results."},
            {"criterion": "Technical depth", "weight": 20, "what_good_looks_like": "Demonstrates practical decisions, trade-offs, and tools."},
            {"criterion": "Communication confidence", "weight": 10, "what_good_looks_like": "Tone is professional, calm, and action-oriented."},
        ]

        prep_sets.append(
            {
                "company": company,
                "role": role,
                "source": item.get("source", "Live Jobs"),
                "job_url": item.get("job_url", ""),
                "questions": questions,
                "mock_scoring": scoring,
                "self_practice_tip": "Use STAR format and keep each answer between 60-120 seconds.",
            }
        )

    return prep_sets

def run_due_job_alerts(limit=None):
    store = read_job_alert_store()
    subscriptions = store.get("subscriptions", [])
    now_ts = utc_now_timestamp()
    processed = 0
    sent = 0
    skipped = 0
    results = []

    for entry in subscriptions:
        if limit is not None and processed >= limit:
            break

        next_run_ts = iso_to_utc_timestamp(entry.get("next_run_at", ""))
        if next_run_ts and next_run_ts > now_ts:
            skipped += 1
            continue

        processed += 1
        skills = entry.get("skills", [])
        role_matches = match_job_roles(skills)
        jobs = aggregate_job_recommendations(skills, role_matches, max_per_source=8)

        location = entry.get("location") or JOB_SEARCH_LOCATION
        min_salary = entry.get("min_salary")
        max_salary = entry.get("max_salary")
        seen_urls = set(entry.get("seen_job_urls", []))

        filtered = []
        for job in jobs:
            normalized_url = normalize_job_url(job.get("url", ""))
            if normalized_url in seen_urls:
                continue
            if not job_matches_location(job, location):
                continue
            if not job_matches_salary_range(job, min_salary=min_salary, max_salary=max_salary):
                continue
            filtered.append(job)

        if not filtered:
            entry["next_run_at"] = datetime.fromtimestamp(
                now_ts + (frequency_to_days(entry.get("frequency", "weekly")) * 86400),
                tz=timezone.utc,
            ).isoformat()
            entry["updated_at"] = current_utc_iso()
            results.append({"email": entry.get("email", ""), "status": "no_new_matches", "sent": 0})
            continue

        sent_ok, status_message = send_job_alert_email(entry.get("email", ""), entry, filtered)
        results.append(
            {
                "email": entry.get("email", ""),
                "status": "sent" if sent_ok else "email_failed",
                "sent": len(filtered[:DEFAULT_ALERT_MAX_ITEMS]),
                "message": status_message,
            }
        )

        if sent_ok:
            sent += 1
            new_seen = [normalize_job_url(job.get("url", "")) for job in filtered[:DEFAULT_ALERT_MAX_ITEMS]]
            entry["seen_job_urls"] = list(dict.fromkeys((entry.get("seen_job_urls", []) + new_seen)))[-500:]
            entry["last_sent_at"] = current_utc_iso()

        entry["next_run_at"] = datetime.fromtimestamp(
            now_ts + (frequency_to_days(entry.get("frequency", "weekly")) * 86400),
            tz=timezone.utc,
        ).isoformat()
        entry["updated_at"] = current_utc_iso()

    store["subscriptions"] = subscriptions
    write_job_alert_store(store)

    return {
        "processed": processed,
        "sent": sent,
        "skipped": skipped,
        "results": results,
    }

def keyword_match_score(resume_text, job_desc, keywords):
    normalized_resume = normalize_text(resume_text)
    normalized_job_desc = normalize_text(job_desc)
    matched = [
        keyword for keyword in keywords
        if contains_normalized_term(normalized_resume, keyword)
        and contains_normalized_term(normalized_job_desc, keyword)
    ]

    if not keywords:
        return 0

    return (len(matched) / len(keywords)) * 35

def skills_section_score(sections):
    return min(sum(SECTION_WEIGHTS.get(section, 0) for section in sections), 20)

def contact_score(contact_signals):
    score = 0
    if contact_signals["email"]:
        score += 3
    if contact_signals["phone"]:
        score += 3
    if contact_signals["linkedin"]:
        score += 2
    if contact_signals["github"] or contact_signals["website"]:
        score += 2
    return min(score, 10)

def estimate_experience_years(resume_text):
    matches = [int(years) for years in YEAR_REGEX.findall(resume_text or "")]
    return max(matches) if matches else 0

def experience_score(resume_text, sections):
    normalized_text = normalize_text(resume_text)
    score = 0
    action_hits = sorted({verb for verb in ACTION_VERBS if contains_normalized_term(normalized_text, verb)})

    if estimate_experience_years(resume_text):
        score += 4
    if "experience" in sections:
        score += 4
    if any(term in normalized_text for term in ["project", "internship", "client", "deployment"]):
        score += 4
    if len(action_hits) >= 3:
        score += 3
    elif action_hits:
        score += 2

    return min(score, 15), action_hits

def achievement_score(resume_text):
    normalized_text = normalize_text(resume_text)
    metrics = METRIC_REGEX.findall(resume_text or "")
    impact_hits = [term for term in IMPACT_TERMS if contains_normalized_term(normalized_text, term)]

    score = min(len(metrics), 3) * 3
    if impact_hits:
        score += 1

    return min(score, 10), metrics[:5]

def resume_length_score(raw_resume_text, normalized_resume_text):
    words = len(normalized_resume_text.split())
    sentences = [segment.strip() for segment in re.split(r"[.!?]+", raw_resume_text or "") if segment.strip()]
    sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence]
    avg_sentence_length = round(sum(sentence_lengths) / len(sentence_lengths), 2) if sentence_lengths else 0
    non_empty_lines = [line for line in (raw_resume_text or "").splitlines() if line.strip()]
    has_list_structure = any(line.lstrip().startswith(("-", "*", "•")) for line in non_empty_lines) or len(non_empty_lines) >= 8

    score = 0
    if 250 <= words <= 900:
        score += 5
    elif 150 <= words <= 1100:
        score += 3
    elif words > 0:
        score += 1

    if 8 <= avg_sentence_length <= 30:
        score += 3
    elif 5 <= avg_sentence_length <= 40:
        score += 2

    if has_list_structure:
        score += 2

    return min(score, 10), words, avg_sentence_length

def job_alignment_score(resume_text, skills, job_desc):
    normalized_resume = normalize_text(resume_text)

    if job_desc:
        target_keywords = extract_target_keywords(job_desc)
        target_skills = extract_skills_nlp(job_desc)
        matched_keywords = [
            keyword for keyword in target_keywords
            if contains_normalized_term(normalized_resume, keyword)
        ]
        missing_keywords = [
            keyword for keyword in target_keywords
            if keyword not in matched_keywords
        ]
        matched_skills = [skill for skill in target_skills if skill in matched_keywords]

        keyword_ratio = len(matched_keywords) / len(target_keywords) if target_keywords else 0
        skill_ratio = len(matched_skills) / len(target_skills) if target_skills else keyword_ratio
        score = (skill_ratio * 20) + (keyword_ratio * 15)

        return {
            "score": min(round(score, 2), 35),
            "mode": "targeted",
            "target_keywords": target_keywords,
            "matched_keywords": matched_keywords,
            "missing_keywords": missing_keywords,
        }

    role_matches = match_job_roles(skills)
    if not role_matches:
        return {
            "score": 0,
            "mode": "general",
            "target_keywords": [],
            "matched_keywords": [],
            "missing_keywords": [],
        }

    top_role, details = role_matches[0]
    score = (details["score"] / 100) * 25 + (min(len(skills), 8) / 8) * 10

    return {
        "score": min(round(score, 2), 35),
        "mode": "general",
        "target_role": top_role,
        "target_keywords": details["matched_skills"] + details["missing_skills"],
        "matched_keywords": details["matched_skills"],
        "missing_keywords": details["missing_skills"],
    }

def calculate_ats_score(resume_text, raw_resume_text, skills, job_desc):
    sections = extract_resume_sections(raw_resume_text)
    contact_signals = extract_contact_signals(raw_resume_text)
    alignment = job_alignment_score(resume_text, skills, job_desc)
    section_score = skills_section_score(sections)
    contact_details_score = contact_score(contact_signals)
    experience_details_score, action_verbs = experience_score(raw_resume_text, sections)
    achievement_details_score, metrics = achievement_score(raw_resume_text)
    formatting_score, word_count, avg_sentence_length = resume_length_score(raw_resume_text, resume_text)
    top_terms = extract_top_terms(raw_resume_text)

    score = round(
        alignment["score"]
        + section_score
        + contact_details_score
        + experience_details_score
        + achievement_details_score
        + formatting_score,
        2,
    )

    return {
        "ats_score": min(score, 100),
        "alignment": alignment,
        "sections": sections,
        "contact_signals": contact_signals,
        "action_verbs": action_verbs,
        "metrics": metrics,
        "top_terms": top_terms,
        "word_count": word_count,
        "avg_sentence_length": avg_sentence_length,
        "breakdown": [
            {"label": "Job alignment", "score": round(alignment["score"], 2), "max_score": 35},
            {"label": "Section completeness", "score": round(section_score, 2), "max_score": 20},
            {"label": "Contact completeness", "score": round(contact_details_score, 2), "max_score": 10},
            {"label": "Experience evidence", "score": round(experience_details_score, 2), "max_score": 15},
            {"label": "Achievements and metrics", "score": round(achievement_details_score, 2), "max_score": 10},
            {"label": "Formatting readability", "score": round(formatting_score, 2), "max_score": 10},
        ],
        "estimated_experience_years": estimate_experience_years(raw_resume_text),
    }

def ats_feedback(score, targeted):
    if score >= 85:
        message = "Strong ATS match"
    elif score >= 70:
        message = "Good ATS match"
    elif score >= 55:
        message = "Moderate ATS match"
    else:
        message = "Needs targeted improvement"

    if not targeted:
        return f"{message} based on general resume quality"
    return message

def build_resume_analysis(pdf_details, job_desc):
    if not pdf_details["has_text"]:
        if pdf_details.get("ocr_available"):
            no_text_hint = "No readable text was found, even after OCR processing."
            ocr_tip = "The file might be blank or very low quality. Upload a cleaner PDF for better extraction."
        else:
            no_text_hint = "No readable text found, so ATS scoring could not be completed."
            ocr_tip = "OCR support is not enabled on this server. Install OCR dependencies to parse scanned resumes."

        return {
            "has_text": False,
            "raw_text": "",
            "skills": [],
            "recommended_skills": [],
            "ats_score": 0,
            "ats_message": no_text_hint,
            "alignment": {"mode": "unavailable", "target_keywords": [], "matched_keywords": [], "missing_keywords": []},
            "sections": [],
            "contact_signals": {"email": False, "phone": False, "linkedin": False, "github": False, "website": False},
            "metrics": [],
            "top_terms": [],
            "word_count": 0,
            "avg_sentence_length": 0,
            "breakdown": [
                {"label": "Job alignment", "score": 0, "max_score": 35},
                {"label": "Section completeness", "score": 0, "max_score": 20},
                {"label": "Contact completeness", "score": 0, "max_score": 10},
                {"label": "Experience evidence", "score": 0, "max_score": 15},
                {"label": "Achievements and metrics", "score": 0, "max_score": 10},
                {"label": "Formatting readability", "score": 0, "max_score": 10},
            ],
            "estimated_experience_years": 0,
            "improvement_tips": [
                "Upload a PDF that contains selectable text.",
                "If the file is scanned as an image, enable OCR support or export it as a text-based PDF before uploading.",
                "A blank PDF cannot be scored because no resume content is available.",
                ocr_tip,
            ],
        }

    raw_resume_text = pdf_details["raw_text"]
    normalized_resume_text = pdf_details["text"]
    skills = extract_skills_nlp(normalized_resume_text)
    ats_details = calculate_ats_score(normalized_resume_text, raw_resume_text, skills, job_desc)
    role_matches = match_job_roles(skills)

    if job_desc:
        recommended_skills = [
            skill for skill in extract_skills_nlp(job_desc)
            if skill not in skills
        ]
    elif role_matches:
        recommended_skills = role_matches[0][1]["missing_skills"]
    else:
        recommended_skills = [skill for skill in skill_keywords if skill not in skills][:6]

    improvement_tips = []
    if ats_details["alignment"]["missing_keywords"]:
        missing_keywords = ", ".join(ats_details["alignment"]["missing_keywords"][:5])
        improvement_tips.append(f"Add proof of these target keywords where they are genuinely relevant: {missing_keywords}.")
    missing_sections = [section for section in SECTION_PATTERNS if section not in ats_details["sections"]]
    if missing_sections:
        improvement_tips.append(f"Add or strengthen these resume sections: {', '.join(missing_sections[:4])}.")
    if not ats_details["contact_signals"]["email"] or not ats_details["contact_signals"]["phone"]:
        improvement_tips.append("Include a clear email address and phone number near the top of the resume.")
    if not ats_details["metrics"]:
        improvement_tips.append("Quantify achievements with numbers, percentages, or business impact to improve ATS relevance.")
    if ats_details["word_count"] < 250 or ats_details["word_count"] > 900:
        improvement_tips.append("Keep the resume content concise but complete. A text length around 250 to 900 words performs better.")

    return {
        "has_text": True,
        "raw_text": raw_resume_text,
        "skills": skills,
        "recommended_skills": recommended_skills[:8],
        "ats_score": ats_details["ats_score"],
        "ats_message": ats_feedback(ats_details["ats_score"], bool(job_desc)),
        "alignment": ats_details["alignment"],
        "sections": ats_details["sections"],
        "contact_signals": ats_details["contact_signals"],
        "metrics": ats_details["metrics"],
        "top_terms": ats_details["top_terms"],
        "word_count": ats_details["word_count"],
        "avg_sentence_length": ats_details["avg_sentence_length"],
        "breakdown": ats_details["breakdown"],
        "estimated_experience_years": ats_details["estimated_experience_years"],
        "improvement_tips": improvement_tips[:5],
    }

def cleanup_cv_exports():
    now = time.time()
    expired = [
        key for key, item in CV_EXPORT_STORE.items()
        if now - item["created_at"] > CV_EXPORT_TTL_SECONDS
    ]
    for key in expired:
        CV_EXPORT_STORE.pop(key, None)

def cache_cv_payload(payload):
    cleanup_cv_exports()
    cv_id = uuid.uuid4().hex
    CV_EXPORT_STORE[cv_id] = {
        "payload": payload,
        "created_at": time.time(),
    }
    return cv_id

def get_cached_cv_payload(cv_id):
    cleanup_cv_exports()
    item = CV_EXPORT_STORE.get(cv_id)
    if not item:
        return None
    return item["payload"]

def looks_like_name_line(line):
    candidate = (line or "").strip()
    if not candidate or len(candidate) > 60:
        return False
    if "@" in candidate or any(char.isdigit() for char in candidate):
        return False
    words = [word for word in candidate.split() if word]
    if not (2 <= len(words) <= 4):
        return False
    return all(re.match(r"^[A-Za-z][A-Za-z'.-]*$", word) for word in words)

def extract_resume_name(raw_text):
    lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]
    for line in lines[:8]:
        if looks_like_name_line(line):
            return " ".join(word.capitalize() for word in line.split())
    return "Professional Candidate"

def extract_first_regex_match(pattern, text):
    match = pattern.search(text or "")
    return match.group(0) if match else ""

def detect_section_heading(line):
    normalized_line = normalize_text(line)

    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            normalized_pattern = normalize_text(pattern)
            if normalized_line in {normalized_pattern, f"{normalized_pattern}:"}:
                return section, ""
            if normalized_line.startswith(f"{normalized_pattern}:"):
                _, remainder = line.split(":", 1)
                return section, remainder.strip()

    return None, ""

def split_resume_by_sections(raw_text):
    sections = {section: [] for section in SECTION_PATTERNS}
    sections["general"] = []
    active_section = "general"

    for line in [entry.strip() for entry in (raw_text or "").splitlines() if entry.strip()]:
        heading, remainder = detect_section_heading(line)
        if heading:
            active_section = heading
            if remainder:
                sections[heading].append(remainder)
            continue

        sections[active_section].append(line)

    return sections

def polish_resume_line(text):
    cleaned = (text or "").replace("\n", " ")
    cleaned = re.sub(r"[\u2022\u2023\u25E6\u2043]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    cleaned = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\s*\|\s*", " | ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,.-")
    return cleaned

def unique_trimmed(items, limit=6):
    seen = set()
    output = []

    for item in items:
        clean_item = polish_resume_line(item)
        if not clean_item:
            continue
        marker = clean_item.lower()
        if marker in seen:
            continue
        seen.add(marker)
        output.append(clean_item)
        if len(output) >= limit:
            break

    return output

def extract_lines_by_keywords(lines, keywords, limit=6):
    keyword_hits = []
    for line in lines:
        normalized_line = normalize_text(line)
        if any(contains_normalized_term(normalized_line, keyword) for keyword in keywords):
            keyword_hits.append(line)

    return unique_trimmed(keyword_hits, limit=limit)

def summarize_points_for_email(points, fallback, limit=2):
    snippets = unique_trimmed(points, limit=limit)
    if not snippets:
        return fallback
    return "; ".join(snippets)

def infer_degree_and_university(raw_text, education_points):
    degree_terms = [
        "bachelor", "master", "b.tech", "m.tech", "b.e", "m.e",
        "bsc", "msc", "bca", "mca", "phd", "diploma", "degree"
    ]
    institution_terms = ["university", "college", "institute", "school"]

    extra_lines = extract_lines_by_keywords(
        (raw_text or "").splitlines(),
        ["education", "degree", "university", "college", "institute", "bachelor", "master"],
        limit=6,
    )
    candidates = unique_trimmed(list(education_points or []) + extra_lines, limit=8)

    degree = ""
    university = ""
    for line in candidates:
        normalized_line = normalize_text(line)
        if not degree and any(term in normalized_line for term in degree_terms):
            degree = line
        if not university and any(term in normalized_line for term in institution_terms):
            university = line
        if degree and university:
            break

    return degree or "a relevant degree", university or "a recognized university"

def infer_target_company_name(job_desc):
    text = unescape(job_desc or "").strip()
    if not text:
        return "your company"

    patterns = [
        r"\b(?:at|with|join|joining)\s+([A-Za-z][A-Za-z0-9&.'\- ]{1,50}?)(?=\s+(?:for|as|to|that|which|where)\b|[,.\n]|$)",
        r"\bcompany\s*[:\-]\s*([A-Za-z][A-Za-z0-9&.'\- ]{1,50})(?=[,.\n]|$)",
        r"\borganization\s*[:\-]\s*([A-Za-z][A-Za-z0-9&.'\- ]{1,50})(?=[,.\n]|$)",
    ]
    blocked_values = {
        "company", "organization", "team", "role", "position", "the company", "your company"
    }

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue

        candidate = polish_resume_line(match.group(1))
        candidate = re.sub(r"^(the|a|an)\s+", "", candidate, flags=re.IGNORECASE)
        candidate = candidate.strip(" .,:;-")
        normalized_candidate = normalize_text(candidate)

        if not candidate:
            continue
        if normalized_candidate in blocked_values:
            continue
        if len(candidate.split()) > 6:
            continue
        return candidate

    return "your company"

def infer_first_name(full_name):
    words = [word.strip() for word in (full_name or "").split() if word.strip()]
    if not words:
        return "there"
    return words[0]

def clean_company_name(company_name):
    candidate = polish_resume_line(company_name or "")
    if not candidate:
        return "the company"

    cleaned = re.sub(r"\s+", " ", candidate).strip(" .,-")
    if not cleaned:
        return "the company"
    return cleaned

def title_from_job_entry(job):
    title = polish_resume_line((job or {}).get("title", ""))
    if title:
        return title
    return "the open role"

def build_recruiter_outreach_sequences(analysis, user_email, job_desc, job_matches, job_recommendations, max_companies=6):
    base_payload = build_cv_payload(analysis, user_email, job_desc, job_matches)

    candidate_name = base_payload.get("name", "Professional Candidate")
    candidate_first_name = infer_first_name(candidate_name)
    headline = base_payload.get("headline", "Technology Professional")
    role_title = base_payload.get("role_title", headline)
    skills_phrase = base_payload.get("skills_phrase", "software development fundamentals")
    project_summary = base_payload.get("project_summary", "academic and personal projects")
    experience_summary = base_payload.get("experience_summary", "project implementation and collaboration")
    company_reason_default = base_payload.get("company_reason", "strong engineering culture and meaningful products")
    contact_email = base_payload.get("email") or (user_email or "")
    contact_phone = base_payload.get("phone", "")
    linkedin_url = base_payload.get("linkedin", "")
    github_url = base_payload.get("github", "")
    website_url = base_payload.get("website", "")

    preferred_companies = []
    seen_company_key = set()

    for job in job_recommendations or []:
        company = clean_company_name(job.get("company"))
        company_key = normalize_text(company)
        if not company_key or company_key in seen_company_key:
            continue

        seen_company_key.add(company_key)
        preferred_companies.append(
            {
                "company": company,
                "role": title_from_job_entry(job),
                "source": job.get("source", "Live Jobs"),
                "job_url": clean_url(job.get("url", "")),
                "location": polish_resume_line(job.get("location", "")) or "Not specified",
            }
        )

        if len(preferred_companies) >= max_companies:
            break

    if not preferred_companies:
        fallback_company = clean_company_name(base_payload.get("company_name") or infer_target_company_name(job_desc))
        preferred_companies.append(
            {
                "company": fallback_company,
                "role": role_title,
                "source": "Resume Target",
                "job_url": "",
                "location": "Not specified",
            }
        )

    outreach_sequences = []

    for company_entry in preferred_companies:
        company_name = company_entry["company"]
        target_role = company_entry["role"] or role_title
        company_reason = company_reason_default
        if analysis.get("alignment", {}).get("matched_keywords"):
            company_reason = (
                "focus on "
                + ", ".join(unique_trimmed(analysis["alignment"].get("matched_keywords", []), limit=2))
                + " and high-impact delivery"
            )

        linkedin_message = (
            f"Hi {{recruiter_name}}, I am {candidate_name}, a {headline} interested in {target_role} opportunities at {company_name}. "
            f"I have hands-on work in {skills_phrase} and recently delivered {project_summary}. "
            f"Would you be open to a quick conversation to see if my background fits your hiring needs?"
        )

        email_subject = f"Application Interest: {target_role} at {company_name} | {candidate_name}"

        email_step_1 = (
            f"Hi {{recruiter_name}},\n\n"
            f"I hope you are doing well. I am {candidate_name}, and I am reaching out regarding {target_role} openings at {company_name}. "
            f"My background includes {skills_phrase}, with recent work across {experience_summary} and {project_summary}.\n\n"
            f"I am particularly interested in {company_name} because of its {company_reason}. "
            "If there is a suitable opportunity, I would be glad to share my resume and discuss how I can contribute.\n\n"
            "Thank you for your time.\n"
            f"Best regards,\n{candidate_name}"
        )

        email_step_2 = (
            f"Hi {{recruiter_name}},\n\n"
            f"Following up on my previous note about {target_role} opportunities at {company_name}. "
            f"I wanted to reiterate my interest and availability to interview.\n\n"
            f"I can contribute with practical strengths in {skills_phrase}, and I would value the chance to support your team.\n\n"
            "Happy to share additional project details if helpful.\n\n"
            f"Thanks again,\n{candidate_name}"
        )

        email_step_3 = (
            f"Hi {{recruiter_name}},\n\n"
            f"I wanted to send one final follow-up regarding {target_role} roles at {company_name}. "
            "If the team is currently hiring, I would appreciate being considered or directed to the appropriate contact.\n\n"
            "Thank you for your consideration, and I hope to connect soon.\n\n"
            f"Sincerely,\n{candidate_name}"
        )

        contact_lines = []
        if contact_email:
            contact_lines.append(f"Email: {contact_email}")
        if contact_phone:
            contact_lines.append(f"Phone: {contact_phone}")
        if linkedin_url:
            contact_lines.append(f"LinkedIn: {linkedin_url}")
        if github_url:
            contact_lines.append(f"GitHub: {github_url}")
        if website_url:
            contact_lines.append(f"Portfolio: {website_url}")

        outreach_sequences.append(
            {
                "company": company_name,
                "role": target_role,
                "source": company_entry.get("source", "Live Jobs"),
                "location": company_entry.get("location", "Not specified"),
                "job_url": company_entry.get("job_url", ""),
                "candidate_first_name": candidate_first_name,
                "linkedin_message": linkedin_message,
                "email_subject": email_subject,
                "email_step_1": email_step_1,
                "email_step_2": email_step_2,
                "email_step_3": email_step_3,
                "contact_block": "\n".join(contact_lines),
            }
        )

    return outreach_sequences

def build_cv_payload(analysis, user_email, job_desc, job_matches):
    raw_text = analysis.get("raw_text", "")
    sections = split_resume_by_sections(raw_text)
    all_lines = [line for lines in sections.values() for line in lines]

    name = extract_resume_name(raw_text)
    email = extract_first_regex_match(EMAIL_REGEX, raw_text) or (user_email or "")
    phone = extract_first_regex_match(PHONE_REGEX, raw_text)

    urls = [clean_url(url) for url in URL_REGEX.findall(raw_text)]
    linkedin = next((url for url in urls if "linkedin.com" in url.lower()), "")
    github = next((url for url in urls if "github.com" in url.lower()), "")
    website = next(
        (
            url for url in urls
            if "linkedin.com" not in url.lower() and "github.com" not in url.lower()
        ),
        "",
    )

    if analysis.get("alignment", {}).get("mode") == "general" and analysis.get("alignment", {}).get("target_role"):
        headline = analysis["alignment"]["target_role"]
    elif job_matches:
        headline = job_matches[0][0]
    elif analysis.get("skills"):
        headline = f"{analysis['skills'][0].title()} Specialist"
    else:
        headline = "Technology Professional"

    summary_lines = unique_trimmed(sections.get("summary", []), limit=3)
    if summary_lines:
        summary = " ".join(summary_lines)
    else:
        skills_preview = ", ".join(analysis.get("skills", [])[:6]) or "modern software tools"
        years = analysis.get("estimated_experience_years", 0)
        if years:
            summary = (
                f"Results-driven professional with {years}+ years of practical experience delivering business outcomes "
                f"using {skills_preview}. Focused on quality execution, measurable impact, and collaborative delivery."
            )
        else:
            summary = (
                f"Results-driven professional with practical strengths in {skills_preview}. "
                "Focused on clear communication, reliable execution, and measurable improvements."
            )

    experience_points = unique_trimmed(sections.get("experience", []), limit=6)
    if not experience_points:
        # Search for experience-specific content, excluding project keywords to avoid duplication
        experience_points = extract_lines_by_keywords(
            all_lines,
            list(ACTION_VERBS) + ["client", "team", "delivery", "api", "led", "managed"],
            limit=6,
        )

    project_points = unique_trimmed(sections.get("projects", []), limit=5)
    if not project_points:
        # Search specifically for project content
        project_points = extract_lines_by_keywords(
            all_lines,
            ["project", "built", "developed", "implemented", "deployed"],
            limit=5,
        )
    experience_markers = {normalize_text(item) for item in experience_points}
    project_points = [point for point in project_points if normalize_text(point) not in experience_markers]

    education_points = unique_trimmed(sections.get("education", []), limit=4)
    if not education_points:
        education_points = extract_lines_by_keywords(
            all_lines,
            ["university", "college", "bachelor", "master", "education", "degree"],
            limit=4,
        )

    certification_points = unique_trimmed(sections.get("certifications", []), limit=4)
    if not certification_points:
        certification_points = extract_lines_by_keywords(
            all_lines,
            ["certification", "certificate", "certified", "aws", "azure", "google"],
            limit=4,
        )

    skill_list = analysis.get("skills", []) or analysis.get("top_terms", [])
    matched_keywords = analysis.get("alignment", {}).get("matched_keywords", [])
    degree, university = infer_degree_and_university(raw_text, education_points)
    company_name = infer_target_company_name(job_desc)
    role_title = analysis.get("alignment", {}).get("target_role") or headline or "Entry-Level Technology Position"

    skills_for_email = unique_trimmed(skill_list, limit=8)
    skills_phrase = ", ".join(skills_for_email[:5]) or "software development fundamentals"
    foundation_phrase = ", ".join(skills_for_email[:3]) or "software development, communication, and collaboration"
    project_summary = summarize_points_for_email(
        project_points,
        "academic and personal projects relevant to the role",
        limit=2,
    )
    experience_summary = summarize_points_for_email(
        experience_points,
        "collaborative implementation and problem-solving assignments",
        limit=2,
    )
    company_reason = (
        f"focus on {', '.join(unique_trimmed(matched_keywords, limit=2))} and practical innovation"
        if matched_keywords
        else "learning culture, meaningful projects, and growth opportunities"
    )
    passion_field = normalize_text(role_title) or "technology"

    return {
        "name": name,
        "headline": headline,
        "role_title": role_title,
        "recipient_name": "Hiring Manager",
        "company_name": company_name,
        "degree": degree,
        "university": university,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
        "website": website,
        "summary": summary,
        "skills": unique_trimmed(skill_list, limit=16),
        "experience_points": experience_points or ["Experience details were extracted from the uploaded resume."],
        "project_points": project_points,
        "education_points": education_points,
        "certification_points": certification_points,
        "skills_phrase": skills_phrase,
        "foundation_phrase": foundation_phrase,
        "project_summary": project_summary,
        "experience_summary": experience_summary,
        "company_reason": company_reason,
        "passion_field": passion_field,
        "ats_score": analysis.get("ats_score", 0),
        "matched_keywords": unique_trimmed(matched_keywords, limit=8),
        "targeted_role": analysis.get("alignment", {}).get("target_role", ""),
        "job_desc_used": bool(job_desc),
    }

def generate_cv_pdf_buffer(cv_payload):
    ensure_cv_export_backends()

    if not CV_EXPORT_AVAILABLE:
        details = []
        if CV_EXPORT_IMPORT_ERROR:
            details.append(f"reportlab: {CV_EXPORT_IMPORT_ERROR}")
        if CV_EXPORT_FALLBACK_IMPORT_ERROR:
            details.append(f"fpdf2: {CV_EXPORT_FALLBACK_IMPORT_ERROR}")
        if CV_EXPORT_BOOTSTRAP_ERROR:
            details.append(f"bootstrap: {CV_EXPORT_BOOTSTRAP_ERROR}")
        detail = f" Details: {'; '.join(details)}" if details else ""
        raise ValueError(f"Cold email PDF generation dependencies are unavailable. Install reportlab or fpdf2.{detail}")

    if not SimpleDocTemplate and FPDF:
        return generate_cv_pdf_buffer_with_fpdf(cv_payload)

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50,
        title=f"Cold Email - {cv_payload['name']}",
    )

    styles = getSampleStyleSheet()

    subject_style = ParagraphStyle(
        "EmailSubject",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=colors.black,
        leading=14,
        spaceAfter=10,
    )

    body_style = ParagraphStyle(
        "EmailBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.black,
        leading=14,
        spaceAfter=8,
        alignment=4,
    )

    closing_style = ParagraphStyle(
        "EmailClosing",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.black,
        leading=14,
        spaceAfter=4,
    )

    signature_style = ParagraphStyle(
        "Signature",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.black,
        leading=12,
        spaceAfter=0,
    )

    story = []

    name = cv_payload.get("name", "Professional Candidate")
    role_title = cv_payload.get("role_title", cv_payload.get("headline", "Entry-Level Position"))
    recipient_name = cv_payload.get("recipient_name", "Hiring Manager")
    company_name = cv_payload.get("company_name", "your company")
    degree = cv_payload.get("degree", "a relevant degree")
    university = cv_payload.get("university", "a recognized university")
    skills_phrase = cv_payload.get("skills_phrase", "software development fundamentals")
    project_summary = cv_payload.get("project_summary", "academic and personal projects")
    foundation_phrase = cv_payload.get("foundation_phrase", "software development, communication, and collaboration")
    company_reason = cv_payload.get("company_reason", "learning culture and growth opportunities")
    experience_summary = cv_payload.get("experience_summary", "project implementation and team collaboration")
    passion_field = cv_payload.get("passion_field", "technology")

    subject_line = f"Subject: Application for {role_title} - {name}"
    story.append(Paragraph(escape(subject_line), subject_style))

    story.append(Paragraph(escape(f"Dear {recipient_name},"), body_style))

    paragraph_1 = (
        f"I hope this email finds you well. My name is {name}, and I am a recent graduate in {degree} "
        f"from {university}. I am reaching out to express my interest in any entry-level opportunities at "
        f"{company_name} that align with my skills and qualifications."
    )
    story.append(Paragraph(escape(paragraph_1), body_style))

    paragraph_2 = (
        f"During my academic journey, I have gained knowledge in {skills_phrase} and have completed projects in "
        f"{project_summary}. These experiences have equipped me with a solid foundation in {foundation_phrase} "
        f"and a strong passion for {passion_field}."
    )
    story.append(Paragraph(escape(paragraph_2), body_style))

    paragraph_3 = (
        f"I am particularly drawn to {company_name} because of its {company_reason}. I believe my background in "
        f"{experience_summary} and my eagerness to learn and contribute can make me a valuable addition to your team."
    )
    story.append(Paragraph(escape(paragraph_3), body_style))

    paragraph_4 = (
        "I would love to explore how my skills and enthusiasm can contribute to the success of your organization. "
        "Please find my resume attached for your reference. "
        f"I would greatly appreciate the opportunity to discuss how I can contribute to {company_name} in a {role_title} role."
    )
    story.append(Paragraph(escape(paragraph_4), body_style))

    story.append(Paragraph(
        "Thank you for your time and consideration. I look forward to hearing from you.",
        body_style,
    ))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Warm regards,", closing_style))
    story.append(Paragraph(escape(name), closing_style))

    contact_parts = []
    if cv_payload.get("email"):
        contact_parts.append(cv_payload["email"])
    if cv_payload.get("phone"):
        contact_parts.append(cv_payload["phone"])
    if contact_parts:
        story.append(Paragraph(escape(" | ".join(contact_parts)), signature_style))

    if cv_payload.get("linkedin"):
        story.append(Paragraph(escape(cv_payload["linkedin"]), signature_style))
    if cv_payload.get("github"):
        story.append(Paragraph(escape(cv_payload["github"]), signature_style))
    if cv_payload.get("website"):
        story.append(Paragraph(escape(cv_payload["website"]), signature_style))

    document.build(story)
    buffer.seek(0)
    return buffer

def generate_cv_pdf_buffer_with_fpdf(cv_payload):
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    name = cv_payload.get("name", "Professional Candidate")
    role_title = cv_payload.get("role_title", cv_payload.get("headline", "Entry-Level Position"))
    recipient_name = cv_payload.get("recipient_name", "Hiring Manager")
    company_name = cv_payload.get("company_name", "your company")
    degree = cv_payload.get("degree", "a relevant degree")
    university = cv_payload.get("university", "a recognized university")
    skills_phrase = cv_payload.get("skills_phrase", "software development fundamentals")
    project_summary = cv_payload.get("project_summary", "academic and personal projects")
    foundation_phrase = cv_payload.get("foundation_phrase", "software development, communication, and collaboration")
    company_reason = cv_payload.get("company_reason", "learning culture and growth opportunities")
    experience_summary = cv_payload.get("experience_summary", "project implementation and team collaboration")
    passion_field = cv_payload.get("passion_field", "technology")

    subject_line = f"Subject: Application for {role_title} - {name}"

    paragraph_1 = (
        f"I hope this email finds you well. My name is {name}, and I am a recent graduate in {degree} "
        f"from {university}. I am reaching out to express my interest in any entry-level opportunities at "
        f"{company_name} that align with my skills and qualifications."
    )
    paragraph_2 = (
        f"During my academic journey, I have gained knowledge in {skills_phrase} and have completed projects in "
        f"{project_summary}. These experiences have equipped me with a solid foundation in {foundation_phrase} "
        f"and a strong passion for {passion_field}."
    )
    paragraph_3 = (
        f"I am particularly drawn to {company_name} because of its {company_reason}. I believe my background in "
        f"{experience_summary} and my eagerness to learn and contribute can make me a valuable addition to your team."
    )
    paragraph_4 = (
        "I would love to explore how my skills and enthusiasm can contribute to the success of your organization. "
        "Please find my resume attached for your reference. "
        f"I would greatly appreciate the opportunity to discuss how I can contribute to {company_name} in a {role_title} role."
    )

    def write_line(text, bold=False, spacing=2):
        pdf.set_font("Helvetica", "B" if bold else "", 11 if bold else 10)
        pdf.multi_cell(0, 6, unescape(text))
        if spacing:
            pdf.ln(spacing)

    write_line(subject_line, bold=True, spacing=3)
    write_line(f"Dear {recipient_name},")
    write_line(paragraph_1)
    write_line(paragraph_2)
    write_line(paragraph_3)
    write_line(paragraph_4)
    write_line("Thank you for your time and consideration. I look forward to hearing from you.")
    write_line("Warm regards,")
    write_line(name)

    contact_parts = []
    if cv_payload.get("email"):
        contact_parts.append(cv_payload["email"])
    if cv_payload.get("phone"):
        contact_parts.append(cv_payload["phone"])
    if contact_parts:
        write_line(" | ".join(contact_parts), spacing=1)

    if cv_payload.get("linkedin"):
        write_line(cv_payload["linkedin"], spacing=1)
    if cv_payload.get("github"):
        write_line(cv_payload["github"], spacing=1)
    if cv_payload.get("website"):
        write_line(cv_payload["website"], spacing=1)

    pdf_output = pdf.output(dest="S")
    if isinstance(pdf_output, str):
        pdf_bytes = pdf_output.encode("latin-1")
    elif isinstance(pdf_output, bytearray):
        pdf_bytes = bytes(pdf_output)
    else:
        pdf_bytes = pdf_output

    buffer = BytesIO(pdf_bytes)
    buffer.seek(0)
    return buffer

def recommend_courses(missing_skills):
    return {skill: f"Take a professional certification course in {skill.title()}" for skill in missing_skills}

def match_job_roles(extracted_skills):
    matches = {}
    for role, required_skills in job_roles.items():
        matched_skills = set(extracted_skills) & set(required_skills)
        match_score = (len(matched_skills) / len(required_skills)) * 100 if required_skills else 0
        matches[role] = {
            "score": round(match_score, 2),
            "matched_skills": list(matched_skills),
            "missing_skills": list(set(required_skills) - set(extracted_skills))
        }
    return sorted(matches.items(), key=lambda x: x[1]["score"], reverse=True)

# -------------------------------------------------
# Job Recommendation Functions
# -------------------------------------------------

LIVE_JOB_DOMAINS = {
    "LinkedIn": "linkedin.com",
    "Indeed": "indeed.com",
    "Naukri": "naukri.com",
    "Glassdoor": "glassdoor.com",
}

SOURCE_DOMAIN_HINTS = {
    "LinkedIn": ["linkedin.com", "in.linkedin.com"],
    "Indeed": ["indeed.com"],
    "Naukri": ["naukri.com"],
    "Glassdoor": ["glassdoor.com", "glassdoor.co.in"],
}

SOURCE_QUERY_HINTS = {
    "LinkedIn": ["site:linkedin.com/jobs", "site:in.linkedin.com/jobs", "site:linkedin.com/jobs/view"],
    "Indeed": ["site:indeed.com/jobs", "site:indeed.com job"],
    "Naukri": ["site:naukri.com", "site:www.naukri.com"],
    "Glassdoor": ["site:glassdoor.co.in/Job", "site:glassdoor.com/Job", "site:glassdoor.com/job"],
}

JOB_SOURCE_PRIORITY = {
    "LinkedIn": 0,
    "Indeed": 1,
    "Naukri": 2,
    "Glassdoor": 3,
}

RECENT_POSTING_REGEX = re.compile(
    r"(?:(\d+)\+?\s*(hour|day|week|month)s?\s*ago|today|yesterday|just\s*posted|recent|new)",
    re.IGNORECASE,
)

JOB_SIGNAL_REGEX = re.compile(
    r"\b(job|jobs|hiring|vacancy|vacancies|opening|openings|position|positions|intern|developer|engineer|analyst|scientist)\b",
    re.IGNORECASE,
)

NON_JOB_TITLE_HINTS = {
    "sign in",
    "login",
    "create an account",
    "career community",
    "best places to work",
    "jobseeker's login",
    "your job search starts here",
}

def search_recency_timelimit():
    if JOB_RECENCY_DAYS <= 1:
        return "d"
    if JOB_RECENCY_DAYS <= 7:
        return "w"
    if JOB_RECENCY_DAYS <= 31:
        return "m"
    return "y"

def recency_label_and_rank(text, published_at=""):
    if published_at:
        try:
            published_dt = parsedate_to_datetime(published_at)
            if published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            age_seconds = (now - published_dt.astimezone(timezone.utc)).total_seconds()
            age_days = max(0, int(age_seconds // 86400))

            if age_days == 0:
                return "Today", 0

            suffix = "" if age_days == 1 else "s"
            return f"{age_days} day{suffix} ago", age_days
        except Exception:
            pass

    lower_text = normalize_text(text)
    if "just posted" in lower_text or "today" in lower_text:
        return "Today", 0
    if "yesterday" in lower_text:
        return "1 day ago", 1

    match = re.search(r"(\d+)\+?\s*(hour|day|week|month)s?\s*ago", lower_text)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        if unit == "hour":
            rank = 0
        elif unit == "day":
            rank = value
        elif unit == "week":
            rank = value * 7
        else:
            rank = value * 30

        suffix = "" if value == 1 else "s"
        return f"{value} {unit}{suffix} ago", rank

    if "new" in lower_text or "recent" in lower_text:
        return "Recent", 2

    return "Recent", JOB_RECENCY_DAYS + 1

def is_recent_rank(rank):
    return rank <= max(JOB_RECENCY_DAYS * 2, 14)

def normalize_job_url(url):
    clean = (url or "").strip()
    if not clean:
        return "#"
    if clean.startswith("//"):
        clean = f"https:{clean}"
    return clean

def infer_company_name(title, snippet):
    title_text = (title or "").strip()
    snippet_text = (snippet or "").strip()

    if " at " in title_text.lower():
        parts = re.split(r"\bat\b", title_text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            company = parts[-1].strip(" -|:")
            if company:
                return company

    split_tokens = re.split(r"\s+[|\-]\s+", title_text)
    if len(split_tokens) >= 2:
        likely_company = split_tokens[-1].strip()
        if likely_company and len(likely_company) <= 40:
            return likely_company

    snippet_company = re.search(r"\bat\s+([A-Za-z0-9&.,\-\s]{2,40})", snippet_text, re.IGNORECASE)
    if snippet_company:
        return snippet_company.group(1).strip(" -|:")

    return "Not specified"

def source_query_bases(source_name, domain):
    bases = SOURCE_QUERY_HINTS.get(source_name, [])
    if bases:
        return bases[:JOB_QUERY_VARIANTS_PER_SOURCE]
    return [f"site:{domain}"]

def source_domain_tokens(source_name, domain):
    tokens = set()
    for value in SOURCE_DOMAIN_HINTS.get(source_name, []):
        token = normalize_text(value).replace("www.", "")
        if token:
            tokens.add(token)

    fallback_token = normalize_text(domain.split("/")[0]).replace("www.", "")
    if fallback_token:
        tokens.add(fallback_token)

    return tokens

def job_belongs_to_source(source_name, domain, *text_parts):
    haystack = normalize_text(" ".join(part for part in text_parts if part))
    if not haystack:
        return False

    for token in source_domain_tokens(source_name, domain):
        if token in haystack:
            return True

    source_name_token = normalize_text(source_name)
    if source_name_token and source_name_token in haystack:
        return True

    return False

def looks_like_job_posting(title, description):
    text = normalize_text(f"{title} {description}")
    if not text:
        return False

    for blocked in NON_JOB_TITLE_HINTS:
        if blocked in text:
            return False

    return bool(JOB_SIGNAL_REGEX.search(text))

def fetch_recent_jobs_from_google_news(source_name, domain, keywords, location, max_results, deadline=None):
    def pull_from_query(query_terms):
        query = quote(" ".join(term for term in query_terms if term))
        feed_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        pulled_jobs = []

        request_timeout = bounded_request_timeout(deadline)
        if request_timeout <= 0:
            return pulled_jobs

        response = requests.get(
            feed_url,
            timeout=request_timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)

        for item in root.findall("./channel/item"):
            title = unescape((item.findtext("title") or "N/A").strip())
            url = normalize_job_url((item.findtext("link") or "").strip())
            description_html = unescape((item.findtext("description") or "").strip())
            description = re.sub(r"<[^>]+>", " ", description_html)
            description = re.sub(r"\s+", " ", description).strip()
            posted_raw = (item.findtext("pubDate") or "").strip()

            source_node = item.find("source")
            source_text = unescape((source_node.text or "").strip()) if source_node is not None else ""
            source_url = normalize_job_url((source_node.get("url") or "").strip()) if source_node is not None else ""

            if not job_belongs_to_source(source_name, domain, title, url, description, source_text, source_url):
                continue

            if not looks_like_job_posting(title, description):
                continue

            posted, rank = recency_label_and_rank(
                f"{posted_raw} {title} {description}",
                published_at=posted_raw,
            )
            if not is_recent_rank(rank):
                continue

            pulled_jobs.append(
                {
                    "title": title,
                    "company": infer_company_name(title, description),
                    "location": location,
                    "url": url,
                    "description": (description[:220] + "...") if len(description) > 220 else description,
                    "salary": "Not listed",
                    "source": source_name,
                    "posted": posted,
                    "_recency_rank": rank,
                }
            )

            if len(pulled_jobs) >= max_results:
                break

        return pulled_jobs

    jobs = []
    try:
        query_batches = []
        for query_base in source_query_bases(source_name, domain):
            query_batches.append([query_base] + keywords[:4] + [location, "jobs", f"when:{JOB_RECENCY_DAYS}d"])
            query_batches.append([query_base, location, "jobs", f"when:{JOB_RECENCY_DAYS}d"])

        seen_queries = set()
        for query_terms in query_batches:
            if deadline_exceeded(deadline):
                break

            query_key = tuple(query_terms)
            if query_key in seen_queries:
                continue
            seen_queries.add(query_key)

            pulled = pull_from_query(query_terms)
            jobs.extend(pulled)
            if len(jobs) >= max_results:
                break
    except Exception as error:
        print(f"{source_name} RSS fallback error: {error}")

    deduped = []
    seen = set()
    for job in jobs:
        key = (
            normalize_job_url(job.get("url", "")).lower(),
            job.get("title", "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(job)
        if len(deduped) >= max_results:
            break

    return deduped

def fetch_recent_jobs_from_bing_rss(source_name, domain, keywords, location, max_results, deadline=None):
    jobs = []

    for query_base in source_query_bases(source_name, domain):
        if deadline_exceeded(deadline):
            break

        query_terms = [query_base] + keywords[:4] + [location, "jobs"]
        query = quote(" ".join(term for term in query_terms if term))
        feed_url = f"https://www.bing.com/search?q={query}&format=rss"

        try:
            request_timeout = bounded_request_timeout(deadline)
            if request_timeout <= 0:
                break

            response = requests.get(
                feed_url,
                timeout=request_timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

            root = ET.fromstring(response.content)

            for item in root.findall("./channel/item"):
                title = unescape((item.findtext("title") or "N/A").strip())
                url = normalize_job_url((item.findtext("link") or "").strip())
                description = unescape((item.findtext("description") or "").strip())
                posted_raw = (item.findtext("pubDate") or "").strip()

                if not job_belongs_to_source(source_name, domain, title, url, description):
                    continue

                if not looks_like_job_posting(title, description):
                    continue

                posted, rank = recency_label_and_rank(
                    f"{posted_raw} {title} {description}",
                    published_at=posted_raw,
                )
                if not is_recent_rank(rank):
                    continue

                jobs.append(
                    {
                        "title": title,
                        "company": infer_company_name(title, description),
                        "location": location,
                        "url": url,
                        "description": (description[:220] + "...") if len(description) > 220 else description,
                        "salary": "Not listed",
                        "source": source_name,
                        "posted": posted,
                        "_recency_rank": rank,
                    }
                )

                if len(jobs) >= max_results:
                    break
        except Exception as error:
            print(f"{source_name} Bing RSS fallback error: {error}")

        if len(jobs) >= max_results:
            break

    return jobs

def fetch_recent_jobs_from_domain(source_name, domain, keywords, location, max_results, deadline=None):
    if deadline_exceeded(deadline):
        return []

    search_terms = " ".join(keywords[:5]) if keywords else "software engineer"
    query = f"{source_query_bases(source_name, domain)[0]} {search_terms} {location} jobs"
    timelimit = search_recency_timelimit()
    jobs = []

    if not LIVE_JOB_SEARCH_AVAILABLE:
        combined = fetch_recent_jobs_from_google_news(source_name, domain, keywords, location, max_results, deadline=deadline)
        if len(combined) < max_results:
            combined.extend(fetch_recent_jobs_from_bing_rss(source_name, domain, keywords, location, max_results, deadline=deadline))
        jobs = combined
    else:
        try:
            with DDGS() as ddgs:
                results = ddgs.text(
                    query,
                    region=JOB_SEARCH_REGION,
                    safesearch="off",
                    timelimit=timelimit,
                    max_results=max(max_results * 4, 16),
                )

                for item in results or []:
                    url = normalize_job_url(item.get("href", ""))
                    title = (item.get("title") or "N/A").strip()
                    snippet = (item.get("body") or "").strip()
                    if not job_belongs_to_source(source_name, domain, title, url, snippet):
                        continue

                    if not looks_like_job_posting(title, snippet):
                        continue

                    posted, rank = recency_label_and_rank(f"{title} {snippet}")

                    if not is_recent_rank(rank):
                        continue

                    jobs.append(
                        {
                            "title": title,
                            "company": infer_company_name(title, snippet),
                            "location": location,
                            "url": url,
                            "description": (snippet[:220] + "...") if len(snippet) > 220 else snippet,
                            "salary": "Not listed",
                            "source": source_name,
                            "posted": posted,
                            "_recency_rank": rank,
                        }
                    )

                    if len(jobs) >= max_results:
                        break
        except Exception as error:
            print(f"{source_name} live search error: {error}")

    if len(jobs) < max(1, max_results // 2) and not deadline_exceeded(deadline):
        fallback_jobs = fetch_recent_jobs_from_google_news(source_name, domain, keywords, location, max_results, deadline=deadline)
        jobs.extend(fallback_jobs)

    if len(jobs) < max_results and not deadline_exceeded(deadline):
        jobs.extend(fetch_recent_jobs_from_bing_rss(source_name, domain, keywords, location, max_results, deadline=deadline))

    deduped = []
    seen = set()
    for job in jobs:
        key = (
            normalize_job_url(job.get("url", "")).lower(),
            job.get("title", "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(job)
        if len(deduped) >= max_results:
            break

    return deduped

def fetch_linkedin_jobs(keywords, location=JOB_SEARCH_LOCATION, max_results=8, deadline=None):
    return fetch_recent_jobs_from_domain("LinkedIn", LIVE_JOB_DOMAINS["LinkedIn"], keywords, location, max_results, deadline=deadline)

def fetch_indeed_jobs(keywords, location=JOB_SEARCH_LOCATION, max_results=8, deadline=None):
    return fetch_recent_jobs_from_domain("Indeed", LIVE_JOB_DOMAINS["Indeed"], keywords, location, max_results, deadline=deadline)

def fetch_naukri_jobs(keywords, location=JOB_SEARCH_LOCATION, max_results=8, deadline=None):
    return fetch_recent_jobs_from_domain("Naukri", LIVE_JOB_DOMAINS["Naukri"], keywords, location, max_results, deadline=deadline)

def fetch_glassdoor_jobs(keywords, location=JOB_SEARCH_LOCATION, max_results=8, deadline=None):
    return fetch_recent_jobs_from_domain("Glassdoor", LIVE_JOB_DOMAINS["Glassdoor"], keywords, location, max_results, deadline=deadline)

def aggregate_job_recommendations(skills, job_roles_matched, max_per_source=12):
    """Fetch ALL relevant live jobs from LinkedIn, Indeed, Naukri, and Glassdoor based on resume skills and job roles."""
    all_jobs = []
    
    # Build comprehensive keyword list that includes ALL skills + ALL matched job roles
    keywords = []
    
    # Add all extracted skills (not just first 5)
    keywords.extend([skill for skill in skills if skill])
    
    # Add ALL matched job roles (not just the top one) - top 5 with highest match scores
    if job_roles_matched:
        top_roles = job_roles_matched[:5]  # Use top 5 matched roles for comprehensive search
        keywords.extend([role[0] for role in top_roles])
    
    # Fallback if no skills/roles found
    if not keywords:
        keywords = ["software engineer", "developer", "analyst"]
    
    # Remove duplicates while preserving order
    keywords = list(dict.fromkeys(keywords))

    if not keywords:
        keywords = ["software", "engineer"]

    deadline = time.monotonic() + JOB_FETCH_TOTAL_TIMEOUT_SECONDS

    sources = [
        ("LinkedIn", lambda: fetch_linkedin_jobs(keywords, max_results=max_per_source, deadline=deadline)),
        ("Indeed", lambda: fetch_indeed_jobs(keywords, max_results=max_per_source, deadline=deadline)),
        ("Naukri", lambda: fetch_naukri_jobs(keywords, max_results=max_per_source, deadline=deadline)),
        ("Glassdoor", lambda: fetch_glassdoor_jobs(keywords, max_results=max_per_source, deadline=deadline)),
    ]

    for source_name, fetcher in sources:
        if deadline_exceeded(deadline):
            break

        try:
            source_jobs = fetcher()
            all_jobs.extend(source_jobs)
        except Exception as error:
            print(f"{source_name} fetch skipped due to error: {error}")

    seen = set()
    unique_jobs = []
    for job in all_jobs:
        identifier = (
            normalize_job_url(job.get("url", "")).lower(),
            job.get("title", "").strip().lower(),
            job.get("source", "").strip().lower(),
        )
        if identifier in seen:
            continue
        seen.add(identifier)
        unique_jobs.append(job)

    unique_jobs.sort(
        key=lambda job: (
            job.get("_recency_rank", JOB_RECENCY_DAYS + 2),
            JOB_SOURCE_PRIORITY.get(job.get("source", ""), 99),
        )
    )

    for job in unique_jobs:
        job.pop("_recency_rank", None)

    return unique_jobs[:50]  # Return up to 50 jobs (more comprehensive results)

# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        user_email = request.form.get("email")
        file = request.files.get("resume")
        job_desc = request.form.get("job_desc", "").strip()
        email_status = None
        alert_status = None
        cv_download_id = None
        cv_generation_status = None

        if not file or file.filename == "":
            return render_template("result.html", error="Please upload a resume PDF.")

        if not file.filename.lower().endswith(".pdf"):
            return render_template("result.html", error="Only PDF resumes are supported in the current analyzer.")

        pdf_details = extract_pdf_text_details(file)
        analysis = build_resume_analysis(pdf_details, job_desc)
        skills = analysis["skills"]
        ats_score = analysis["ats_score"]
        ats_message = analysis["ats_message"]

        if user_email and analysis["has_text"]:
            if email_feedback_enabled():
                email_sent = send_feedback_email(user_email, ats_score, ats_message, skills)
                email_status = "Analysis email sent successfully." if email_sent else "Analysis completed, but the email could not be delivered. Check SMTP settings and try again."
            else:
                email_status = "Analysis completed, but email delivery is disabled because SMTP credentials are not configured in environment variables."
        elif user_email:
            email_status = "Email was not sent because the uploaded PDF did not contain readable resume text."

        job_matches = match_job_roles(skills) if analysis["has_text"] else []

        if analysis["has_text"]:
            ensure_cv_export_backends()
            estimated_experience = min(analysis["estimated_experience_years"], 4)
            selection_probability = predict_selection_probability(min(len(skills), 10), estimated_experience)
            match_percent = round((len(analysis["alignment"]["matched_keywords"]) / len(analysis["alignment"]["target_keywords"])) * 100, 2) if analysis["alignment"]["target_keywords"] else round((len(skills) / max(len(skill_keywords), 1)) * 100, 2)
            readiness_score = round((ats_score * 0.7) + (selection_probability * 0.3), 2)
            job_recommendations = aggregate_job_recommendations(skills, job_matches, max_per_source=12)
            outreach_sequences = build_recruiter_outreach_sequences(
                analysis,
                user_email,
                job_desc,
                job_matches,
                job_recommendations,
                max_companies=6,
            )
            interview_prep_sets = build_interview_prep_assistant(
                analysis,
                job_matches,
                job_recommendations,
                max_companies=3,
            )

            enable_job_alerts = request.form.get("enable_job_alerts", "").strip().lower() in {"1", "true", "on", "yes"}
            alert_frequency = request.form.get("alert_frequency", "weekly").strip().lower()
            alert_location = request.form.get("alert_location", "").strip() or JOB_SEARCH_LOCATION
            alert_min_salary = parse_int_or_none(request.form.get("alert_min_salary", ""))
            alert_max_salary = parse_int_or_none(request.form.get("alert_max_salary", ""))

            if enable_job_alerts:
                if user_email:
                    saved, message = save_or_update_job_alert_subscription(
                        user_email,
                        skills,
                        alert_location,
                        alert_min_salary,
                        alert_max_salary,
                        alert_frequency,
                    )
                    alert_status = message if saved else f"Job alert setup failed: {message}"
                else:
                    alert_status = "Job alerts were requested, but email is required to subscribe."

            if CV_EXPORT_AVAILABLE:
                cv_payload = build_cv_payload(analysis, user_email, job_desc, job_matches)
                cv_download_id = cache_cv_payload(cv_payload)
            else:
                dependency_details = []
                if CV_EXPORT_IMPORT_ERROR:
                    dependency_details.append(f"reportlab: {CV_EXPORT_IMPORT_ERROR}")
                if CV_EXPORT_FALLBACK_IMPORT_ERROR:
                    dependency_details.append(f"fpdf2: {CV_EXPORT_FALLBACK_IMPORT_ERROR}")
                if CV_EXPORT_BOOTSTRAP_ERROR:
                    dependency_details.append(f"bootstrap: {CV_EXPORT_BOOTSTRAP_ERROR}")
                detail_suffix = f" (import errors: {'; '.join(dependency_details)})" if dependency_details else ""
                cv_generation_status = (
                    "Cold email PDF export is unavailable because PDF generation dependencies are not installed."
                    " Ensure reportlab or fpdf2 is installed in the runtime environment"
                    f"{detail_suffix}."
                )
        else:
            selection_probability = 0
            match_percent = 0
            readiness_score = 0
            job_recommendations = []
            outreach_sequences = []
            interview_prep_sets = []
            cv_generation_status = "Cold email PDF export is unavailable for resumes with no readable text."

        return render_template(
            "result.html",
            analysis=analysis,
            skills=skills,
            ats_score=ats_score,
            ats_message=ats_message,
            match=round(match_percent, 2),
            prediction=round(selection_probability, 2),
            score=round(readiness_score, 2),
            recommendations=recommend_courses(analysis["recommended_skills"]),
            job_matches=job_matches,
            job_recommendations=job_recommendations,
            page_count=pdf_details["page_count"],
            extractable_pages=pdf_details["extractable_pages"],
            job_desc_provided=bool(job_desc),
            email_status=email_status,
            text_source=pdf_details.get("text_source", "none"),
            ocr_available=pdf_details.get("ocr_available", False),
            ocr_used=pdf_details.get("ocr_used", False),
            cv_download_id=cv_download_id,
            cv_generation_status=cv_generation_status,
            outreach_sequences=outreach_sequences,
            interview_prep_sets=interview_prep_sets,
            alert_status=alert_status,
        )

    except Exception as e:
        return render_template("result.html", error=str(e))

@app.route("/download-cv/<cv_id>", methods=["GET"])
def download_cv(cv_id):
    try:
        payload = get_cached_cv_payload(cv_id)
        if not payload:
            return render_template("result.html", error="Cold email download link expired. Please analyze your resume again.")

        cv_pdf = generate_cv_pdf_buffer(payload)
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", payload.get("name", "candidate")).strip("_") or "candidate"

        return send_file(
            cv_pdf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{safe_name}_cold_email_outreach.pdf",
        )
    except Exception as error:
        return render_template("result.html", error=f"Unable to generate cold email PDF: {error}")

@app.route("/run-job-alerts", methods=["POST", "GET"])
def run_job_alerts_route():
    token = (request.args.get("token") or request.headers.get("X-Job-Alert-Token") or "").strip()
    if JOB_ALERT_RUN_TOKEN and token != JOB_ALERT_RUN_TOKEN:
        return {"ok": False, "error": "Unauthorized"}, 401

    limit_raw = (request.args.get("limit") or "").strip()
    limit = None
    if limit_raw:
        try:
            limit = max(1, int(limit_raw))
        except ValueError:
            limit = None

    report = run_due_job_alerts(limit=limit)
    return {"ok": True, **report}

def start_server_with_port_fallback():
    host = os.getenv("HOST", "0.0.0.0")
    base_port = int(os.getenv("PORT", "8000"))
    max_attempts = 20

    for port in range(base_port, base_port + max_attempts):
        try:
            print(f"Starting server on http://{host}:{port}")
            serve(app, host=host, port=port)
            return
        except OSError as error:
            if getattr(error, "errno", None) == 98 and port < base_port + max_attempts - 1:
                print(f"Port {port} is already in use. Trying port {port + 1}...")
                continue
            raise


if __name__ == "__main__":
    start_server_with_port_fallback()
