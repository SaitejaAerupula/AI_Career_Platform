from flask import Flask, render_template, request
import spacy
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

# ---- Skill List ----
skill_keywords = [
    "python", "java", "c++", "machine learning",
    "data science", "sql", "html", "css", "javascript", "flask"
]

# ---- ML Model (Same as Colab) ----
data = {
    "skills_count": [2, 4, 6, 8, 10],
    "experience": [0, 1, 2, 3, 4],
    "selected": [0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[["skills_count", "experience"]]
y = df["selected"]

model = LogisticRegression()
model.fit(X, y)

# ---- Helper Functions ----
def extract_text(pdf):
    try:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.lower()
    except Exception as e:
        # Handle corrupted PDFs or invalid files
        raise ValueError(f"Error reading PDF: {str(e)}. Please ensure the PDF file is valid and not corrupted.")

def extract_skills(text):
    return [skill for skill in skill_keywords if skill in text]

def career_readiness(skills):
    return min(len(skills) * 10, 100)
def keyword_match_score(resume_text, job_desc, keywords):
    matched = [kw for kw in keywords if kw in resume_text and kw in job_desc]
    if len(keywords) == 0:
        return 0
    return (len(matched) / len(keywords)) * 40
def skills_section_score(skills):
    if len(skills) >= 8:
        return 25
    elif len(skills) >= 5:
        return 18
    elif len(skills) >= 3:
        return 10
    else:
        return 5
def experience_score(resume_text):
    experience_keywords = ["experience", "internship", "project"]
    count = sum(1 for kw in experience_keywords if kw in resume_text)
    return min(count * 7, 20)
def resume_length_score(resume_text):
    words = len(resume_text.split())
    if 300 <= words <= 700:
        return 15
    elif 200 <= words <= 900:
        return 10
    else:
        return 5
def calculate_ats_score(resume_text, skills, job_desc, keywords):
    score = 0
    score += keyword_match_score(resume_text, job_desc, keywords)
    score += skills_section_score(skills)
    score += experience_score(resume_text)
    score += resume_length_score(resume_text)
    return round(score, 2)
def ats_feedback(ats_score):
    if ats_score >= 80:
        return "Your resume is highly ATS-friendly."
    elif ats_score >= 60:
        return "Your resume is moderately ATS-friendly. Add more role-specific keywords."
    else:
        return "Your resume needs improvement to pass ATS filters."


# NEW: Recommendation Function
def recommend_courses(missing_skills):
    recommendations = {}
    for skill in missing_skills:
        # You can customize these strings or link to actual URLs
        recommendations[skill] = f"Take a professional certification course in {skill.title()}"
    return recommendations

# ---- Routes ----
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files["resume"]
        
        # Validate file exists and has a filename
        if not file or file.filename == '':
            return render_template("result.html", error="No file selected. Please upload a PDF file.")
        
        # Validate file extension
        if not file.filename.lower().endswith('.pdf'):
            return render_template("result.html", error="Invalid file type. Please upload a PDF file.")
        
        text = extract_text(file)
        
        # Check if text is empty
        if not text or len(text.strip()) == 0:
            return render_template("result.html", error="Could not extract text from PDF. Please ensure the PDF is readable.")
        
        # 1. Identify skills found in the resume
        skills = extract_skills(text)
        score = career_readiness(skills)

        # 2. Identify missing skills (Difference between Master List and Resume)
        missing_skills = list(set(skill_keywords) - set(skills))
        
        # 3. Get recommendations for those missing skills
        recommendations = recommend_courses(missing_skills)
        job_desc = "python machine learning flask sql data science"
        
        ats_score = calculate_ats_score(
            text,           # full resume text
            skills,         # extracted skills list
            job_desc,       # target job description
            skill_keywords  # your master list
        )
        
        ats_message = ats_feedback(ats_score)

        # 6. ML Prediction (Conceptual)
        # Ensure 'model' is loaded globally if using machine learning
        prediction_data = pd.DataFrame({
            'skills_count': [len(skills)],
            'experience': [1]
        })
        prediction = model.predict_proba(prediction_data)[0][1] * 100
        
        # Calculate simple match percentage for display
        match_percent = (len(skills) / len(skill_keywords)) * 100

        # 4. Pass everything to result.html
        return render_template(
            "result.html",
            skills=skills,
            score=score,
            match=round(match_percent, 2),
            prediction=round(prediction, 2),
            recommendations=recommendations,  
            ats_score=ats_score,
            ats_message=ats_message,
            # <--- Added this
        )
    except ValueError as ve:
        return render_template("result.html", error=str(ve))
    except Exception as e:
        return render_template("result.html", error=f"An unexpected error occurred: {str(e)}. Please try uploading a different file.")
from waitress import serve

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=3000)
