# AI Career Platform

An intelligent career platform that analyzes resumes, provides ATS scores, and recommends relevant jobs from multiple job boards.

## Features

### 🎯 Resume Analysis
- **ATS Score Calculation**: Get a more realistic Applicant Tracking System (ATS) score for your resume
- **Skill Extraction**: AI-powered extraction of technical skills using NLP
- **Job Role Matching**: Match your skills against popular job roles
- **Blank/Scanned PDF Handling**: If a PDF has no readable text, the platform reports that clearly and can use OCR fallback when available
- **Career Readiness Score**: Predictive analysis of job success probability

### 📄 Cold Email PDF Export
- Generate a recruiter-ready cold email from uploaded resume analysis
- Download the generated cold email as a PDF directly from the results page
- Uses `reportlab` as primary PDF engine with `fpdf2` fallback for runtime compatibility

### 💼 Job Recommendations
- **Live Multi-Platform Job Search**: Fetch recent jobs from:
   - **LinkedIn**
   - **Naukri**
   - **Glassdoor**
- **Recent Posting Focus**: The platform prioritizes recently indexed job postings
- **Smart Matching**: Jobs are recommended based on your extracted skills and matched roles
- **Direct Application**: Apply to jobs directly from the platform with external links
- **Comprehensive Details**: View job title, company, location, salary, and description

### 📚 Course Recommendations
- Get personalized course recommendations for missing skills
- Improve your profile to match desired job roles

### 📧 Email Reports
- Receive detailed analysis results via email
- Track your progress over time
- Email sending is optional and configured through environment variables

## Setup

### Prerequisites
- Python 3.8+
- pip package manager
- (Optional for OCR) `tesseract-ocr` and `poppler-utils` system packages

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SaitejaAerupula/AI_Career_Platform.git
cd AI_Career_Platform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Enable OCR for scanned/image-only PDFs:
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

3. Configure optional search tuning in `.env` for live job discovery:
```bash
cp .env.example .env
# Edit .env and adjust these values if needed
JOB_SEARCH_LOCATION=India
JOB_SEARCH_REGION=in-en
JOB_RECENCY_DAYS=7
```

5. (Optional) Configure email delivery in `.env` if you want users to receive analysis reports by email:
```bash
SENDER_EMAIL=your_email@example.com
SENDER_PASSWORD=your_app_password
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
```

### Running the Application

```bash
python app1.py
```

The application will be available at `http://localhost:8000`

## Usage

1. **Upload Resume**: Upload your resume in PDF format
2. **Enter Email** (Optional): Receive results via email
3. **Job Description** (Optional): Add a job description for targeted ATS matching against a real role
4. **Analyze**: Click analyze to get:
   - ATS Score and feedback
   - Blank-PDF or image-only PDF detection
   - OCR extraction status when scanned PDFs are uploaded
   - Extracted skills
   - Resume section and contact-signal checks
   - Job role matches
   - **Cold email PDF download option**
   - **Recent live jobs from LinkedIn, Naukri, and Glassdoor**
   - Course recommendations for missing skills

## Deploy to Vercel

This repository is configured to deploy the Flask app on Vercel using:
- `api/index.py` as the Vercel Python entrypoint
- `api/requirements.txt` to ensure the Vercel Python runtime installs project dependencies (including `reportlab` and `fpdf2` for cold email PDF export)
- `vercel.json` for route rewrites so all paths go to Flask

### 1. Push your latest code to GitHub

```bash
git add .
git commit -m "chore: prepare vercel deployment"
git push origin main
```

### 2. Import project in Vercel

1. Open Vercel dashboard
2. Click **Add New Project**
3. Import `SaitejaAerupula/AI_Career_Platform`
4. Keep the **Root Directory** as project root
5. Deploy

### 3. Add environment variables in Vercel Project Settings

Set the same values you use in `.env` (if needed):
- `SENDER_EMAIL`
- `SENDER_PASSWORD`
- `SMTP_HOST`
- `SMTP_PORT`
- `ADZUNA_APP_ID`
- `ADZUNA_APP_KEY`
- `RAPIDAPI_KEY`
- `JOB_SEARCH_LOCATION`
- `JOB_SEARCH_REGION`
- `JOB_RECENCY_DAYS`

### 4. Redeploy after env updates

Use **Deployments -> Redeploy** in Vercel after changing environment variables.

### Important Vercel Notes

- OCR on Vercel is usually limited because system binaries like `tesseract`/`pdftoppm` are not preinstalled.
- For full OCR support, keep using a VM/container host (for example Render) or move OCR into a separate service.

## Current PDF Notes

- Text-based PDFs are fully supported.
- Blank PDFs are detected and reported as having no readable text.
- Scanned image-only PDFs can be analyzed when OCR dependencies are installed in the runtime environment.

## Project Structure

```
AI_Career_Platform/
├── app1.py                 # Main Flask application with job recommendation logic
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment variables
├── templates/
│   ├── index.html         # Resume upload page
│   └── result.html        # Results page with job recommendations
├── frontend/              # React frontend (optional)
└── README.md
```

## Technologies Used

- **Backend**: Flask, Python
- **NLP**: spaCy for skill extraction
- **ML**: Scikit-learn for predictive analysis
- **Job APIs**: Adzuna, RapidAPI JSearch, Remotive, USAJobs
- **Email**: SMTP for result delivery
- **PDF Processing**: PyPDF2
- **OCR (Optional)**: pdf2image + pytesseract (with poppler/tesseract system binaries)
- **CV PDF Export**: ReportLab

## Features in Detail

### Job Recommendation Engine
The platform aggregates recent jobs from LinkedIn, Naukri, and Glassdoor:
- Searches based on extracted skills and matched job roles
- Filters for recent postings and sorts by recency hints
- Removes duplicate listings
- Shows job details including title, company, location, salary, and description
- Provides direct links to the original live posting

### ATS Score Components
- **Job Alignment** (35%): Resume vs target job description keyword and skill match
- **Section Completeness** (20%): Detection of sections such as summary, skills, experience, education, and projects
- **Contact Completeness** (10%): Presence of email, phone, and professional links
- **Experience Evidence** (15%): Experience indicators, project evidence, and action verbs
- **Achievements and Metrics** (10%): Quantified impact such as percentages, users, revenue, or project counts
- **Formatting Readability** (10%): Resume length, sentence readability, and list structure

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Support

For issues or questions, please open an issue on GitHub.
