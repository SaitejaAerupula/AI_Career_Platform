import io
import re
import unittest
from unittest.mock import patch

from PyPDF2 import PdfWriter

from app1 import CV_EXPORT_AVAILABLE, app, calculate_ats_score, extract_skills_nlp, normalize_text

try:
    from reportlab.pdfgen import canvas
except Exception:
    canvas = None


class ResumeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def _build_text_pdf(self, lines):
        if canvas is None:
            self.skipTest("reportlab is not installed")

        pdf_buffer = io.BytesIO()
        document = canvas.Canvas(pdf_buffer)
        y = 790
        for line in lines:
            document.drawString(40, y, line)
            y -= 16
        document.save()
        pdf_buffer.seek(0)
        return pdf_buffer

    def test_blank_pdf_reports_no_readable_text(self):
        pdf_buffer = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=300)
        writer.write(pdf_buffer)
        pdf_buffer.seek(0)

        response = self.client.post(
            "/analyze",
            data={
                "email": "",
                "job_desc": "Backend developer with Python and SQL experience",
                "resume": (pdf_buffer, "blank_resume.pdf"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("No readable text was found in this PDF", page)
        self.assertIn("ATS Score:", page)
        self.assertIn("0%", page)

    def test_targeted_ats_score_rewards_relevant_resume_content(self):
        job_desc = (
            "We are hiring a backend developer with Python, Flask, SQL, Docker, AWS, "
            "Git, REST API experience, and 5 years building production systems."
        )
        strong_resume = """
        John Doe
        john@example.com | 555-123-4567 | linkedin.com/in/johndoe | github.com/johndoe
        Professional Summary
        Backend engineer with 5 years experience building Python and Flask services for cloud platforms.
        Technical Skills
        Python, Flask, SQL, Docker, AWS, Git, REST API
        Professional Experience
        Developed REST API services, improved response time by 35%, reduced deployment failures by 20%,
        and led 12 client projects across Python and SQL systems.
        Projects
        Built a hiring dashboard used by 1000 users.
        Education
        Bachelor of Technology in Computer Science
        """
        weak_resume = """
        Candidate Profile
        Motivated graduate looking for a software role.
        Education
        Bachelor degree completed.
        """

        strong_skills = extract_skills_nlp(normalize_text(strong_resume))
        weak_skills = extract_skills_nlp(normalize_text(weak_resume))

        strong_result = calculate_ats_score(normalize_text(strong_resume), strong_resume, strong_skills, job_desc)
        weak_result = calculate_ats_score(normalize_text(weak_resume), weak_resume, weak_skills, job_desc)

        self.assertGreater(strong_result["ats_score"], weak_result["ats_score"])
        self.assertGreaterEqual(strong_result["alignment"]["score"], 20)
        self.assertIn("python", strong_result["alignment"]["matched_keywords"])
        self.assertIn("docker", strong_result["alignment"]["matched_keywords"])

    def test_cv_pdf_download_link_generates_valid_pdf(self):
        if not CV_EXPORT_AVAILABLE:
            self.skipTest("CV PDF generation dependency is unavailable")

        resume_pdf = self._build_text_pdf(
            [
                "John Doe",
                "john@example.com | 555-123-4567 | linkedin.com/in/johndoe",
                "Professional Summary",
                "Backend engineer with 5 years experience building scalable Python APIs.",
                "Technical Skills",
                "Python, Flask, SQL, Docker, AWS, Git, REST API",
                "Professional Experience",
                "Built API services and reduced latency by 35 percent.",
                "Education",
                "Bachelor of Technology in Computer Science",
            ]
        )

        with patch("app1.aggregate_job_recommendations", return_value=[]):
            analyze_response = self.client.post(
                "/analyze",
                data={
                    "email": "",
                    "job_desc": "Backend engineer with Python, SQL, Docker and REST API experience",
                    "resume": (resume_pdf, "resume.pdf"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(analyze_response.status_code, 200)
        page = analyze_response.get_data(as_text=True)
        match = re.search(r'href="(/download-cv/[a-f0-9]+)"', page)
        self.assertIsNotNone(match, "Expected CV download link in analysis result")

        download_response = self.client.get(match.group(1))
        self.assertEqual(download_response.status_code, 200)
        self.assertIn("application/pdf", download_response.content_type)
        self.assertTrue(download_response.data.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()