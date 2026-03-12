# Quick Start Guide - Job Recommendations

## What's New? 🎉

Your AI Career Platform now recommends relevant jobs from multiple platforms based on your resume!

## How to Use

### Step 1: Upload Your Resume
1. Go to the application homepage
2. Upload your resume (PDF format)
3. Optionally add your email and job description
4. Click "Analyze"

### Step 2: View Your Results
You'll see:
- **ATS Score**: How well your resume passes screening systems
- **Skills Extracted**: Technical skills found in your resume
- **Job Role Matches**: Best-fit career paths
- **Course Recommendations**: Skills to improve
- **✨ Job Recommendations**: Relevant jobs from multiple platforms

### Step 3: Apply to Jobs
- Browse recommended jobs
- Click on any job title to view details
- Click "Apply Now" to visit the application page
- Apply directly on the employer's website

## Getting More Jobs

The platform works out of the box with demo jobs. To see real jobs:

### Option 1: Use Free APIs (Recommended)
These work without extra setup:
- Remotive.io (remote jobs)
- USAJobs (government positions)

### Option 2: Add Premium APIs (More Results)
For more job listings, get free API keys:

#### Adzuna (250 calls/month free)
1. Go to https://developer.adzuna.com/
2. Sign up for free
3. Create an application
4. Copy App ID and App Key
5. Add to `.env`:
   ```
   ADZUNA_APP_ID=your_app_id_here
   ADZUNA_APP_KEY=your_app_key_here
   ```

#### RapidAPI JSearch (2,500 calls/month free)
1. Go to https://rapidapi.com/
2. Sign up for free
3. Search for "JSearch" and subscribe
4. Copy your API key
5. Add to `.env`:
   ```
   RAPIDAPI_KEY=your_rapidapi_key_here
   ```

## Understanding Job Results

### Job Card Information
Each job shows:
- **Title**: Position name
- **Company**: Employer
- **Location**: City/State or Remote
- **Salary**: Compensation range
- **Description**: Brief overview
- **Source**: Which platform it's from

### Job Sources
- **Remotive**: Remote-first positions
- **Adzuna**: Global job aggregator
- **JSearch**: Meta-search across LinkedIn, Indeed, etc.
- **USAJobs**: US Government positions
- **Demo**: Sample listings (when no APIs configured)

## Tips for Best Results

### Resume Optimization
1. **Include Keywords**: Add relevant technical skills
2. **Clear Format**: Use standard section headings
3. **Experience Section**: Mention internships and projects
4. **Optimal Length**: Keep it 300-700 words

### More Relevant Jobs
Jobs are matched based on:
- Skills found in your resume
- Your top job role match
- Industry keywords

Improve your resume skills to get better matches!

## Troubleshooting

### No Real Jobs Showing?
- **Demo jobs appear**: API keys not configured
- **Solution**: Add API keys to `.env` file
- **Alternative**: Free APIs may be temporarily down

### Incorrect Job Matches?
- **Cause**: Resume skills not clearly stated
- **Solution**: Update resume with explicit skill names
- **Example**: Write "Python" not "Programming language"

### Slow Loading?
- Normal: 2-5 seconds for API calls
- If slower: Check internet connection
- APIs have timeouts to prevent long waits

## Privacy & Security

### Your Data
- Resumes are analyzed locally
- Not stored permanently
- Not shared with third parties

### External Links
- Job links go to original employer sites
- Apply directly through those platforms
- We don't collect application data

## Need Help?

### Resources
- **Full Documentation**: See `JOB_RECOMMENDATION_DOCS.md`
- **API Setup Guide**: See `.env.example`
- **Test APIs**: Run `python test_job_apis.py`
- **GitHub Issues**: Report bugs or request features

### Common Questions

**Q: Are these real jobs?**
A: Yes, when APIs are configured. Demo jobs appear as placeholders when APIs aren't set up.

**Q: Do I need to pay for APIs?**
A: No! Free tiers are generous enough for personal use.

**Q: How often are jobs updated?**
A: Jobs are fetched fresh each time you analyze a resume.

**Q: Can I filter jobs by location/salary?**
A: Not yet, but it's on the roadmap!

**Q: Do you take a commission on applications?**
A: No, we're not affiliated with employers. This is a free tool.

## What's Next?

Planned features:
- [ ] Save favorite jobs
- [ ] Email job alerts
- [ ] Location and salary filters
- [ ] Application tracking
- [ ] More job sources
- [ ] Job relevancy scoring

## Feedback

Help us improve! Please share:
- Which job sources are most useful
- What features you'd like
- Any bugs or issues

Thank you for using AI Career Platform! 🚀
