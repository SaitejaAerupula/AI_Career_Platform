# Job Recommendation Feature Documentation

## Overview
The AI Career Platform now includes an intelligent job recommendation feature that fetches relevant job listings from multiple platforms based on your resume analysis.

## How It Works

### 1. Resume Analysis
When you upload a resume, the system:
- Extracts skills using NLP (spaCy)
- Matches skills against job roles
- Identifies the best-fit career paths

### 2. Job Search
Based on the analysis, the system:
- Creates search queries from your top skills and matched job roles
- Fetches jobs from multiple platforms simultaneously
- Aggregates and deduplicates results

### 3. Job Sources

#### Free Sources (No API Key Required)
1. **Remotive.io**
   - Remote jobs from various industries
   - Completely free, no registration needed
   - Focus on tech and creative roles

2. **USAJobs**
   - US Government job listings
   - Free public API
   - Wide range of positions

#### Premium Sources (Optional API Keys)
3. **Adzuna**
   - Global job search engine
   - Free tier: 250 API calls/month
   - Sign up at: https://developer.adzuna.com/
   - Covers multiple countries and industries

4. **RapidAPI JSearch**
   - Aggregates jobs from multiple sources (Indeed, LinkedIn, etc.)
   - Free tier: 2,500 requests/month
   - Sign up at: https://rapidapi.com/
   - Subscribe to JSearch API for comprehensive results

## Setup Instructions

### Without API Keys (Basic)
The platform works immediately with:
- Remotive (remote jobs)
- USAJobs (government positions)

No setup required!

### With API Keys (Enhanced)
For more job listings, add API keys:

1. Create a `.env` file in the project root
2. Add your API keys:
```env
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_app_key
RAPIDAPI_KEY=your_rapidapi_key
```

### Getting API Keys

#### Adzuna
1. Visit https://developer.adzuna.com/
2. Sign up for a free account
3. Create an application
4. Copy your App ID and App Key

#### RapidAPI
1. Visit https://rapidapi.com/
2. Create a free account
3. Search for "JSearch" API
4. Subscribe to the free tier
5. Copy your RapidAPI Key from the dashboard

## Features

### Job Display
Each job listing shows:
- **Title**: Job position name (clickable link)
- **Company**: Employer name
- **Location**: City/State or "Remote"
- **Salary**: Salary range or "Not specified"
- **Description**: Brief job description (first 200 chars)
- **Source**: Which platform the job is from
- **Apply Button**: Direct link to application page

### Search Algorithm
The system:
1. Uses your top 5 extracted skills
2. Adds your best-matched job role (e.g., "Data Scientist")
3. Searches each platform with these keywords
4. Fetches up to 5 jobs per platform
5. Removes duplicates based on title and company
6. Returns the top 15 most relevant jobs

### Error Handling
- If an API is unavailable, the system continues with other sources
- If no API keys are provided, it uses free sources only
- All API timeouts are set to 5 seconds to prevent delays
- Errors are logged but don't break the analysis flow

## Technical Details

### API Integration
- Each job source has its own dedicated function
- Requests are made sequentially with small delays (0.1s) between calls
- Response data is normalized to a common format
- All external links open in new tabs for security

### Performance
- API calls are made only after resume analysis completes
- Maximum 5 jobs fetched per source
- Total processing time: typically 2-5 seconds
- Results are cached during the session

### Security
- API keys are stored in environment variables
- Never expose credentials in frontend
- All external job links use `rel="noopener noreferrer"`
- HTTPS is recommended for production deployment

## Troubleshooting

### No Jobs Showing
1. Check if your resume contains recognizable skills
2. Verify API keys are correctly set in `.env`
3. Check terminal/console for API error messages
4. Ensure internet connection is active

### API Rate Limits
Free tiers have monthly limits:
- Adzuna: 250 calls/month
- RapidAPI JSearch: 2,500 calls/month

Monitor your usage on respective dashboards.

### Slow Response
- Normal response time: 2-5 seconds
- If slower, check your internet connection
- Some APIs may be temporarily slow
- Consider disabling slower APIs if needed

## Future Enhancements

Potential additions:
- Save favorite jobs
- Email job alerts
- Application tracking
- More job sources (LinkedIn, Indeed Direct, Glassdoor)
- Job relevancy scoring
- Advanced filtering (salary range, location, remote only)
- Job matching percentage

## Code Structure

### Main Functions

```python
# Fetch jobs from Adzuna
fetch_adzuna_jobs(keywords, location, max_results)

# Fetch jobs from RapidAPI JSearch
fetch_rapidapi_jobs(keywords, location, max_results)

# Fetch remote jobs
fetch_github_jobs(keywords, max_results)

# Fetch government jobs
fetch_usajobs(keywords, max_results)

# Main aggregator
aggregate_job_recommendations(skills, job_roles_matched, max_per_source)
```

### Flow
1. User uploads resume → `/analyze` endpoint
2. Skills extracted → NLP processing
3. Job roles matched → Scoring algorithm
4. Jobs fetched → `aggregate_job_recommendations()`
5. Results displayed → `result.html` template

## Support

For issues or questions:
1. Check the logs for API errors
2. Verify your API keys are valid
3. Open an issue on GitHub
4. Contact support

## License

This feature is part of the AI Career Platform and follows the same license terms.
