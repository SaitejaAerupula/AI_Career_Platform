"""
Test script for job recommendation feature
Run this to verify that job recommendation APIs are working
"""

import os
from dotenv import load_dotenv
import requests
import time

# Load environment variables
load_dotenv()

def test_remotive_api():
    """Test Remotive.io API (no key required)"""
    print("\n🔍 Testing Remotive.io API...")
    try:
        url = "https://remotive.com/api/remote-jobs"
        params = {"search": "python developer", "limit": 3}
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            jobs = data.get("jobs", [])
            if jobs:
                print(f"✅ SUCCESS: Found {len(jobs)} jobs")
                print(f"   Sample: {jobs[0].get('title')} at {jobs[0].get('company_name')}")
                return True
            else:
                print("⚠️  No jobs found but API responded")
                return False
        else:
            print(f"❌ FAILED: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_usajobs_api():
    """Test USAJobs API (no key required)"""
    print("\n🔍 Testing USAJobs API...")
    try:
        url = "https://data.usajobs.gov/api/search"
        headers = {
            "Host": "data.usajobs.gov",
            "User-Agent": "test@example.com"
        }
        params = {"Keyword": "software engineer", "ResultsPerPage": 3}
        response = requests.get(url, headers=headers, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("SearchResult", {}).get("SearchResultItems", [])
            if items:
                print(f"✅ SUCCESS: Found {len(items)} jobs")
                job = items[0].get("MatchedObjectDescriptor", {})
                print(f"   Sample: {job.get('PositionTitle')} at {job.get('OrganizationName')}")
                return True
            else:
                print("⚠️  No jobs found but API responded")
                return False
        else:
            print(f"❌ FAILED: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_adzuna_api():
    """Test Adzuna API (requires key)"""
    print("\n🔍 Testing Adzuna API...")
    app_id = os.getenv("ADZUNA_APP_ID", "")
    app_key = os.getenv("ADZUNA_APP_KEY", "")
    
    if not app_id or not app_key:
        print("⚠️  SKIPPED: No API credentials found")
        print("   Get credentials at: https://developer.adzuna.com/")
        return None
    
    try:
        url = "https://api.adzuna.com/v1/api/jobs/us/search/1"
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "results_per_page": 3,
            "what": "python developer"
        }
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            jobs = data.get("results", [])
            if jobs:
                print(f"✅ SUCCESS: Found {len(jobs)} jobs")
                print(f"   Sample: {jobs[0].get('title')} at {jobs[0].get('company', {}).get('display_name')}")
                return True
            else:
                print("⚠️  No jobs found but API responded")
                return False
        elif response.status_code == 401:
            print("❌ FAILED: Invalid credentials")
            return False
        else:
            print(f"❌ FAILED: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_rapidapi():
    """Test RapidAPI JSearch (requires key)"""
    print("\n🔍 Testing RapidAPI JSearch...")
    api_key = os.getenv("RAPIDAPI_KEY", "")
    
    if not api_key:
        print("⚠️  SKIPPED: No API key found")
        print("   Get key at: https://rapidapi.com/")
        return None
    
    try:
        url = "https://jsearch.p.rapidapi.com/search"
        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }
        params = {"query": "python developer", "num_pages": "1"}
        response = requests.get(url, headers=headers, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            jobs = data.get("data", [])
            if jobs:
                print(f"✅ SUCCESS: Found {len(jobs)} jobs")
                print(f"   Sample: {jobs[0].get('job_title')} at {jobs[0].get('employer_name')}")
                return True
            else:
                print("⚠️  No jobs found but API responded")
                return False
        elif response.status_code == 403:
            print("❌ FAILED: Invalid API key or subscription")
            return False
        else:
            print(f"❌ FAILED: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print("=" * 60)
    print("Job Recommendation API Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test free APIs (no credentials required)
    results["Remotive"] = test_remotive_api()
    time.sleep(0.5)
    results["USAJobs"] = test_usajobs_api()
    time.sleep(0.5)
    
    # Test premium APIs (credentials required)
    results["Adzuna"] = test_adzuna_api()
    time.sleep(0.5)
    results["RapidAPI"] = test_rapidapi()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    working = [name for name, status in results.items() if status is True]
    failed = [name for name, status in results.items() if status is False]
    skipped = [name for name, status in results.items() if status is None]
    
    print(f"\n✅ Working: {len(working)}")
    for name in working:
        print(f"   - {name}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for name in failed:
            print(f"   - {name}")
    
    if skipped:
        print(f"\n⚠️  Skipped (no credentials): {len(skipped)}")
        for name in skipped:
            print(f"   - {name}")
    
    print("\n" + "=" * 60)
    
    if len(working) >= 1:
        print("🎉 Job recommendation feature is working!")
        print("   At least one job source is available.")
    else:
        print("⚠️  Warning: No job sources are currently working.")
        print("   Please check your internet connection and API credentials.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
