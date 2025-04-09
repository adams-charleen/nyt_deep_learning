import requests
import pandas as pd
import sqlite3
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from collections import Counter
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK data
nltk.download("vader_lexicon")

# Initialize spaCy for entity recognition
nlp = spacy.load("en_core_web_sm")

# NYT API key (your key)
api_key = "INSERT KEY"

# List of user agents to rotate
user_agents = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:129.0) Gecko/20100101 Firefox/129.0",
]

# Step 1: Fetch article metadata using NYT API with retry mechanism
base_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
query = "Israel OR Israeli OR Palestine OR Palestinian"
begin_date = "20231001"  # October 1, 2023
end_date = "20250307"    # March 7, 2025
page = 0
articles = []
request_count = 0

print("Fetching articles from NYT API...")
while True:
    params = {
        "q": query,
        "begin_date": begin_date,
        "end_date": end_date,
        "page": page,
        "api-key": api_key
    }
    
    # Retry mechanism for API requests
    for attempt in range(3):  # Retry up to 3 times
        response = requests.get(base_url, params=params)
        request_count += 1
        logging.info(f"API request {request_count}: Page {page}, Status Code: {response.status_code}")
        if response.status_code == 200:
            break
        elif response.status_code == 429:  # Too many requests
            wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
            print(f"Rate limit hit (429). Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
        else:
            print(f"API error: {response.status_code}")
            break
    else:
        print("Failed to fetch page after retries. Stopping.")
        break

    data = response.json()
    if "response" not in data or not data["response"]["docs"]:
        break

    articles.extend(data["response"]["docs"])
    page += 1
    # Random delay to avoid bot detection (10-12 seconds)
    delay = random.uniform(10, 12)
    print(f"Waiting {delay:.2f} seconds before next API request...")
    time.sleep(delay)

# Extract relevant metadata
data = []
for article in articles:
    headline = article.get("headline", {}).get("main", "")
    web_url = article.get("web_url", "")
    pub_date = article.get("pub_date", "")
    byline = article.get("byline", {}).get("original", "")
    section = article.get("section_name", "")
    data.append({
        "headline": headline,
        "web_url": web_url,
        "pub_date": pub_date,
        "byline": byline,
        "section": section,
        "full_text": ""  # Placeholder for full text
    })

df = pd.DataFrame(data)
print(f"Collected {len(df)} articles.")

# Step 1.5: Save metadata to a temporary database before scraping
print("Saving metadata to temporary database...")
conn = sqlite3.connect("nyt_articles_metadata.db")
df.to_sql("articles", conn, if_exists="replace", index=False)
conn.close()
print("Metadata saved to nyt_articles_metadata.db")
