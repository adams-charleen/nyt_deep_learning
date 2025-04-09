# Import necessary libraries
import pandas as pd
import sqlite3
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import logging
import statistics
from scipy.stats import ttest_ind

# Download NLTK data
nltk.download("vader_lexicon")
nltk.download("stopwords")

# Initialize spaCy for tokenization
nlp = spacy.load("en_core_web_sm")

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Set up stopwords
stop_words = set(stopwords.words("english"))

# Set up directories
results_dir = "/Users/charleenadams/nyt/results_deep_learning4"
figures_dir = "/Users/charleenadams/nyt/figures_deep_learning4"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Set up logging
log_file = os.path.join(results_dir, "analysis_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Load data
logger.info("Loading metadata from nyt_articles_metadata.db...")
conn = sqlite3.connect("/Users/charleenadams/nyt/nyt_articles_metadata.db")
df = pd.read_sql_query("SELECT * FROM articles", conn)
conn.close()
logger.info(f"Loaded {len(df)} articles from the database.")

# Filter relevant terms
filter_terms = ["Anti-semitism", "Antisemitism", "Genocide", "Holocaust", "Terrorist", "Terror", "UNRWA", "Hamas", "Israel", "Israeli", "Israelis", "Jew", "Jewish", "Jews", "Zionist", "Zionism", "Palestine", "Palestinian", "Palestinians", "Gaza", "Gazans", "IDF", "Netanyahu", "West Bank"]
df_filtered = df[df["headline"].str.lower().str.contains("|".join([term.lower() for term in filter_terms]), na=False)]
logger.info(f"After filtering, {len(df_filtered)} articles remain.")

# Terms for analysis
israeli_terms = ["Israel", "Israeli", "IDF", "Israelis"]
palestinian_terms = ["Palestinian", "Palestine", "Hamas", "Gaza"]

# VADER Sentiment Analysis
logger.info("Performing VADER Sentiment Analysis on the filtered subset...")
israeli_sentiments = []
palestinian_sentiments = []
sentiment_records = []
israeli_mentions = 0
palestinian_mentions = 0

for headline in df_filtered["headline"]:
    sentiment_scores = sia.polarity_scores(headline)
    compound_score = sentiment_scores["compound"]
    headline_lower = headline.lower()
    has_israeli = any(term.lower() in headline_lower for term in israeli_terms)
    has_palestinian = any(term.lower() in headline_lower for term in palestinian_terms)

    label = "Neither"
    if has_israeli:
        israeli_sentiments.append(compound_score)
        israeli_mentions += 1
        label = "Israeli"
    if has_palestinian:
        palestinian_sentiments.append(compound_score)
        palestinian_mentions += 1
        label = "Palestinian" if label == "Neither" else "Both"

    sentiment_records.append({
        "headline": headline,
        "compound_score": compound_score,
        "mention_type": label
    })

# Save per-headline sentiment scores
sentiment_df = pd.DataFrame(sentiment_records)
sentiment_df.to_csv(os.path.join(results_dir, "vader_scores_per_headline.csv"), index=False)
logger.info("Saved VADER scores per headline to vader_scores_per_headline.csv")

# Descriptive stats
def describe_scores(scores):
    return {
        "count": len(scores),
        "mean": round(statistics.mean(scores), 3),
        "min": round(min(scores), 3),
        "max": round(max(scores), 3),
        "range": round(max(scores) - min(scores), 3),
        "stdev": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0
    }

israeli_stats = describe_scores(israeli_sentiments)
palestinian_stats = describe_scores(palestinian_sentiments)

# Save summary to file
with open(os.path.join(results_dir, "vader_sentiment_analysis.txt"), "w") as f:
    f.write("VADER Sentiment Analysis Results (Filtered Subset):\n\n")
    f.write("Israeli Mentions:\n")
    for key, val in israeli_stats.items():
        f.write(f"  {key}: {val}\n")
    f.write("\nPalestinian Mentions:\n")
    for key, val in palestinian_stats.items():
        f.write(f"  {key}: {val}\n")
    f.write(f"\nRatio of mentions (Israeli:Palestinian): {israeli_mentions/palestinian_mentions:.2f}\n" if palestinian_mentions else "N/A\n")

logger.info(f"Israeli stats: {israeli_stats}")
logger.info(f"Palestinian stats: {palestinian_stats}")

# Welch's t-test
t_stat, p_val = ttest_ind(israeli_sentiments, palestinian_sentiments, equal_var=False)
logger.info(f"Welch’s t-test: T-statistic: {t_stat:.3f}, P-value: {p_val:.4f}")
with open(os.path.join(results_dir, "vader_sentiment_test.txt"), "w") as f:
    f.write("Welch’s t-test for VADER Sentiment Scores:\n")
    f.write(f"T-statistic: {t_stat:.3f}\n")
    f.write(f"P-value: {p_val:.4f}\n")

# Visualization: Comparison and distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=["Israeli", "Palestinian"], y=[israeli_stats['mean'], palestinian_stats['mean']], palette="coolwarm")
plt.title("Average VADER Sentiment in Filtered Headlines", fontsize=14)
plt.ylabel("Average Sentiment Score", fontsize=12)
plt.ylim(-1, 1)
plt.savefig(os.path.join(figures_dir, "vader_sentiment_comparison.png"), dpi=600)
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(israeli_sentiments, bins=20, kde=True, color="blue", label="Israeli Mentions")
sns.histplot(palestinian_sentiments, bins=20, kde=True, color="red", label="Palestinian Mentions")
plt.title("VADER Sentiment Distribution in Filtered Headlines", fontsize=14)
plt.xlabel("Sentiment Score", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.savefig(os.path.join(figures_dir, "vader_sentiment_distribution.png"), dpi=600)
plt.close()
logger.info("Saved VADER sentiment distribution and comparison plots")
