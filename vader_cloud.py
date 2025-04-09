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

# Download NLTK data
nltk.download("vader_lexicon")
nltk.download("stopwords")

# Initialize spaCy for tokenization
nlp = spacy.load("en_core_web_sm")

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Set up stopwords
stop_words = set(stopwords.words("english"))

# Use the existing directories from the second script
results_dir = "/Users/charleenadams/nyt/results_deep_learning4"
figures_dir = "/Users/charleenadams/nyt/figures_deep_learning4"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Use the existing logging setup from the second script
log_file = os.path.join(results_dir, "analysis_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger()

# Step 1: Load and filter the data (same as the second script)
logger.info("Loading metadata from nyt_articles_metadata.db...")
conn = sqlite3.connect("/Users/charleenadams/nyt/nyt_articles_metadata.db")
df = pd.read_sql_query("SELECT * FROM articles", conn)
conn.close()
logger.info(f"Loaded {len(df)} articles from the database.")

# Define Israeli and Palestinian terms for filtering (same as the second script)
filter_terms = ["Anti-semitism", "Antisemitism", "Genocide", "Holocaust", "Terrorist", "Terror", "UNRWA", "Hamas", "Israel", "Israeli", "Israelis", "Jew", "Jewish", "Jews", "Zionist", "Zionism", "Palestine", "Palestinian", "Palestinians", "Gaza", "Gazans", "IDF", "Netanyahu", "West Bank"]

# Filter headlines that mention relevant terms
logger.info("Filtering headlines that mention relevant terms...")
df_filtered = df[df["headline"].str.lower().str.contains("|".join([term.lower() for term in filter_terms]), na=False)]
logger.info(f"After filtering, {len(df_filtered)} articles remain.")

# Define Israeli and Palestinian terms for mention counting (same as the second script)
israeli_terms = ["Israel", "Israeli", "IDF", "Israelis"]
palestinian_terms = ["Palestinian", "Palestine", "Hamas", "Gaza"]

# Step 2: VADER Sentiment Analysis on the Filtered Subset
logger.info("Performing VADER Sentiment Analysis on the filtered subset...")
israeli_sentiments = []
palestinian_sentiments = []
israeli_mentions = 0
palestinian_mentions = 0

for headline in df_filtered["headline"]:
    sentiment_scores = sia.polarity_scores(headline)
    compound_score = sentiment_scores["compound"]

    headline_lower = headline.lower()
    has_israeli = any(term.lower() in headline_lower for term in israeli_terms)
    has_palestinian = any(term.lower() in headline_lower for term in palestinian_terms)

    if has_israeli:
        israeli_sentiments.append(compound_score)
        israeli_mentions += 1
    if has_palestinian:
        palestinian_sentiments.append(compound_score)
        palestinian_mentions += 1

# Calculate average sentiment
avg_israeli_sentiment = sum(israeli_sentiments) / len(israeli_sentiments) if israeli_sentiments else 0
avg_palestinian_sentiment = sum(palestinian_sentiments) / len(palestinian_sentiments) if palestinian_sentiments else 0

# Save sentiment results to a file
with open(os.path.join(results_dir, "vader_sentiment_analysis.txt"), "w") as f:
    f.write("VADER Sentiment Analysis Results (Filtered Subset):\n")
    f.write(f"Average sentiment for headlines mentioning Israeli terms: {avg_israeli_sentiment:.3f}\n")
    f.write(f"Average sentiment for headlines mentioning Palestinian terms: {avg_palestinian_sentiment:.3f}\n")
    f.write(f"Total mentions of Israeli terms: {israeli_mentions}\n")
    f.write(f"Total mentions of Palestinian terms: {palestinian_mentions}\n")
    f.write(f"Ratio of mentions (Israeli:Palestinian): {israeli_mentions/palestinian_mentions:.2f}\n" if palestinian_mentions else "N/A\n")

logger.info(f"Average sentiment for Israeli mentions: {avg_israeli_sentiment:.3f}")
logger.info(f"Average sentiment for Palestinian mentions: {avg_palestinian_sentiment:.3f}")
logger.info(f"Total Israeli mentions: {israeli_mentions}")
logger.info(f"Total Palestinian mentions: {palestinian_mentions}")

# Visualization: Sentiment Comparison
plt.figure(figsize=(8, 6))
sns.barplot(x=["Israeli", "Palestinian"], y=[avg_israeli_sentiment, avg_palestinian_sentiment], palette="coolwarm")
plt.title("Average VADER Sentiment in Filtered Headlines", fontsize=14)
plt.ylabel("Average Sentiment Score", fontsize=12)
plt.ylim(-1, 1)
plt.savefig(os.path.join(figures_dir, "vader_sentiment_comparison.png"), dpi=600)
plt.close()
logger.info("Saved VADER sentiment comparison plot to figures_deep_learning4/vader_sentiment_comparison.png")

# Visualization: Sentiment Distribution
plt.figure(figsize=(10, 6))
sns.histplot(israeli_sentiments, bins=20, kde=True, color="blue", label="Israeli Mentions")
sns.histplot(palestinian_sentiments, bins=20, kde=True, color="red", label="Palestinian Mentions")
plt.title("VADER Sentiment Distribution in Filtered Headlines", fontsize=14)
plt.xlabel("Sentiment Score", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.savefig(os.path.join(figures_dir, "vader_sentiment_distribution.png"), dpi=600)
plt.close()
logger.info("Saved VADER sentiment distribution plot to figures_deep_learning4/vader_sentiment_distribution.png")

# Step 3: Word Cloud Analysis on the Filtered Subset
logger.info("Performing Word Cloud Analysis on the filtered subset...")
all_words = []
israeli_cooccur = []
palestinian_cooccur = []

for headline in df_filtered["headline"]:
    doc = nlp(headline)
    words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop and token.text.lower() not in stop_words]
    all_words.extend(words)

    headline_lower = headline.lower()
    has_israeli = any(term.lower() in headline_lower for term in israeli_terms)
    has_palestinian = any(term.lower() in headline_lower for term in palestinian_terms)

    if has_israeli:
        israeli_cooccur.extend(words)
    if has_palestinian:
        palestinian_cooccur.extend(words)

# Keyword frequency
word_freq = Counter(all_words)
israeli_cooccur_freq = Counter(israeli_cooccur)
palestinian_cooccur_freq = Counter(palestinian_cooccur)

# Save keyword frequency results
with open(os.path.join(results_dir, "keyword_frequency_filtered.txt"), "w") as f:
    f.write("Top 20 Most Frequent Words in Filtered Headlines:\n")
    for word, freq in word_freq.most_common(20):
        f.write(f"{word}: {freq}\n")
    f.write("\nTop 20 Words Co-Occurring with Israeli Terms in Filtered Headlines:\n")
    for word, freq in israeli_cooccur_freq.most_common(20):
        f.write(f"{word}: {freq}\n")
    f.write("\nTop 20 Words Co-Occurring with Palestinian Terms in Filtered Headlines:\n")
    for word, freq in palestinian_cooccur_freq.most_common(20):
        f.write(f"{word}: {freq}\n")

# Generate and save word clouds
wordcloud_all = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(word_freq))
wordcloud_israeli = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(israeli_cooccur_freq))
wordcloud_palestinian = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(palestinian_cooccur_freq))

# Word Cloud: All Filtered Headlines
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_all, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud: All Filtered Headlines", fontsize=14)
plt.savefig(os.path.join(figures_dir, "wordcloud_all_filtered.png"), dpi=600)
plt.close()
logger.info("Saved word cloud for all filtered headlines to figures_deep_learning4/wordcloud_all_filtered.png")

# Word Cloud: Israeli Co-Occurrence
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_israeli, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud: Israeli Co-Occurrence (Filtered)", fontsize=14)
plt.savefig(os.path.join(figures_dir, "wordcloud_israeli_filtered.png"), dpi=600)
plt.close()
logger.info("Saved word cloud for Israeli co-occurrence to figures_deep_learning4/wordcloud_israeli_filtered.png")

# Word Cloud: Palestinian Co-Occurrence
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_palestinian, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud: Palestinian Co-Occurrence (Filtered)", fontsize=14)
plt.savefig(os.path.join(figures_dir, "wordcloud_palestinian_filtered.png"), dpi=600)
plt.close()
logger.info("Saved word cloud for Palestinian co-occurrence to figures_deep_learning4/wordcloud_palestinian_filtered.png")

# Log completion
logger.info("VADER sentiment analysis and word cloud analysis on filtered subset complete!")
logger.info(f"Results saved to {results_dir} and figures saved to {figures_dir}")
