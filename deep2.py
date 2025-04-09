import pandas as pd
import sqlite3
import numpy as np
from transformers import BertTokenizer, BertModel, pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import sys
import logging
from scipy.stats import norm  # For the two-sample proportion test
import plotly.express as px  # For interactive PCA plot
import plotly.graph_objects as go  # For bar plot in subplot
from plotly.subplots import make_subplots  # For subplot layout

# Set up directories for results
results_dir = "/Users/charleenadams/nyt/results_deep_learning4"
figures_dir = "/Users/charleenadams/nyt/figures_deep_learning4"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Set up logging to both console and a log file
log_file = os.path.join(results_dir, "analysis_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)  # This ensures output goes to both file and console
    ]
)

# Create a logger
logger = logging.getLogger()

# Step 1: Load metadata from the database
logger.info("Loading metadata from nyt_articles_metadata.db...")
conn = sqlite3.connect("/Users/charleenadams/nyt/nyt_articles_metadata.db")
df = pd.read_sql_query("SELECT * FROM articles", conn)
conn.close()
logger.info(f"Loaded {len(df)} articles from the database.")

# Define Israeli and Palestinian terms for filtering
filter_terms = ["Anti-semitism", "Antisemitism", "Genocide", "Holocaust", "Terrorist", "Terror", "UNRWA", "Hamas", "Israel", "Israeli", "Israelis", "Jew", "Jewish", "Jews", "Zionist", "Zionism", "Palestine", "Palestinian", "Palestinians", "Gaza", "Gazans", "IDF", "Netanyahu", "West Bank"]

# Step 2: Filter headlines that mention relevant terms
logger.info("Filtering headlines that mention relevant terms...")
df_filtered = df[df["headline"].str.lower().str.contains("|".join([term.lower() for term in filter_terms]), na=False)]
logger.info(f"After filtering, {len(df_filtered)} articles remain.")

# Define Israeli and Palestinian terms for mention counting
israeli_terms = ["Israel", "Israeli", "IDF", "Israelis"]
palestinian_terms = ["Palestinian", "Palestine", "Hamas", "Gaza"]

# Step 3: Calculate proportions of Israeli and Palestinian mentions and perform a two-sided significance test
logger.info("Calculating proportions of headlines with Israeli and Palestinian mentions...")
total_headlines = len(df_filtered)

# Count headlines with Israeli and Palestinian mentions
israeli_mentions = df_filtered["headline"].str.lower().str.contains("|".join([term.lower() for term in israeli_terms]), na=False).sum()
palestinian_mentions = df_filtered["headline"].str.lower().str.contains("|".join([term.lower() for term in palestinian_terms]), na=False).sum()

# Calculate proportions
prop_israeli = israeli_mentions / total_headlines
prop_palestinian = palestinian_mentions / total_headlines

logger.info(f"Proportion of headlines with Israeli mentions: {prop_israeli:.3f} ({israeli_mentions}/{total_headlines})")
logger.info(f"Proportion of headlines with Palestinian mentions: {prop_palestinian:.3f} ({palestinian_mentions}/{total_headlines})")

# Perform a two-sided test of significance (two-sample proportion z-test)
# Null hypothesis: prop_israeli = prop_palestinian
# Pooled proportion
pooled_prop = (israeli_mentions + palestinian_mentions) / (2 * total_headlines)
# Standard error
se = np.sqrt(pooled_prop * (1 - pooled_prop) * (2 / total_headlines))
# Z-statistic
z_stat = (prop_israeli - prop_palestinian) / se
# Two-sided p-value
p_value = 2 * (1 - norm.cdf(abs(z_stat)))

# Interpret the result
alpha = 0.05  # Significance level
if p_value < alpha:
    significance_result = f"The difference in proportions is statistically significant (z = {z_stat:.3f}, p = {p_value:.3f}, p < {alpha})"
else:
    significance_result = f"The difference in proportions is not statistically significant (z = {z_stat:.3f}, p = {p_value:.3f}, p >= {alpha})"

logger.info("Two-sided test of significance for proportions:")
logger.info(f"Z-statistic: {z_stat:.3f}")
logger.info(f"P-value: {p_value:.3f}")
logger.info(significance_result)

# Create a bar plot for the proportions with statistical analysis
plt.figure(figsize=(8, 6))
proportions = [prop_israeli, prop_palestinian]
labels = ['Israeli\nMentions', 'Palestinian\nMentions']
plt.bar(labels, proportions, color=['blue', 'red'])
plt.title(f"Proportion of Headlines with Israeli vs. Palestinian Mentions\nTotal Headlines: {total_headlines}", fontsize=14)
plt.ylabel("Proportion of Headlines", fontsize=12)
plt.ylim(0, 1)
for i, prop in enumerate(proportions):
    plt.text(i, prop + 0.02, f"{prop:.3f}", ha='center', fontsize=12)
# Add statistical analysis as text on the plot
stats_text = f"Z-statistic: {z_stat:.3f}\nP-value: {p_value:.3f}"
plt.text(0.5, 0.9, stats_text, ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.savefig(os.path.join(figures_dir, "proportions_mentions.png"), dpi=600)
plt.close()

# Step 4: Load BERT model and tokenizer for embeddings
logger.info("Loading BERT model and tokenizer for embeddings...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # Set to evaluation mode

# Function to get BERT embeddings
def get_bert_embedding(text, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Step 5: Generate embeddings for filtered headlines
logger.info("Generating BERT embeddings for filtered headlines...")
embeddings = []
for headline in df_filtered["headline"]:
    embedding = get_bert_embedding(headline)
    embeddings.append(embedding)
embeddings = np.array(embeddings)
logger.info(f"Generated embeddings with shape: {embeddings.shape}")

# Step 6: Apply K-means clustering
n_clusters = 5  # Number of clusters
logger.info(f"Applying K-means clustering with {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Add cluster labels to the filtered dataframe
df_filtered["cluster"] = cluster_labels

# Step 7: Analyze clusters for Israeli/Palestinian mentions
logger.info("Analyzing clusters for Israeli and Palestinian mentions...")
cluster_summary = {}
cluster_israeli_props = {}
cluster_palestinian_props = {}
for cluster in range(n_clusters):
    cluster_df = df_filtered[df_filtered["cluster"] == cluster]
    israeli_count = cluster_df["headline"].str.lower().str.contains("|".join([term.lower() for term in israeli_terms]), na=False).sum()
    palestinian_count = cluster_df["headline"].str.lower().str.contains("|".join([term.lower() for term in palestinian_terms]), na=False).sum()
    cluster_size = len(cluster_df)
    cluster_israeli_props[cluster] = israeli_count / cluster_size if cluster_size > 0 else 0
    cluster_palestinian_props[cluster] = palestinian_count / cluster_size if cluster_size > 0 else 0
    cluster_summary[cluster] = {
        "size": cluster_size,
        "israeli_mentions": israeli_count,
        "palestinian_mentions": palestinian_count,
        "sample_headlines": cluster_df["headline"].head(5).tolist()
    }

# Save cluster summary to a file
with open(os.path.join(results_dir, "cluster_summary.txt"), "w") as f:
    f.write("Cluster Analysis Results:\n")
    for cluster, info in cluster_summary.items():
        f.write(f"\nCluster {cluster}:\n")
        f.write(f"Size: {info['size']}\n")
        f.write(f"Israeli Mentions: {info['israeli_mentions']}\n")
        f.write(f"Palestinian Mentions: {info['palestinian_mentions']}\n")
        f.write("Sample Headlines:\n")
        for headline in info["sample_headlines"]:
            f.write(f"  - {headline}\n")

# Step 8: Load a pre-trained BERT-based sentiment model (fallback until fine-tuning)
logger.info("Loading pre-trained BERT-based sentiment model...")
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Function to predict sentiment using the BERT-based model
def predict_bert_sentiment(headline):
    result = sentiment_analyzer(headline)[0]
    # The model returns a star rating (1 to 5); convert to a sentiment score (-1 to 1)
    star_rating = int(result['label'].split()[0])  # Extract the number (1 to 5)
    # Map star rating to sentiment score: 1 star = -1, 3 stars = 0, 5 stars = 1
    sentiment_score = (star_rating - 3) / 2  # Linear mapping: 1 -> -1, 3 -> 0, 5 -> 1
    return sentiment_score

# Step 9: Examine sentiment within clusters using BERT-based model
logger.info("Calculating sentiment within clusters using BERT-based model...")
cluster_sentiments = {}
for cluster in range(n_clusters):
    cluster_df = df_filtered[df_filtered["cluster"] == cluster]
    sentiments = [predict_bert_sentiment(headline) for headline in cluster_df["headline"]]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    cluster_sentiments[cluster] = avg_sentiment

# Save sentiment results
with open(os.path.join(results_dir, "cluster_sentiment.txt"), "w") as f:
    f.write("Cluster Sentiment Analysis (BERT-based):\n")
    for cluster, sentiment in cluster_sentiments.items():
        f.write(f"Cluster {cluster}: Average Sentiment = {sentiment:.3f}\n")

# Step 10: Temporal Analysis (Trends Over Time)
logger.info("Performing temporal analysis...")
df_filtered["pub_date"] = pd.to_datetime(df_filtered["pub_date"], errors="coerce")
df_filtered["month"] = df_filtered["pub_date"].dt.to_period("M")

# Analyze mentions and sentiment by month for each cluster
temporal_analysis = {}
for cluster in range(n_clusters):
    cluster_df = df_filtered[df_filtered["cluster"] == cluster]
    monthly_israeli = cluster_df[cluster_df["headline"].str.lower().str.contains("|".join([term.lower() for term in israeli_terms]), na=False)].groupby("month").size()
    monthly_palestinian = cluster_df[cluster_df["headline"].str.lower().str.contains("|".join([term.lower() for term in palestinian_terms]), na=False)].groupby("month").size()
    monthly_sentiment = cluster_df.groupby("month")["headline"].apply(lambda x: sum(predict_bert_sentiment(h) for h in x) / len(x) if len(x) > 0 else 0)
    temporal_analysis[cluster] = {
        "monthly_israeli": monthly_israeli,
        "monthly_palestinian": monthly_palestinian,
        "monthly_sentiment": monthly_sentiment
    }

# Save temporal analysis results and visualize trends (individual and panel plots)
with open(os.path.join(results_dir, "temporal_analysis.txt"), "w") as f:
    f.write("Temporal Analysis Results:\n")
    for cluster, data in temporal_analysis.items():
        f.write(f"\nCluster {cluster}:\n")
        f.write("Monthly Israeli Mentions:\n")
        f.write(data["monthly_israeli"].to_string() + "\n")
        f.write("\nMonthly Palestinian Mentions:\n")
        f.write(data["monthly_palestinian"].to_string() + "\n")
        f.write("\nMonthly Sentiment:\n")
        f.write(data["monthly_sentiment"].to_string() + "\n")

# Create individual plots and panel plots for temporal mentions and sentiments
fig_mentions, axes_mentions = plt.subplots(n_clusters, 1, figsize=(12, 5 * n_clusters), sharex=True)
fig_sentiments, axes_sentiments = plt.subplots(n_clusters, 1, figsize=(12, 5 * n_clusters), sharex=True)

for cluster in range(n_clusters):
    data = temporal_analysis[cluster]

    # Individual plot for mentions
    plt.figure(figsize=(12, 6))
    if not data["monthly_israeli"].empty:
        data["monthly_israeli"].plot(label="Israeli Mentions", color="blue")
    if not data["monthly_palestinian"].empty:
        data["monthly_palestinian"].plot(label="Palestinian Mentions", color="red")
    plt.title(f"Cluster {cluster}: Monthly Mentions Over Time", fontsize=14)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Number of Mentions", fontsize=12)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"temporal_mentions_cluster_{cluster}.png"), dpi=600)
    plt.close()

    # Individual plot for sentiments
    plt.figure(figsize=(12, 6))
    if not data["monthly_sentiment"].empty:
        data["monthly_sentiment"].plot(label="Sentiment", color="green")
    plt.title(f"Cluster {cluster}: Monthly Sentiment Over Time", fontsize=14)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Average Sentiment Score", fontsize=12)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"temporal_sentiment_cluster_{cluster}.png"), dpi=600)
    plt.close()

    # Add to panel plot for mentions
    ax = axes_mentions[cluster] if n_clusters > 1 else axes_mentions
    if not data["monthly_israeli"].empty:
        data["monthly_israeli"].plot(label="Israeli Mentions", color="blue", ax=ax)
    if not data["monthly_palestinian"].empty:
        data["monthly_palestinian"].plot(label="Palestinian Mentions", color="red", ax=ax)
    ax.set_title(f"Cluster {cluster}", fontsize=12)
    ax.set_ylabel("Number of Mentions", fontsize=10)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

    # Add to panel plot for sentiments
    ax = axes_sentiments[cluster] if n_clusters > 1 else axes_sentiments
    if not data["monthly_sentiment"].empty:
        data["monthly_sentiment"].plot(label="Sentiment", color="green", ax=ax)
    ax.set_title(f"Cluster {cluster}", fontsize=12)
    ax.set_ylabel("Average Sentiment Score", fontsize=10)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

# Finalize panel plots
fig_mentions.suptitle("Temporal Mentions Across Clusters", fontsize=16, y=1.02)
fig_mentions.tight_layout()
fig_mentions.savefig(os.path.join(figures_dir, "temporal_mentions_panel.png"), dpi=600, bbox_inches='tight')
plt.close(fig_mentions)

fig_sentiments.suptitle("Temporal Sentiments Across Clusters", fontsize=16, y=1.02)
fig_sentiments.tight_layout()
fig_sentiments.savefig(os.path.join(figures_dir, "temporal_sentiments_panel.png"), dpi=600, bbox_inches='tight')
plt.close(fig_sentiments)

# Step 11: Section Sentiment Analysis
logger.info("Performing section sentiment analysis...")
section_sentiment = {}
for cluster in range(n_clusters):
    cluster_df = df_filtered[df_filtered["cluster"] == cluster]
    section_groups = cluster_df.groupby("section")
    section_sentiment[cluster] = {}
    for section, group in section_groups:
        sentiments = [predict_bert_sentiment(headline) for headline in group["headline"]]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        section_sentiment[cluster][section] = avg_sentiment

# Save section sentiment results
with open(os.path.join(results_dir, "section_sentiment.txt"), "w") as f:
    f.write("Section Sentiment Analysis (BERT-based):\n")
    for cluster in section_sentiment:
        f.write(f"\nCluster {cluster}:\n")
        for section, sentiment in section_sentiment[cluster].items():
            f.write(f"Section {section}: Average Sentiment = {sentiment:.3f}\n")

# Step 12: Visualize Clusters with PCA (Static and Interactive)
logger.info("Visualizing clusters with PCA (static and interactive)...")
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Compute cluster sizes for the legend
cluster_sizes = df_filtered["cluster"].value_counts().sort_index()

# Define cluster descriptions based on analysis
cluster_descriptions = {
    0: "International Actions and Diplomacy",
    1: "Conflict and Violence",
    2: "Protests and Cultural Support",
    3: "Peace Efforts and Politics",
    4: "U.S. Politics and Protests"
}

# Create legend labels with cluster sizes, descriptions, and proportions
legend_labels = [
    f"Cluster {i} ({cluster_sizes[i]} articles): {cluster_descriptions[i]}\nIsraeli Proportion: {cluster_israeli_props[i]:.2f}, Palestinian Proportion: {cluster_palestinian_props[i]:.2f}"
    for i in range(n_clusters)
]

# Static PCA plot using Seaborn with enhanced legend and title
plt.figure(figsize=(10, 8))
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=cluster_labels, palette="deep", legend="full")
plt.title(f"Headline Clusters (PCA Visualization) - {total_headlines} Articles", fontsize=14)
plt.xlabel("PCA Component 1", fontsize=12)
plt.ylabel("PCA Component 2", fontsize=12)
# Update the legend with cluster sizes, descriptions, and proportions
plt.legend(labels=legend_labels, title="Clusters", loc="best", fontsize=10)
plt.savefig(os.path.join(figures_dir, "cluster_visualization.png"), dpi=600)
plt.close()

# Interactive PCA plot using Plotly with enhanced legend, title, larger hover tooltip, and bar plot subplot
df_plotly = pd.DataFrame({
    "PCA Component 1": embeddings_2d[:, 0],
    "PCA Component 2": embeddings_2d[:, 1],
    "Cluster": cluster_labels,
    "Cluster_Label": [
        f"Cluster {label} ({cluster_sizes[label]} articles): {cluster_descriptions[label]}<br>Israeli Proportion: {cluster_israeli_props[label]:.2f}, Palestinian Proportion: {cluster_palestinian_props[label]:.2f}"
        for label in cluster_labels
    ],
    "Headline": df_filtered["headline"]
})

# Create a subplot layout: PCA scatter on the left, bar plot on the right
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.65, 0.35],  # Adjusted widths to reduce space between plots
    horizontal_spacing=0.1,  # Increased spacing between subplots
    subplot_titles=[
        f"Interactive Headline Clusters (PCA Visualization) - {total_headlines} Articles",
        "Proportions of Mentions"
    ]
)

# Add the PCA scatter plot to the left subplot
scatter = px.scatter(
    df_plotly,
    x="PCA Component 1",
    y="PCA Component 2",
    color="Cluster_Label",
    hover_data=["Headline"],
    color_discrete_sequence=px.colors.qualitative.Plotly
)
for trace in scatter.data:
    fig.add_trace(trace, row=1, col=1)

# Add the bar plot to the right subplot
fig.add_trace(
    go.Bar(
        x=[prop_israeli, prop_palestinian],
        y=['Israeli\nMentions', 'Palestinian\nMentions'],
        orientation='h',
        marker_color=['blue', 'red'],
        text=[f"{prop_israeli:.3f}", f"{prop_palestinian:.3f}"],
        textposition='auto',
        name="Proportion of Mentions",  # Relabel "Trace 5" to "Proportion of Mentions"
        showlegend=True
    ),
    row=1, col=2
)

# Add statistical significance annotation to the bar plot
fig.add_annotation(
    x=0.5,  # Center of the bar plot subplot
    y=1.5,  # Position above the bars
    text=f"Z-statistic: {z_stat:.3f}<br>P-value: {p_value:.3f}",
    showarrow=False,
    font=dict(size=10),
    align="center",
    xref="x2",  # Reference the x-axis of the second subplot
    yref="y2",  # Reference the y-axis of the second subplot
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    row=1, col=2
)

# Update layout for the entire figure
fig.update_traces(
    marker=dict(size=8),
    hoverlabel=dict(
        font_size=12,
        namelength=-1  # Ensures the full headline is displayed without truncation
    ),
    row=1, col=1
)
fig.update_layout(
    width=1200,
    height=600,
    title_font_size=14,
    showlegend=True,
    legend_title_text="Clusters",
    legend=dict(
        font=dict(size=10),
        x=1.05,
        y=0.5,
        xanchor="left",
        yanchor="middle"
    ),
    margin=dict(r=300)
)
fig.update_xaxes(title_text="PCA Component 1", row=1, col=1)
fig.update_yaxes(title_text="PCA Component 2", row=1, col=1)
fig.update_xaxes(title_text="Proportion", row=1, col=2)
fig.update_yaxes(title_text="", row=1, col=2)

fig.write_html(os.path.join(figures_dir, "pca_interactive.html"))
logger.info(f"Interactive PCA plot with bar subplot saved as pca_interactive.html in {figures_dir}")

logger.info(f"Deep learning analysis complete! Results saved to {results_dir} and figures saved to {figures_dir}")
