# ========================================================
# Topic Modeling of Financial News Articles using BERTopic
# ========================================================

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import os


# Project Path Definition
# --------------------------------------------------------

PROJECT_PATH = "/content/drive/MyDrive/SP500_Project"

INPUT_DIR = os.path.join(PROJECT_PATH, "preprocessed_data")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "topic_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------
# Load Cleaned Financial News Dataset
# --------------------------------------------------------
cleaned_file = os.path.join(INPUT_DIR, "preprocessed_financial_news.csv")
df = pd.read_csv(cleaned_file)

# --------------------------------------------------------
# Select Text for Topic Modeling
# --------------------------------------------------------
# Using cleaned_article column (stopwords removed, suitable for BERTopic)
texts = df["cleaned_article"].astype(str).tolist()


# Generate Sentence Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts, show_progress_bar=True)

# Initialize BERTopic Model
topic_model = BERTopic(
    language="english",
    calculate_probabilities=True
)

# --------------------------------------------------------
# Fit BERTopic Model
# --------------------------------------------------------
topics, probabilities = topic_model.fit_transform(texts, embeddings)

# --------------------------------------------------------
# Store Topic Results
# --------------------------------------------------------
df["topic"] = topics
df["probability"] = probabilities.max(axis=1)


# Save Topic Modeling Output
output_file = os.path.join(OUTPUT_DIR, "bertopic_output.csv" )

df.to_csv(output_file, index=False)

print("✓ BERTopic results saved to:", output_file)

# --------------------------------------------------------
# Display Discovered Topics
# --------------------------------------------------------
topic_info = topic_model.get_topic_info()
print(topic_info)
