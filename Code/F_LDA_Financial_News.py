# ========================================================
# LDA Topic Modeling on Cleaned Financial News Articles
# ========================================================

# Step 0: Install Required Libraries (run once)
# !pip install -q gensim pyLDAvis scikit-learn

# Step 1: Import Libraries
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models
from IPython.display import display


# Project Path Definition

PROJECT_PATH = "C:\Research\Finance_Research\Data"

INPUT_DIR = os.path.join(PROJECT_PATH, "preprocessed_data")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "topic_results/LDA")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load Cleaned Dataset

input_file = os.path.join(INPUT_DIR, "preprocessed_financial_news.csv")
df = pd.read_csv(input_file)


# Step 2: Filter Relevant Financial & Political Articles

df = df[df['cleaned_article'].str.contains(
    r"stock|market|fed|economy|finance|earnings|bank|politics|government|trump|biden|senate",
    case=False,
    regex=True
)]

# Reset index
df = df.reset_index(drop=True)


# Step 2a: Remove common reporting words

df['cleaned_article'] = df['cleaned_article'].str.replace(
    r'\b(said|reuters|according|news)\b', '', regex=True
)


# Step 3: Prepare Documents

documents = df["cleaned_article"].astype(str).tolist()


# Step 4: Tokenization & Vectorization

vectorizer = CountVectorizer(
    stop_words="english",
    min_df=5,
    max_df=0.9
)
X = vectorizer.fit_transform(documents)

# Convert to gensim corpus
corpus = [
    [word for word in doc.split() if word in vectorizer.vocabulary_]
    for doc in documents
]

# Create dictionary
dictionary = corpora.Dictionary(corpus)

# Create Bag-of-Words corpus
bow_corpus = [dictionary.doc2bow(text) for text in corpus]


# Step 5: Train LDA Model

num_topics = 10

lda_model = models.LdaModel(
    corpus=bow_corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42,
    update_every=1,
    passes=10,
    alpha="auto",
    per_word_topics=True
)


# Step 6: Display Topics

for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}\n")


# Step 7: Assign Dominant Topic to Each Document

def get_dominant_topic(bow):
    topic_probs = lda_model.get_document_topics(bow)
    topic_probs_sorted = sorted(topic_probs, key=lambda x: x[1], reverse=True)
    return topic_probs_sorted[0][0], topic_probs_sorted[0][1]

dominant_topics = [get_dominant_topic(bow) for bow in bow_corpus]
for i in range(num_topics):
    df[f"topic_{i}_prob"] = [dict(lda_model.get_document_topics(bow)).get(i, 0) for bow in bow_corpus]


# Step 8: Save LDA Results

lda_output_file = os.path.join(OUTPUT_DIR, "lda_topic_10_filtered.csv")
df.to_csv(lda_output_file, index=False)
print("✓ LDA topic modeling results saved at:", lda_output_file)


# Step 9: Topic Visualization (Colab-safe)

lda_vis = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dictionary)

# Option A: Display inside notebook (recommended)
pyLDAvis.enable_notebook()
display(lda_vis)



# summary file

# Create LDA Topic Summary with Top Keywords

num_top_words = 20  # number of top keywords per topic
topics_summary = []

for i in range(num_topics):
    # Get top words for the topic
    top_words = lda_model.show_topic(i, topn=num_top_words)
    keywords = [word for word, prob in top_words]
    topics_summary.append({
        "topic_id": i,
        "top_keywords": ", ".join(keywords)
    })

# Convert to DataFrame
topics_summary_df = pd.DataFrame(topics_summary)

# Save topic summary to CSV
topic_summary_file = os.path.join(OUTPUT_DIR, "lda_topic_10_summary.csv")
topics_summary_df.to_csv(topic_summary_file, index=False)
print("✓ LDA topic summary saved at:", topic_summary_file)

# Optional: Display the topic summary
display(topics_summary_df)
