# ========================================================
# Preprocessing Financial News Dataset for FinBERT & BERTopic
# ========================================================

# Step 1: Import Required Libraries
# ========================================================
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords


# Step 2: Dataset Path Configuration
# ========================================================
# Define main project directory
PROJECT_PATH = '/content/drive/MyDrive/SP500_Project'
INPUT_DATASET_PATH = os.path.join(PROJECT_PATH, 'Data/financial_News_final.csv')
OUTPUT_DIR_PATH = os.path.join(PROJECT_PATH, 'preprocessed_data')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

# Step 3: Load Original Dataset
# ========================================================
print("Loading dataset...")
df = pd.read_csv(INPUT_DATASET_PATH)
print(f"Original dataset shape: {df.shape}")
print("\nFirst 5 rows of original dataset:")
print(df.head())

# Step 4: Handle Missing Values
# ========================================================
print("\nHandling missing values...")

# Drop rows with missing article content
df = df.dropna(subset=['article'])
print(f"After dropping rows with missing articles: {df.shape}")

# Fill missing titles with empty string
df['title'] = df['title'].fillna('')

# Drop rows with missing dates
df = df.dropna(subset=['date'])
print(f"After dropping rows with missing dates: {df.shape}")

# Remove publication column if not needed for research
if 'publication' in df.columns:
    df = df.drop(columns=['publication'])
    print("Publication column removed")

# Step 5: Standardize Date Format
# ========================================================
print("\nStandardizing date format...")
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
print("Date format standardized to YYYY-MM-DD")

# Step 6: Text Cleaning Function
# ========================================================
def clean_text(text):
    """
    Clean text by removing special characters, URLs, and normalizing whitespace.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    # Convert to string if not already
    text = str(text)
    
    # Remove newline and carriage return characters
    text = re.sub(r'\n|\r', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters (keep words, spaces, and periods)
    text = re.sub(r'[^\w\s\.]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    return text

# Step 7: Clean Title and Article Columns
# ========================================================
print("\nCleaning text data...")
df['cleaned_title'] = df['title'].apply(clean_text)
df['cleaned_article'] = df['article'].apply(clean_text)
print("Title and article columns cleaned")

# Step 8: Stopwords Removal (Optional)
# ========================================================
print("\nSetting up stopwords removal...")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    """
    Remove English stopwords from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with stopwords removed
    """
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Apply stopwords removal to article column (optional for title)
df['cleaned_article'] = df['cleaned_article'].apply(remove_stopwords)
print("Stopwords removed from articles")

# Step 9: Create Combined Text Field
# ========================================================
print("\nCreating combined text field...")
df['cleaned_text'] = df['cleaned_title'] + '. ' + df['cleaned_article']
print("Combined text field created for FinBERT & BERTopic analysis")

# Step 10: Sort by Date
# ========================================================
print("\nSorting data by date...")
df = df.sort_values('date')
print(f"Data sorted by date. Date range: {df['date'].min()} to {df['date'].max()}")

# Step 11: Save Preprocessed Dataset
# ========================================================
print("\nSaving preprocessed datasets...")

# Save article-level preprocessed data
ARTICLE_LEVEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR_PATH, 'preprocessed_financial_news.csv')
df.to_csv(ARTICLE_LEVEL_OUTPUT_PATH, index=False)
print(f"Article-level preprocessed data saved to: {ARTICLE_LEVEL_OUTPUT_PATH}")

# Step 12: Create Daily Aggregated Dataset
# ========================================================
print("\nCreating daily aggregated dataset...")
daily_df = df.groupby('date')['cleaned_text'].apply(lambda x: ' '.join(x)).reset_index()

# Save daily aggregated data
DAILY_AGGREGATED_OUTPUT_PATH = os.path.join(OUTPUT_DIR_PATH, 'financial_news_daily.csv')
daily_df.to_csv(DAILY_AGGREGATED_OUTPUT_PATH, index=False)
print(f"Daily aggregated data saved to: {DAILY_AGGREGATED_OUTPUT_PATH}")

# Step 13: Summary Statistics
# ========================================================
print("\n" + "="*50)
print("PREPROCESSING COMPLETE - SUMMARY STATISTICS")
print("="*50)
print(f"Total articles processed: {len(df)}")
print(f"Total unique dates: {df['date'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Average article length (characters): {df['cleaned_text'].str.len().mean():.0f}")
print(f"Output files saved in: {OUTPUT_DIR_PATH}")
print("="*50)

# Display sample of processed data
print("\nSample of processed data (first 3 rows):")
print(df[['date', 'cleaned_title', 'cleaned_article']].head(3))