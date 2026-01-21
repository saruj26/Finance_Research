# PDF Feature Extraction Guide - User Perspective

## **What Happens When User Uploads PDF?**

### **Step 1: User Uploads PDF File**
```
User selects PDF file in frontend
         ↓
Frontend sends file to backend API
         ↓
Backend receives PDF bytes
```

---

## **Step 2: Extract Text from PDF**

### **Code Location**: `inference_pipeline.py` → `extract_text_from_pdf()`

```python
def extract_text_from_pdf(self, file_stream):
    """
    USER UPLOADS PDF → THIS FUNCTION EXTRACTS ALL TEXT
    
    Example: User uploads "financial_news.pdf" with 3 pages
    Output: All text from all 3 pages combined into one string
    """
    try:
        # Convert file stream to bytes
        if hasattr(file_stream, 'read'):
            pdf_bytes = file_stream.read()  # Read uploaded file
            pdf_file = BytesIO(pdf_bytes)   # Convert to BytesIO object
        else:
            pdf_file = file_stream
        
        # Use PyPDF2 library to read PDF
        reader = PyPDF2.PdfReader(pdf_file)
        
        # Loop through ALL pages and extract text
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"  # Add newline between pages
        
        return text.strip()  # Return cleaned text
    
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")
```

### **Example**
```
Input PDF Content (3 pages):
┌─────────────────────────────────┐
│ Page 1: "Fed cuts interest rates │
│ and it impacts stock market..."  │
├─────────────────────────────────┤
│ Page 2: "S&P 500 rallies 2%..."  │
├─────────────────────────────────┤
│ Page 3: "Market sentiment shows  │
│ positive momentum..."             │
└─────────────────────────────────┘
         ↓
Output Text String:
"Fed cuts interest rates and it impacts stock market...
S&P 500 rallies 2%...
Market sentiment shows positive momentum..."
```

---

## **Step 3: Clean Extracted Text**

### **Code Location**: `inference_pipeline.py` → `clean_text()`

```python
def clean_text(self, text):
    """
    RAW PDF TEXT → CLEAN TEXT (removes noise)
    
    Example:
    Input: "Fed cuts rates.\nhttp://example.com\n@#$%^&*()"
    Output: "fed cuts rates"
    """
    text = str(text)
    
    # Remove line breaks and carriage returns
    text = re.sub(r'\n|\r', ' ', text)
    # "Line1\nLine2" → "Line1 Line2"
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # "Visit http://example.com" → "Visit"
    
    # Remove special characters (keep only letters, numbers, spaces, dots)
    text = re.sub(r'[^\w\s\.]', '', text)
    # "stock@market#!" → "stockmarket"
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # "word    word" → "word word"
    
    # Convert to lowercase
    text = text.lower()
    # "MARKET" → "market"
    
    return text
```

### **Example**
```
Raw PDF Text:
"The FEDERAL RESERVE announced rate cuts!!! (https://news.com)
S&P 500 is up 2.5%..."

After cleaning:
"the federal reserve announced rate cuts sp500 is up 25"
```

---

## **Step 4: Extract 27 Features from Cleaned Text**

### **Overview: 27 Features = 22 Base + 3 Market + 2 Interactions**

```
Cleaned Text Input
       ↓
┌──────────────────────────────────────────────────────┐
│  FEATURE EXTRACTION (4 Methods)                      │
├──────────────────────────────────────────────────────┤
│ 1. Sentiment Analysis (FinBERT)  → 1 feature         │
│ 2. Topic Analysis (LDA Keywords) → 10 features       │
│ 3. Keyword Frequency (TF-IDF)    → 11 features       │
│ 4. Market Data (Lagged + Interaction) → 5 features   │
└──────────────────────────────────────────────────────┘
       ↓
   27 Features Total
```

---

### **Feature 1: SENTIMENT SCORE (1 feature)**

#### **Code**: `extract_sentiment()`

```python
def extract_sentiment(self, text):
    """
    INPUT: Cleaned text from PDF
    OUTPUT: sentiment_score (-1.0 to +1.0)
    
    -1.0 = Very Negative (market will go DOWN)
     0.0 = Neutral
    +1.0 = Very Positive (market will go UP)
    
    MODEL USED: FinBERT (Financial BERT)
    """
    try:
        # Prepare text for FinBERT model
        inputs = self.finbert_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        # Note: FinBERT max length is 512 words
        # If PDF text is longer, it gets truncated
        
        # Get predictions from FinBERT
        with torch.no_grad():
            outputs = self.finbert_model(**inputs)
        
        # Convert to probabilities
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        # probs[0] = negative probability
        # probs[1] = neutral probability  
        # probs[2] = positive probability
        
        # Calculate sentiment score
        sentiment_score = float(probs[2] - probs[0])
        # positive_prob - negative_prob = sentiment_score
        
        return sentiment_score
    
    except Exception as e:
        print(f"Error in sentiment extraction: {e}")
        return 0.0  # Default to neutral if error
```

### **Example**
```
PDF Text: "Federal Reserve cuts rates. Stock market rallies. Positive outlook."

FinBERT Analysis:
├─ Negative words: "cuts" → low probability
├─ Positive words: "rallies", "positive" → high probability
└─ Result: sentiment_score = 0.82 (very positive)
```

---

### **Feature 2: TOPIC FEATURES (10 features)**

#### **Code**: `extract_topic_features()`

```python
def extract_topic_features(self, text):
    """
    INPUT: Cleaned text from PDF
    OUTPUT: 10 binary topic features (0 or 1)
    
    HOW IT WORKS: Check if specific keywords appear in text
    If keywords found → 1.0 (topic present)
    If keywords NOT found → 0.0 (topic absent)
    """
    text_lower = text.lower()
    
    topics = {
        'credit_ratings_risk': 1.0 if any(
            w in text_lower for w in ['credit', 'rating', 'fitch', 'moody', 'risk']
        ) else 0.0,
        # Example: If text contains "credit rating", this feature = 1.0
        
        'monetary_policy_inflation': 1.0 if any(
            w in text_lower for w in ['fed', 'federal reserve', 'interest rate', 'monetary', 'inflation']
        ) else 0.0,
        # Example: If text contains "federal reserve", this feature = 1.0
        
        'banking_financial_markets': 1.0 if any(
            w in text_lower for w in ['bank', 'banking', 'financial', 'market']
        ) else 0.0,
        
        'us_politics_geopolitics': 1.0 if any(
            w in text_lower for w in ['trump', 'politics', 'election', 'government', 'geopolitical']
        ) else 0.0,
        
        'economic_data_releases': 1.0 if any(
            w in text_lower for w in ['gdp', 'unemployment', 'inflation data', 'economic data', 'employment']
        ) else 0.0,
        
        'stock_market_performance': 1.0 if any(
            w in text_lower for w in ['stock', 'equity', 'market', 'rally', 'decline']
        ) else 0.0,
        
        'trade_war_oil': 1.0 if any(
            w in text_lower for w in ['trade', 'tariff', 'oil', 'energy', 'commodity']
        ) else 0.0,
        
        'corporate_business_activity': 1.0 if any(
            w in text_lower for w in ['corporate', 'business', 'company', 'earnings', 'revenue']
        ) else 0.0,
        
        'elections_fiscal_policy': 1.0 if any(
            w in text_lower for w in ['election', 'tax', 'budget', 'fiscal', 'stimulus']
        ) else 0.0,
        
        'index_earnings': 1.0 if any(
            w in text_lower for w in ['sp500', 's&p 500', 'earnings', 'index', 'dow']
        ) else 0.0,
    }
    
    return topics
```

### **Example**
```
PDF Text: "Fed announces interest rate cut. Stock market rallies on positive earnings."

Topic Analysis:
├─ Contains "fed" + "interest rate" → monetary_policy_inflation = 1.0
├─ Contains "stock market" → stock_market_performance = 1.0
├─ Contains "earnings" → index_earnings = 1.0
├─ Does NOT contain "credit/rating" → credit_ratings_risk = 0.0
└─ Does NOT contain "politics" → us_politics_geopolitics = 0.0

Result: [0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
```

---

### **Feature 3: TF-IDF FEATURES (11 features)**

#### **Code**: `extract_tfidf_features()`

```python
def extract_tfidf_features(self, text):
    """
    INPUT: Cleaned text from PDF
    OUTPUT: 11 keyword frequency features (0.0 to 1.0+)
    
    TF-IDF = Term Frequency / Inverse Document Frequency
    Measures: How frequently each keyword appears in text
    """
    text_lower = text.lower()
    
    keywords = {
        'tf_market': text_lower.count('market') / 100,
        # Count "market" occurrences ÷ 100
        # Example: "market" appears 5 times → feature = 0.05
        
        'tf_economy': text_lower.count('economy') / 100,
        # Count "economy" occurrences ÷ 100
        
        'tf_bank': text_lower.count('bank') / 100,
        
        'tf_oil_energy': (text_lower.count('oil') + text_lower.count('energy')) / 100,
        # Count BOTH "oil" and "energy" combined
        
        'tf_trade': text_lower.count('trade') / 100,
        
        'tf_stock': text_lower.count('stock') / 100,
        
        'tf_fed': text_lower.count('fed') / 100,
        
        'tf_rate': text_lower.count('rate') / 100,
        
        'tf_inflation': text_lower.count('inflation') / 100,
        
        'tf_earnings': text_lower.count('earnings') / 100,
        
        'tf_debt': text_lower.count('debt') / 100,
    }
    
    return keywords
```

### **Example**
```
PDF Text: "Market analysis shows strong earnings. 
The market continues rallying. Energy stocks lead the market."

TF-IDF Analysis:
├─ "market" appears 3 times → tf_market = 3 / 100 = 0.03
├─ "earnings" appears 1 time → tf_earnings = 1 / 100 = 0.01
├─ "energy" appears 1 time → tf_oil_energy = 1 / 100 = 0.01
├─ "economy" appears 0 times → tf_economy = 0.0
└─ ...other keywords...
```

---

### **Feature 4: MARKET FEATURES (5 features)**

#### **Code**: `extract_market_features()`

```python
def extract_market_features(self, sentiment_score):
    """
    INPUT: sentiment_score (extracted above)
    OUTPUT: 5 market-based features
    
    NOTE: These use HISTORICAL market data (previous trading day)
    NOT from PDF text - prevents data leakage!
    """
    # 3 Lagged Market Features (from previous day's market data)
    market_features = {
        'lagged_market_returns': self.latest_market_return,
        # Example: S&P 500 returned +1.5% yesterday → 0.015
        
        'lagged_volume_scaled': self.latest_volume_scaled,
        # Example: Yesterday's trading volume (log-scaled) → 20.5
        
        'lagged_price_momentum': self.latest_price_momentum,
        # Example: 5-day momentum from yesterday → 0.023
    }
    
    # 2 Interaction Features (sentiment × market)
    # These combine sentiment from PDF with market data
    interaction_features = {
        'sentiment_x_returns': sentiment_score * self.latest_market_return,
        # Example: 0.82 sentiment × 0.015 returns = 0.0123
        # Higher when sentiment AND returns are both positive
        
        'sentiment_x_momentum': sentiment_score * self.latest_price_momentum,
        # Example: 0.82 sentiment × 0.023 momentum = 0.0189
        # Captures interaction between text sentiment and price momentum
    }
    
    return {**market_features, **interaction_features}
```

### **Example**
```
Previous Trading Day Market Data:
├─ S&P 500 returned: +1.5% → lagged_market_returns = 0.015
├─ Trading volume: log(100M) = 18.4 → lagged_volume_scaled = 18.4
└─ 5-day momentum: +2.3% → lagged_price_momentum = 0.023

Current PDF Sentiment: 0.82 (positive)

Market Features Result:
├─ lagged_market_returns: 0.015
├─ lagged_volume_scaled: 18.4
├─ lagged_price_momentum: 0.023
├─ sentiment_x_returns: 0.82 × 0.015 = 0.0123
└─ sentiment_x_momentum: 0.82 × 0.023 = 0.0189
```

---

## **Step 5: Combine All 27 Features**

```python
# All features from different sources
features = {
    'sentiment_score': 0.82,                           # 1 feature
    'credit_ratings_risk': 1.0,                        # 10 topic features
    'monetary_policy_inflation': 1.0,
    'banking_financial_markets': 0.0,
    ... (7 more topic features)
    'tf_market': 0.03,                                 # 11 TF-IDF features
    'tf_earnings': 0.01,
    ... (9 more TF-IDF features)
    'lagged_market_returns': 0.015,                    # 5 market features
    'lagged_volume_scaled': 18.4,
    'lagged_price_momentum': 0.023,
    'sentiment_x_returns': 0.0123,
    'sentiment_x_momentum': 0.0189
}

# Total = 1 + 10 + 11 + 5 = 27 features
```

---

## **Step 6: Send to XGBoost Model for Prediction**

```python
# Create feature vector in exact order
feature_vector = np.array([
    [0.82, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,  # 11 features
     0.03, 0.01, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01, 0.02, 0.01, 0.0,  # 11 features
     0.015, 18.4, 0.023, 0.0123, 0.0189]  # 5 features
])

# XGBoost Model processes 27 features
prediction_proba = model.predict_proba(feature_vector)
# Returns: [probability_DOWN, probability_UP]
# Example: [0.25, 0.75] = 75% chance market goes UP
```

---

## **Complete User Journey Summary**

```
┌────────────────────────────────────────────────────┐
│ 1. USER UPLOADS PDF FILE                           │
│    (e.g., "financial_news_2026.pdf")              │
└────────────┬─────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────┐
│ 2. EXTRACT TEXT FROM PDF                           │
│    PyPDF2 library reads all pages                  │
│    Output: Raw text string                         │
└────────────┬─────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────┐
│ 3. CLEAN TEXT                                      │
│    Remove URLs, special chars, extra spaces        │
│    Output: "fed cuts rates stock market rallies"  │
└────────────┬─────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────┐
│ 4. EXTRACT 27 FEATURES                             │
│                                                    │
│ • sentiment_score (FinBERT) = 0.82                │
│ • 10 topic features (keyword matching) = [0,1,0...│
│ • 11 TF-IDF features (word count) = [0.03,0.01.. │
│ • 5 market features (historical data) = [0.015...│
└────────────┬─────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────┐
│ 5. SEND TO XGBOOST MODEL                           │
│    Input: 27 features                              │
│    Output: Prediction probability                  │
└────────────┬─────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────┐
│ 6. RETURN RESULT TO USER                           │
│    Prediction: "UP" or "DOWN"                      │
│    Confidence: 0.75 (75%)                          │
│    All 27 features displayed                       │
└────────────────────────────────────────────────────┘
```

---

## **Key Libraries Used**

| Library | Function | Code Location |
|---------|----------|---------------|
| **PyPDF2** | Extract text from PDF | `extract_text_from_pdf()` |
| **regex (re)** | Clean text | `clean_text()` |
| **FinBERT** | Sentiment analysis | `extract_sentiment()` |
| **KeywordMatching** | Topic detection | `extract_topic_features()` |
| **Counting** | TF-IDF features | `extract_tfidf_features()` |
| **pandas/numpy** | Market data processing | `extract_market_features()` |
| **XGBoost** | Final prediction | `predict()` in `xgboost_sp500_model.pkl` |

---

## **For Developers: How to Modify Features**

### **Add New Topic Feature**
```python
# In extract_topic_features()
'new_topic_name': 1.0 if any(
    w in text_lower for w in ['keyword1', 'keyword2', 'keyword3']
) else 0.0,
```

### **Add New TF-IDF Feature**
```python
# In extract_tfidf_features()
'tf_newkeyword': text_lower.count('newkeyword') / 100,
```

### **Modify Sentiment Extraction**
```python
# In extract_sentiment()
# Can change FinBERT to other sentiment models:
# - VADER (fast, simple)
# - RoBERTa (more accurate)
# - Custom trained model
```

---

**End of Feature Extraction Guide**
