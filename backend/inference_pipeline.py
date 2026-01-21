# Inference Pipeline Module
# Implements the full research pipeline for real-time prediction

import os
import re
import numpy as np
import pandas as pd
import torch
import joblib
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import PyPDF2
from io import BytesIO

class InferencePipeline:
    """
    End-to-end inference pipeline for S&P 500 index movement prediction
    """
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(self.project_root, "Data", "results", "xgboost_sp500_model.pkl")
        
        # Load XGBoost model
        self.model_loaded = False
        try:
            self.xgb_model = joblib.load(self.model_path)
            self.model_loaded = True
            print(f"✓ Loaded XGBoost model from {self.model_path}")
        except Exception as e:
            print(f"✗ Failed to load XGBoost model: {e}")
            self.xgb_model = None
        
        # Load FinBERT model
        self.finbert_model_name = "ProsusAI/finbert"
        try:
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(self.finbert_model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(self.finbert_model_name)
            self.finbert_model.eval()
            print(f"✓ Loaded FinBERT model")
        except Exception as e:
            print(f"✗ Failed to load FinBERT: {e}")
            self.finbert_tokenizer = None
            self.finbert_model = None
        
        # Feature names (must match training data order)
        self.feature_names = [
            'sentiment_score',
            'credit_ratings_fitch',
            'fed_rates_macro',
            'banks_markets_bonds_risk',
            'us_politics_geopolitics_trump',
            'italy_europe_data_treasury',
            'markets_fx_commodities',
            'china_trade_tariffs_energy',
            'corporate_business_activity',
            'election_tax_budget_us_politics',
            'sp500_earnings_us_equities',
            'tf_market',
            'tf_economy',
            'tf_bank',
            'tf_oil_energy',
            'tf_trade',
            'tf_stock',
            'tf_fed',
            'tf_rate',
            'tf_inflation',
            'tf_earnings',
            'tf_debt'
        ]
    
    def clean_text(self, text):
        """Clean and preprocess text (same as research pipeline)"""
        text = str(text)
        text = re.sub(r'\n|\r', ' ', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^\w\s\.]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        return text
    
    def extract_sentiment(self, text):
        """Extract FinBERT sentiment score"""
        if not self.finbert_model or not self.finbert_tokenizer:
            return 0.0
        
        try:
            # Truncate text if too long
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
            
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            # Sentiment score = positive - negative
            sentiment_score = float(probs[2] - probs[0])
            
            return sentiment_score
        except Exception as e:
            print(f"Error in sentiment extraction: {e}")
            return 0.0
    
    def extract_topic_features(self, text):
        """
        Extract LDA topic features (simplified for demo)
        In production, you would load the actual LDA model
        For now, return dummy values based on keyword matching
        """
        text_lower = text.lower()
        
        # Simplified topic detection based on keywords
        topics = {
            'credit_ratings_fitch': 1.0 if any(w in text_lower for w in ['credit', 'rating', 'fitch', 'moody']) else 0.0,
            'fed_rates_macro': 1.0 if any(w in text_lower for w in ['fed', 'federal reserve', 'interest rate', 'monetary']) else 0.0,
            'banks_markets_bonds_risk': 1.0 if any(w in text_lower for w in ['bank', 'bond', 'market', 'risk']) else 0.0,
            'us_politics_geopolitics_trump': 1.0 if any(w in text_lower for w in ['trump', 'politics', 'election', 'government']) else 0.0,
            'italy_europe_data_treasury': 1.0 if any(w in text_lower for w in ['europe', 'italy', 'treasury', 'eu']) else 0.0,
            'markets_fx_commodities': 1.0 if any(w in text_lower for w in ['commodity', 'forex', 'currency', 'gold', 'oil']) else 0.0,
            'china_trade_tariffs_energy': 1.0 if any(w in text_lower for w in ['china', 'trade', 'tariff', 'energy']) else 0.0,
            'corporate_business_activity': 1.0 if any(w in text_lower for w in ['corporate', 'business', 'company', 'earnings']) else 0.0,
            'election_tax_budget_us_politics': 1.0 if any(w in text_lower for w in ['election', 'tax', 'budget', 'fiscal']) else 0.0,
            'sp500_earnings_us_equities': 1.0 if any(w in text_lower for w in ['sp500', 's&p', 'earnings', 'stock', 'equity']) else 0.0,
        }
        
        return topics
    
    def extract_tfidf_features(self, text):
        """
        Extract TF-IDF keyword features (simplified)
        In production, you would use the actual TF-IDF vectorizer
        """
        text_lower = text.lower()
        
        keywords = {
            'tf_market': text_lower.count('market') / 100,
            'tf_economy': text_lower.count('economy') / 100,
            'tf_bank': text_lower.count('bank') / 100,
            'tf_oil_energy': (text_lower.count('oil') + text_lower.count('energy')) / 100,
            'tf_trade': text_lower.count('trade') / 100,
            'tf_stock': text_lower.count('stock') / 100,
            'tf_fed': text_lower.count('fed') / 100,
            'tf_rate': text_lower.count('rate') / 100,
            'tf_inflation': text_lower.count('inflation') / 100,
            'tf_earnings': text_lower.count('earnings') / 100,
            'tf_debt': text_lower.count('debt') / 100,
        }
        
        return keywords
    
    def predict(self, article_text, date=None):
        """
        Main prediction method
        
        Args:
            article_text: Raw article text
            date: Optional date string (not used in current model)
        
        Returns:
            dict with prediction, probability, and features
        """
        if not self.model_loaded:
            raise Exception("XGBoost model not loaded")
        
        # Step 1: Clean text
        cleaned_text = self.clean_text(article_text)
        
        # Step 2: Extract sentiment
        sentiment_score = self.extract_sentiment(cleaned_text)
        
        # Step 3: Extract topic features
        topic_features = self.extract_topic_features(cleaned_text)
        
        # Step 4: Extract TF-IDF features
        tfidf_features = self.extract_tfidf_features(cleaned_text)
        
        # Step 5: Combine all features
        features = {
            'sentiment_score': sentiment_score,
            **topic_features,
            **tfidf_features
        }
        
        # Create feature vector in correct order
        feature_vector = np.array([[features[name] for name in self.feature_names]])
        
        # Step 6: Make prediction
        prediction_proba = self.xgb_model.predict_proba(feature_vector)[0]
        prediction_class = int(self.xgb_model.predict(feature_vector)[0])
        
        # Prepare result
        result = {
            "prediction": "up" if prediction_class == 1 else "down",
            "probability": float(prediction_proba[prediction_class]),
            "confidence": float(max(prediction_proba)),
            "probabilities": {
                "down": float(prediction_proba[0]),
                "up": float(prediction_proba[1])
            },
            "features": features,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def extract_text_from_pdf(self, file_stream):
        """
        Extract text from PDF file
        
        Args:
            file_stream: File object or BytesIO stream
        
        Returns:
            Extracted text string
        """
        try:
            # Read PDF
            if hasattr(file_stream, 'read'):
                pdf_bytes = file_stream.read()
                pdf_file = BytesIO(pdf_bytes)
            else:
                pdf_file = file_stream
            
            reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
