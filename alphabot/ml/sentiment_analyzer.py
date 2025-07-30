#!/usr/bin/env python3
"""
Sentiment DL Analyzer - AlphaBot Deep Learning Sentiment Analysis
Analyseur de sentiment utilisant FinBERT et modèles NLP avancés
pour capturer l'humeur du marché à partir de news et réseaux sociaux
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

# Imports NLP/DL (optionnels)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from sentence_transformers import SentenceTransformer
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Résultat d'analyse de sentiment"""
    sentiment_score: float  # -1 (très négatif) à 1 (très positif)
    confidence: float
    sentiment_label: str  # POSITIVE, NEGATIVE, NEUTRAL
    sources_analyzed: int
    key_phrases: List[str]
    reasoning: List[str]
    timeframe: str
    momentum: str  # IMPROVING, DECLINING, STABLE


class SentimentAnalyzer:
    """
    Analyseur de sentiment Deep Learning
    
    Combine:
    1. FinBERT pour analyse financière spécialisée
    2. Modèles transformers généraux
    3. Analyse classique VADER
    4. Feature engineering avec TF-IDF
    5. Ensemble ML pour robustesse
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "./alphabot/ml/sentiment_models/"
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        
        # Modèles NLP/DL
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.general_sentiment_pipeline = None
        self.sentence_transformer = None
        self.vader_analyzer = None
        self.ensemble_classifier = None
        self.tfidf_vectorizer = None
        
        # Configuration
        self.confidence_threshold = 0.7
        self.max_text_length = 512
        
        # Initialiser les modèles
        self._initialize_models()
        
        logger.info("🧠 Sentiment DL Analyzer initialized")
    
    def _initialize_models(self):
        """Initialiser ou charger les modèles NLP/DL"""
        try:
            if NLP_AVAILABLE:
                # Initialiser NLTK
                try:
                    nltk.data.find('sentiment/vader_lexicon.zip')
                except LookupError:
                    nltk.download('vader_lexicon', quiet=True)
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                
                # VADER Analyzer
                self.vader_analyzer = SentimentIntensityAnalyzer()
                
                # Charger ou créer FinBERT
                finbert_path = os.path.join(self.model_path, "finbert")
                if os.path.exists(finbert_path):
                    try:
                        self.finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_path)
                        self.finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_path)
                        logger.info("✅ FinBERT loaded")
                    except Exception as e:
                        logger.warning(f"⚠️ FinBERT loading failed: {e}")
                        self._load_finbert_from_hub()
                else:
                    self._load_finbert_from_hub()
                
                # Pipeline sentiment général
                self.general_sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Sentence Transformer pour embeddings
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Charger ou créer ensemble classifier
                ensemble_path = os.path.join(self.model_path, "sentiment_ensemble.pkl")
                tfidf_path = os.path.join(self.model_path, "sentiment_tfidf.pkl")
                
                if os.path.exists(ensemble_path) and os.path.exists(tfidf_path):
                    self.ensemble_classifier = joblib.load(ensemble_path)
                    self.tfidf_vectorizer = joblib.load(tfidf_path)
                    logger.info("✅ Sentiment ensemble loaded")
                else:
                    self.ensemble_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                    self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                    logger.info("🆕 Sentiment ensemble created")
            else:
                logger.warning("⚠️ NLP libraries not available")
                
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            self._create_fallback_models()
    
    def _load_finbert_from_hub(self):
        """Charger FinBERT depuis Hugging Face Hub"""
        try:
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
            logger.info("✅ FinBERT loaded from hub")
        except Exception as e:
            logger.error(f"❌ FinBERT loading from hub failed: {e}")
            self.finbert_tokenizer = None
            self.finbert_model = None
    
    def _create_fallback_models(self):
        """Créer des modèles fallback simples"""
        if NLP_AVAILABLE:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                self.ensemble_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
                self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
                logger.info("⚠️ Using fallback sentiment models")
            except Exception as e:
                logger.error(f"❌ Fallback models creation failed: {e}")
    
    async def analyze_sentiment_batch(self, symbols: List[str], news_data: Dict[str, List[str]]) -> Dict[str, SentimentResult]:
        """
        Analyser le sentiment pour une liste de symboles
        """
        results = {}
        
        for symbol in symbols:
            if symbol in news_data and news_data[symbol]:
                try:
                    sentiment_result = await self.analyze_sentiment(symbol, news_data[symbol])
                    results[symbol] = sentiment_result
                except Exception as e:
                    logger.error(f"Sentiment analysis failed for {symbol}: {e}")
                    # Créer un résultat fallback
                    results[symbol] = SentimentResult(
                        sentiment_score=0.0,
                        confidence=0.5,
                        sentiment_label="NEUTRAL",
                        sources_analyzed=0,
                        key_phrases=[],
                        reasoning=["Sentiment analysis unavailable"],
                        timeframe="recent",
                        momentum="STABLE"
                    )
        
        return results
    
    async def analyze_sentiment(self, symbol: str, texts: List[str]) -> SentimentResult:
        """
        Analyser le sentiment pour un symbole spécifique
        
        Combine:
        1. FinBERT (spécialisé finance)
        2. RoBERTa (général)
        3. VADER (classique)
        4. Ensemble ML
        """
        try:
            if not texts:
                return SentimentResult(
                    sentiment_score=0.0,
                    confidence=0.0,
                    sentiment_label="NEUTRAL",
                    sources_analyzed=0,
                    key_phrases=[],
                    reasoning=["No text data available"],
                    timeframe="recent",
                    momentum="STABLE"
                )
            
            # Nettoyer et prétraiter les textes
            cleaned_texts = [self._preprocess_text(text) for text in texts if text and len(text.strip()) > 10]
            
            if not cleaned_texts:
                return SentimentResult(
                    sentiment_score=0.0,
                    confidence=0.0,
                    sentiment_label="NEUTRAL",
                    sources_analyzed=0,
                    key_phrases=[],
                    reasoning=["No valid text after preprocessing"],
                    timeframe="recent",
                    momentum="STABLE"
                )
            
            # 1. Analyse FinBERT
            finbert_results = await self._analyze_with_finbert(cleaned_texts)
            
            # 2. Analyse RoBERTa
            roberta_results = await self._analyze_with_roberta(cleaned_texts)
            
            # 3. Analyse VADER
            vader_results = await self._analyze_with_vader(cleaned_texts)
            
            # 4. Analyse Ensemble ML
            ensemble_results = await self._analyze_with_ensemble(cleaned_texts)
            
            # 5. Extraire les phrases clés
            key_phrases = self._extract_key_phrases(cleaned_texts)
            
            # Fusionner tous les résultats
            final_result = self._fuse_sentiment_results(
                finbert_results, roberta_results, vader_results, ensemble_results,
                key_phrases, len(cleaned_texts)
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return SentimentResult(
                sentiment_score=0.0,
                confidence=0.0,
                sentiment_label="ERROR",
                sources_analyzed=0,
                key_phrases=[],
                reasoning=[str(e)],
                timeframe="recent",
                momentum="STABLE"
            )
    
    async def _analyze_with_finbert(self, texts: List[str]) -> Dict[str, Any]:
        """Analyse avec FinBERT"""
        if not self.finbert_model or not self.finbert_tokenizer:
            return {'score': 0.0, 'confidence': 0.0, 'label': 'NEUTRAL'}
        
        try:
            scores = []
            confidences = []
            
            for text in texts[:10]:  # Limiter pour performance
                # Tronquer le texte si nécessaire
                inputs = self.finbert_tokenizer(
                    text[:self.max_text_length], 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.finbert_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    score = predictions[0].tolist()
                
                # FinBERT labels: 0: Negative, 1: Neutral, 2: Positive
                max_idx = np.argmax(score)
                confidence = max(score)
                
                # Convertir en score -1 à 1
                if max_idx == 0:  # Negative
                    sentiment_score = -(confidence * 2 - 1)
                elif max_idx == 2:  # Positive
                    sentiment_score = confidence * 2 - 1
                else:  # Neutral
                    sentiment_score = 0.0
                
                scores.append(sentiment_score)
                confidences.append(confidence)
            
            avg_score = np.mean(scores) if scores else 0.0
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'score': avg_score,
                'confidence': avg_confidence,
                'label': self._score_to_label(avg_score)
            }
            
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'label': 'NEUTRAL'}
    
    async def _analyze_with_roberta(self, texts: List[str]) -> Dict[str, Any]:
        """Analyse avec RoBERTa"""
        if not self.general_sentiment_pipeline:
            return {'score': 0.0, 'confidence': 0.0, 'label': 'NEUTRAL'}
        
        try:
            results = self.general_sentiment_pipeline(texts[:10])  # Limiter pour performance
            
            scores = []
            confidences = []
            
            for result in results:
                label = result['label']
                score = result['score']
                
                # Convertir en score -1 à 1
                if label == 'LABEL_2':  # Positive
                    sentiment_score = score * 2 - 1
                elif label == 'LABEL_0':  # Negative
                    sentiment_score = -(score * 2 - 1)
                else:  # LABEL_1: Neutral
                    sentiment_score = 0.0
                
                scores.append(sentiment_score)
                confidences.append(score)
            
            avg_score = np.mean(scores) if scores else 0.0
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'score': avg_score,
                'confidence': avg_confidence,
                'label': self._score_to_label(avg_score)
            }
            
        except Exception as e:
            logger.error(f"RoBERTa analysis failed: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'label': 'NEUTRAL'}
    
    async def _analyze_with_vader(self, texts: List[str]) -> Dict[str, Any]:
        """Analyse avec VADER"""
        if not self.vader_analyzer:
            return {'score': 0.0, 'confidence': 0.0, 'label': 'NEUTRAL'}
        
        try:
            scores = []
            confidences = []
            
            for text in texts:
                vs = self.vader_analyzer.polarity_scores(text)
                compound_score = vs['compound']
                
                # Normaliser compound score (-1 à 1)
                sentiment_score = compound_score
                
                # Confiance basée sur l'intensité
                confidence = abs(compound_score)
                
                scores.append(sentiment_score)
                confidences.append(confidence)
            
            avg_score = np.mean(scores) if scores else 0.0
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'score': avg_score,
                'confidence': avg_confidence,
                'label': self._score_to_label(avg_score)
            }
            
        except Exception as e:
            logger.error(f"VADER analysis failed: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'label': 'NEUTRAL'}
    
    async def _analyze_with_ensemble(self, texts: List[str]) -> Dict[str, Any]:
        """Analyse avec ensemble ML"""
        if not self.ensemble_classifier or not self.tfidf_vectorizer:
            return {'score': 0.0, 'confidence': 0.0, 'label': 'NEUTRAL'}
        
        try:
            # Créer features TF-IDF
            tfidf_features = self.tfidf_vectorizer.transform(texts)
            
            # Faire des prédictions
            predictions = self.ensemble_classifier.predict_proba(tfidf_features)
            
            # Calculer le score moyen
            if predictions.shape[1] >= 3:  # 3 classes: neg, neut, pos
                scores = []
                for pred in predictions:
                    # Pondérer: pos - neg
                    score = pred[2] - pred[0] if pred.shape[0] >= 3 else 0.0
                    confidence = np.max(pred)
                    scores.append((score, confidence))
                
                avg_score = np.mean([s[0] for s in scores]) if scores else 0.0
                avg_confidence = np.mean([s[1] for s in scores]) if scores else 0.0
            else:
                avg_score = 0.0
                avg_confidence = 0.0
            
            return {
                'score': avg_score,
                'confidence': avg_confidence,
                'label': self._score_to_label(avg_score)
            }
            
        except Exception as e:
            logger.error(f"Ensemble analysis failed: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'label': 'NEUTRAL'}
    
    def _preprocess_text(self, text: str) -> str:
        """Prétraiter le texte pour analyse"""
        try:
            # Convertir en minuscules
            text = text.lower()
            
            # Supprimer les URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Supprimer les mentions @ et hashtags
            text = re.sub(r'\@\w+|\#\w+', '', text)
            
            # Supprimer les caractères spéciaux mais garder la ponctuation importante
            text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
            
            # Supprimer les espaces multiples
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            return text
    
    def _extract_key_phrases(self, texts: List[str]) -> List[str]:
        """Extraire les phrases clés des textes"""
        try:
            if not NLP_AVAILABLE:
                return []
            
            # Mots financiers importants à chercher
            financial_keywords = [
                'profit', 'loss', 'revenue', 'earnings', 'growth', 'decline',
                'increase', 'decrease', 'bullish', 'bearish', 'rally', 'crash',
                'upgrade', 'downgrade', 'buy', 'sell', 'hold', 'target',
                'forecast', 'guidance', 'outperform', 'underperform'
            ]
            
            key_phrases = []
            
            for text in texts[:5]:  # Limiter pour performance
                words = word_tokenize(text.lower())
                
                # Chercher des bigrammes/trigrammes avec mots financiers
                for i, word in enumerate(words):
                    if word in financial_keywords:
                        # Extraire la phrase contenant le mot clé
                        phrase_start = max(0, i-3)
                        phrase_end = min(len(words), i+4)
                        phrase = ' '.join(words[phrase_start:phrase_end])
                        key_phrases.append(phrase)
            
            # Dédupliquer et limiter
            unique_phrases = list(set(key_phrases))[:10]
            
            return unique_phrases
            
        except Exception as e:
            logger.error(f"Key phrase extraction failed: {e}")
            return []
    
    def _fuse_sentiment_results(self, finbert: Dict[str, Any], roberta: Dict[str, Any],
                               vader: Dict[str, Any], ensemble: Dict[str, Any],
                               key_phrases: List[str], sources_count: int) -> SentimentResult:
        """Fusionner tous les résultats d'analyse de sentiment"""
        
        # Pondérer les différents modèles
        weights = {
            'finbert': 0.35,    # Plus fiable pour la finance
            'roberta': 0.25,    # Général mais robuste
            'vader': 0.20,      # Rapide mais moins précis
            'ensemble': 0.20    # ML classique
        }
        
        # Calculer le score pondéré
        weighted_score = (
            weights['finbert'] * finbert['score'] +
            weights['roberta'] * roberta['score'] +
            weights['vader'] * vader['score'] +
            weights['ensemble'] * ensemble['score']
        )
        
        # Calculer la confiance pondérée
        weighted_confidence = (
            weights['finbert'] * finbert['confidence'] +
            weights['roberta'] * roberta['confidence'] +
            weights['vader'] * vader['confidence'] +
            weights['ensemble'] * ensemble['confidence']
        )
        
        # Déterminer le label
        sentiment_label = self._score_to_label(weighted_score)
        
        # Déterminer le momentum
        momentum = self._determine_momentum([finbert['score'], roberta['score'], vader['score']])
        
        # Générer le reasoning
        reasoning = []
        
        if abs(finbert['score']) > 0.3:
            reasoning.append(f"FinBERT: {finbert['label'].lower()} sentiment")
        
        if abs(roberta['score']) > 0.3:
            reasoning.append(f"RoBERTa: {roberta['label'].lower()} sentiment")
        
        if abs(vader['score']) > 0.3:
            reasoning.append(f"VADER: {vader['label'].lower()} sentiment")
        
        if weighted_confidence > 0.7:
            reasoning.append("High confidence in sentiment analysis")
        elif weighted_confidence < 0.4:
            reasoning.append("Low confidence - mixed signals")
        
        return SentimentResult(
            sentiment_score=weighted_score,
            confidence=weighted_confidence,
            sentiment_label=sentiment_label,
            sources_analyzed=sources_count,
            key_phrases=key_phrases,
            reasoning=reasoning,
            timeframe="recent",
            momentum=momentum
        )
    
    def _score_to_label(self, score: float) -> str:
        """Convertir un score en label"""
        if score > 0.2:
            return "POSITIVE"
        elif score < -0.2:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    def _determine_momentum(self, scores: List[float]) -> str:
        """Déterminer le momentum du sentiment"""
        if len(scores) < 2:
            return "STABLE"
        
        # Simple tendance basée sur la variance
        variance = np.var(scores)
        
        if variance > 0.1:
            return "VOLATILE"
        elif np.mean(scores) > 0.1:
            return "IMPROVING"
        elif np.mean(scores) < -0.1:
            return "DECLINING"
        else:
            return "STABLE"
    
    def save_models(self):
        """Sauvegarder les modèles entraînés"""
        try:
            if self.ensemble_classifier:
                joblib.dump(self.ensemble_classifier, os.path.join(self.model_path, "sentiment_ensemble.pkl"))
            
            if self.tfidf_vectorizer:
                joblib.dump(self.tfidf_vectorizer, os.path.join(self.model_path, "sentiment_tfidf.pkl"))
            
            # Sauvegarder FinBERT localement si chargé
            if self.finbert_model and self.finbert_tokenizer:
                finbert_path = os.path.join(self.model_path, "finbert")
                self.finbert_tokenizer.save_pretrained(finbert_path)
                self.finbert_model.save_pretrained(finbert_path)
            
            logger.info("💾 Sentiment models saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to save sentiment models: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtenir des informations sur les modèles"""
        return {
            'finbert_available': self.finbert_model is not None,
            'roberta_available': self.general_sentiment_pipeline is not None,
            'vader_available': self.vader_analyzer is not None,
            'ensemble_available': self.ensemble_classifier is not None,
            'nlp_available': NLP_AVAILABLE,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold
        }
