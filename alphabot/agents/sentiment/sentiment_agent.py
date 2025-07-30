"""
Sentiment Agent - Analyse de sentiment pour AlphaBot
Utilise FinBERT pour analyser le sentiment des news et données textuelles
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import re
from dataclasses import dataclass

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from alphabot.agents.TEMPLATE_agent import AlphaBotAgentTemplate


@dataclass
class SentimentScore:
    """Score de sentiment structuré"""
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float  # 0.0 to 1.0
    score: float  # -1.0 to 1.0 (-1 très négatif, +1 très positif)
    text_length: int
    timestamp: str


class SentimentAgent(AlphaBotAgentTemplate):
    """Agent d'analyse de sentiment utilisant FinBERT"""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(
            agent_name="SentimentAgent",
            description="Analyse de sentiment avec FinBERT",
            config_path=config_path
        )
        self.logger = logging.getLogger(__name__)
        
        # Configuration modèle
        self.model_name = "ProsusAI/finbert"
        self.max_length = 512
        self.batch_size = 8
        
        # Cache pour optimiser les requêtes
        self._sentiment_cache = {}
        self._pipeline = None
        
        # Mots-clés financiers pour pondération
        self.financial_keywords = {
            'positive': ['profit', 'growth', 'increase', 'strong', 'bullish', 'upgrade', 'beat', 'outperform'],
            'negative': ['loss', 'decline', 'weak', 'bearish', 'downgrade', 'miss', 'underperform', 'risk'],
            'neutral': ['stable', 'maintain', 'unchanged', 'neutral', 'hold']
        }
        
        # Seuils de classification
        self.sentiment_thresholds = {
            'strong_positive': 0.8,
            'positive': 0.6,
            'neutral': 0.4,
            'negative': -0.6,
            'strong_negative': -0.8
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialise le modèle FinBERT"""
        if not HAS_TRANSFORMERS:
            self.logger.warning("Transformers non installé. Utilisation du mode dégradé.")
            return
            
        try:
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1,  # CPU
                return_all_scores=True
            )
            self.logger.info(f"Modèle {self.model_name} chargé avec succès")
        except Exception as e:
            self.logger.error(f"Erreur chargement modèle: {e}")
            self._pipeline = None
    
    def _process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Traite les messages pour l'analyse de sentiment"""
        try:
            msg_type = message.get('type')
            
            if msg_type == 'analyze_sentiment':
                return self._analyze_sentiment(message)
            elif msg_type == 'batch_sentiment':
                return self._batch_sentiment_analysis(message)
            elif msg_type == 'news_sentiment':
                return self._analyze_news_sentiment(message)
            elif msg_type == 'aggregate_sentiment':
                return self._aggregate_sentiment_scores(message)
            else:
                return {
                    'status': 'error',
                    'message': f'Type de message non supporté: {msg_type}'
                }
                
        except Exception as e:
            self.logger.error(f"Erreur processing message: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _analyze_sentiment(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse le sentiment d'un texte"""
        text = message.get('text', '')
        ticker = message.get('ticker', 'UNKNOWN')
        
        if not text or not text.strip():
            return {
                'status': 'error',
                'message': 'Texte vide ou manquant'
            }
        
        # Préprocessing du texte
        clean_text = self._preprocess_text(text)
        
        # Calcul du sentiment
        sentiment_score = self._calculate_sentiment(clean_text)
        
        # Ajustement basé sur mots-clés financiers
        keyword_boost = self._calculate_keyword_boost(clean_text)
        adjusted_score = self._adjust_sentiment_score(sentiment_score, keyword_boost)
        
        return {
            'status': 'success',
            'ticker': ticker,
            'text_preview': text[:100] + "..." if len(text) > 100 else text,
            'sentiment': adjusted_score.sentiment,
            'confidence': adjusted_score.confidence,
            'score': adjusted_score.score,
            'raw_score': sentiment_score.score,
            'keyword_boost': keyword_boost,
            'text_length': len(text),
            'timestamp': datetime.now().isoformat()
        }
    
    def _batch_sentiment_analysis(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse le sentiment pour un batch de textes"""
        texts = message.get('texts', [])
        tickers = message.get('tickers', [])
        
        if not texts:
            return {
                'status': 'error',
                'message': 'Liste de textes vide'
            }
        
        # Assurer que tickers a la même longueur que texts
        if len(tickers) != len(texts):
            tickers = ['UNKNOWN'] * len(texts)
        
        results = []
        
        # Traitement par batches pour optimiser
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_tickers = tickers[i:i + self.batch_size]
            
            for text, ticker in zip(batch_texts, batch_tickers):
                try:
                    analysis = self._analyze_sentiment({
                        'text': text,
                        'ticker': ticker
                    })
                    if analysis['status'] == 'success':
                        results.append(analysis)
                except Exception as e:
                    self.logger.error(f"Erreur analyse sentiment pour {ticker}: {e}")
        
        # Statistiques du batch
        if results:
            scores = [r['score'] for r in results]
            avg_sentiment = np.mean(scores)
            sentiment_distribution = self._calculate_sentiment_distribution(scores)
        else:
            avg_sentiment = 0.0
            sentiment_distribution = {}
        
        return {
            'status': 'success',
            'total_analyzed': len(results),
            'results': results,
            'avg_sentiment_score': avg_sentiment,
            'sentiment_distribution': sentiment_distribution,
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_news_sentiment(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse le sentiment d'articles de news"""
        news_data = message.get('news_data', [])
        ticker = message.get('ticker', 'MARKET')
        time_decay_hours = message.get('time_decay_hours', 24)
        
        if not news_data:
            return {
                'status': 'error',
                'message': 'Aucune donnée news fournie'
            }
        
        sentiment_scores = []
        current_time = datetime.now()
        
        for news_item in news_data:
            try:
                title = news_item.get('title', '')
                summary = news_item.get('summary', '')
                published = news_item.get('published', current_time.isoformat())
                
                # Combiner titre et résumé
                full_text = f"{title}. {summary}".strip()
                
                if not full_text:
                    continue
                
                # Calculer le poids temporel (decay exponentiel)
                try:
                    pub_time = datetime.fromisoformat(published.replace('Z', '+00:00'))
                    time_diff_hours = (current_time - pub_time.replace(tzinfo=None)).total_seconds() / 3600
                    time_weight = np.exp(-time_diff_hours / time_decay_hours)
                except:
                    time_weight = 1.0
                
                # Analyse sentiment
                sentiment_result = self._analyze_sentiment({
                    'text': full_text,
                    'ticker': ticker
                })
                
                if sentiment_result['status'] == 'success':
                    weighted_score = sentiment_result['score'] * time_weight
                    sentiment_scores.append({
                        'title': title[:50] + "..." if len(title) > 50 else title,
                        'sentiment': sentiment_result['sentiment'],
                        'score': sentiment_result['score'],
                        'weighted_score': weighted_score,
                        'time_weight': time_weight,
                        'confidence': sentiment_result['confidence'],
                        'published': published
                    })
                    
            except Exception as e:
                self.logger.error(f"Erreur analyse news: {e}")
        
        # Agrégation des scores
        if sentiment_scores:
            weighted_scores = [item['weighted_score'] for item in sentiment_scores]
            avg_weighted_sentiment = np.mean(weighted_scores)
            
            # Classification finale
            final_sentiment = self._classify_sentiment_score(avg_weighted_sentiment)
        else:
            avg_weighted_sentiment = 0.0
            final_sentiment = 'neutral'
        
        return {
            'status': 'success',
            'ticker': ticker,
            'total_articles': len(sentiment_scores),
            'avg_weighted_sentiment': avg_weighted_sentiment,
            'final_sentiment': final_sentiment,
            'individual_scores': sentiment_scores,
            'time_decay_hours': time_decay_hours,
            'timestamp': datetime.now().isoformat()
        }
    
    def _aggregate_sentiment_scores(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Agrège plusieurs scores de sentiment avec pondération"""
        sentiment_data = message.get('sentiment_data', [])
        weights = message.get('weights', None)
        
        if not sentiment_data:
            return {
                'status': 'error',
                'message': 'Aucune donnée de sentiment fournie'
            }
        
        scores = []
        confidences = []
        
        for item in sentiment_data:
            if isinstance(item, dict) and 'score' in item:
                scores.append(item['score'])
                confidences.append(item.get('confidence', 1.0))
            elif isinstance(item, (int, float)):
                scores.append(item)
                confidences.append(1.0)
        
        if not scores:
            return {
                'status': 'error',
                'message': 'Aucun score valide trouvé'
            }
        
        # Pondération
        if weights is None:
            weights = confidences
        elif len(weights) != len(scores):
            weights = [1.0] * len(scores)
        
        # Calcul score agrégé
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        aggregated_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Statistiques
        stats = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': min(scores),
            'max': max(scores),
            'count': len(scores)
        }
        
        return {
            'status': 'success',
            'aggregated_score': aggregated_score,
            'aggregated_sentiment': self._classify_sentiment_score(aggregated_score),
            'total_weight': total_weight,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Préprocessing du texte pour l'analyse"""
        if not text:
            return ""
        
        # Nettoyage basique
        clean_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        clean_text = re.sub(r'@\w+', '', clean_text)  # Mentions Twitter
        clean_text = re.sub(r'#\w+', '', clean_text)  # Hashtags
        clean_text = re.sub(r'\s+', ' ', clean_text)  # Espaces multiples
        clean_text = clean_text.strip()
        
        # Truncate si trop long
        if len(clean_text) > self.max_length:
            clean_text = clean_text[:self.max_length]
        
        return clean_text
    
    def _calculate_sentiment(self, text: str) -> SentimentScore:
        """Calcule le score de sentiment avec FinBERT ou fallback"""
        if self._pipeline and HAS_TRANSFORMERS:
            return self._calculate_sentiment_finbert(text)
        else:
            return self._calculate_sentiment_fallback(text)
    
    def _calculate_sentiment_finbert(self, text: str) -> SentimentScore:
        """Calcule sentiment avec FinBERT"""
        try:
            results = self._pipeline(text)
            
            # FinBERT retourne positive, negative, neutral
            # results peut être une liste de listes si return_all_scores=True
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    results = results[0]  # Prendre le premier élément si c'est une liste de listes
            
            label_scores = {result['label'].lower(): result['score'] for result in results}
            
            positive_score = label_scores.get('positive', 0)
            negative_score = label_scores.get('negative', 0)
            neutral_score = label_scores.get('neutral', 0)
            
            # Conversion en score -1 à +1
            sentiment_score = positive_score - negative_score
            
            # Déterminer le sentiment principal
            max_label = max(label_scores, key=label_scores.get)
            confidence = max(label_scores.values())
            
            return SentimentScore(
                sentiment=max_label,
                confidence=confidence,
                score=sentiment_score,
                text_length=len(text),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Erreur FinBERT: {e}")
            return self._calculate_sentiment_fallback(text)
    
    def _calculate_sentiment_fallback(self, text: str) -> SentimentScore:
        """Calcule sentiment avec méthode fallback basée sur mots-clés"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.financial_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.financial_keywords['negative'] if word in text_lower)
        
        total_keywords = positive_count + negative_count
        
        if total_keywords == 0:
            sentiment = 'neutral'
            score = 0.0
            confidence = 0.5
        else:
            score = (positive_count - negative_count) / max(total_keywords, 1)
            
            if score > 0.2:
                sentiment = 'positive'
            elif score < -0.2:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            confidence = min(0.8, 0.5 + abs(score) * 0.3)
        
        return SentimentScore(
            sentiment=sentiment,
            confidence=confidence,
            score=score,
            text_length=len(text),
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_keyword_boost(self, text: str) -> float:
        """Calcule un boost basé sur la présence de mots-clés financiers"""
        text_lower = text.lower()
        boost = 0.0
        
        for category, keywords in self.financial_keywords.items():
            count = sum(1 for word in keywords if word in text_lower)
            if category == 'positive':
                boost += count * 0.1
            elif category == 'negative':
                boost -= count * 0.1
        
        return np.clip(boost, -0.3, 0.3)  # Limiter le boost
    
    def _adjust_sentiment_score(self, sentiment_score: SentimentScore, keyword_boost: float) -> SentimentScore:
        """Ajuste le score de sentiment avec le boost de mots-clés"""
        adjusted_score = np.clip(sentiment_score.score + keyword_boost, -1.0, 1.0)
        
        # Réclassifier si nécessaire
        new_sentiment = self._classify_sentiment_score(adjusted_score)
        
        return SentimentScore(
            sentiment=new_sentiment,
            confidence=sentiment_score.confidence,
            score=adjusted_score,
            text_length=sentiment_score.text_length,
            timestamp=sentiment_score.timestamp
        )
    
    def _classify_sentiment_score(self, score: float) -> str:
        """Classifie un score numérique en sentiment"""
        if score >= self.sentiment_thresholds['strong_positive']:
            return 'strong_positive'
        elif score >= self.sentiment_thresholds['positive']:
            return 'positive'
        elif score <= self.sentiment_thresholds['strong_negative']:
            return 'strong_negative'
        elif score <= self.sentiment_thresholds['negative']:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_sentiment_distribution(self, scores: List[float]) -> Dict[str, float]:
        """Calcule la distribution des sentiments"""
        if not scores:
            return {}
        
        classifications = [self._classify_sentiment_score(score) for score in scores]
        total = len(classifications)
        
        distribution = {}
        for sentiment in ['strong_negative', 'negative', 'neutral', 'positive', 'strong_positive']:
            count = classifications.count(sentiment)
            distribution[sentiment] = count / total if total > 0 else 0.0
        
        return distribution
    
    def health_check(self) -> bool:
        """Vérifie la santé de l'agent"""
        try:
            test_result = self._analyze_sentiment({
                'text': 'The company reported strong quarterly earnings.',
                'ticker': 'TEST'
            })
            return test_result['status'] == 'success'
        except Exception:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut de l'agent"""
        return {
            'agent_name': self.agent_name,
            'health': self.health_check(),
            'model_loaded': self._pipeline is not None,
            'model_name': self.model_name,
            'has_transformers': HAS_TRANSFORMERS,
            'cache_size': len(self._sentiment_cache),
            'thresholds': self.sentiment_thresholds
        }


if __name__ == "__main__":
    # Test rapide
    agent = SentimentAgent()
    print(f"Sentiment Agent initialized: {agent.get_status()}")
    
    # Test basique
    test_result = agent._analyze_sentiment({
        'text': 'The company reported strong quarterly earnings beating estimates.',
        'ticker': 'AAPL'
    })
    print(f"Test sentiment: {test_result}")