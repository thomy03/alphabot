#!/usr/bin/env python3
"""
AlphaBot ML Package - Machine Learning Components
Package regroupant tous les composants ML/DL pour AlphaBot
"""

from .pattern_detector import MLPatternDetector, PatternResult
from .sentiment_analyzer import SentimentAnalyzer, SentimentResult
from .rag_integrator import RAGAnalyzer, RAGResult

__version__ = "1.0.0"
__author__ = "AlphaBot Team"

# Export des classes principales
__all__ = [
    "MLPatternDetector",
    "PatternResult", 
    "SentimentAnalyzer",
    "SentimentResult",
    "RAGAnalyzer",
    "RAGResult"
]

# Information sur les composants disponibles
def get_ml_info():
    """Retourne les informations sur les composants ML disponibles"""
    return {
        "version": __version__,
        "components": {
            "pattern_detector": "ML Pattern Detector - Deep Learning pattern recognition",
            "sentiment_analyzer": "Sentiment DL Analyzer - Deep Learning sentiment analysis", 
            "rag_integrator": "RAG Integrator - Retrieval-Augmented Generation for context"
        },
        "features": [
            "LSTM pattern detection",
            "CNN volume analysis", 
            "Ensemble learning with Random Forest and Gradient Boosting",
            "FinBERT and RoBERTa sentiment analysis",
            "Semantic search with sentence transformers",
            "FAISS vector indexing",
            "TF-IDF document search"
        ]
    }
