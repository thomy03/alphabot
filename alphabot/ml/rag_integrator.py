#!/usr/bin/env python3
"""
RAG Integrator - AlphaBot Retrieval-Augmented Generation
Intégration RAG pour l'analyse contextuelle avancée
Combine recherche d'informations pertinentes et génération de insights
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

# Imports RAG/LLM (optionnels)
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import requests
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Résultat d'analyse RAG"""
    context_score: float
    confidence: float
    relevant_info: List[str]
    sources: List[str]
    reasoning: List[str]
    query_type: str
    timestamp: datetime


class RAGIntegrator:
    """
    Intégrateur RAG pour AlphaBot
    
    Capable de:
    1. Recherche sémantique dans les documents financiers
    2. Récupération d'informations en temps réel
    3. Analyse contextuelle des décisions
    4. Génération d'explications
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "./alphabot/ml/rag_models/"
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        
        # Modèles RAG
        self.sentence_model = None
        self.vector_index = None
        self.document_store = {}
        self.tfidf_vectorizer = None
        
        # Configuration
        self.top_k_results = 5
        self.confidence_threshold = 0.6
        
        # Initialiser les modèles
        self._initialize_models()
        
        logger.info("🔍 RAG Integrator initialized")
    
    def _initialize_models(self):
        """Initialiser les modèles RAG"""
        try:
            if RAG_AVAILABLE:
                # Sentence Transformer pour embeddings
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # TF-IDF Vectorizer
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                
                # Initialiser l'index FAISS
                self.vector_index = faiss.IndexFlatL2(384)  # Dimension du modèle
                
                # Charger ou créer la base de documents
                self._initialize_document_store()
                
                logger.info("✅ RAG models initialized")
            else:
                logger.warning("⚠️ RAG libraries not available")
                
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
            self._create_fallback_models()
    
    def _initialize_document_store(self):
        """Initialiser le stockage de documents"""
        try:
            # Documents financiers exemples
            sample_documents = {
                "market_trends": """
                Current market trends show increased volatility in technology stocks.
                Value stocks are outperforming growth stocks in the current environment.
                Interest rate expectations are driving market sentiment.
                """,
                "risk_factors": """
                Key risk factors include inflation concerns, geopolitical tensions,
                and potential regulatory changes in the technology sector.
                Supply chain disruptions continue to affect manufacturing sectors.
                """,
                "sector_analysis": """
                Technology sector shows strong earnings growth but high valuation.
                Healthcare sector provides defensive characteristics with steady growth.
                Financial sector benefits from rising interest rate environment.
                """,
                "economic_indicators": """
                GDP growth remains moderate but stable.
                Unemployment rates are at historical lows.
                Consumer spending shows resilience despite inflation pressures.
                """
            }
            
            # Ajouter les documents au store
            for doc_id, content in sample_documents.items():
                self.add_document(doc_id, content)
                
        except Exception as e:
            logger.error(f"Document store initialization failed: {e}")
    
    def _create_fallback_models(self):
        """Créer des modèles fallback simples"""
        if RAG_AVAILABLE:
            try:
                self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
                logger.info("⚠️ Using fallback RAG models")
            except Exception as e:
                logger.error(f"❌ Fallback RAG models failed: {e}")
    
    def add_document(self, doc_id: str, content: str):
        """Ajouter un document au store RAG"""
        try:
            # Stocker le document
            self.document_store[doc_id] = {
                'content': content,
                'timestamp': datetime.now(),
                'metadata': {}
            }
            
            # Créer l'embedding et ajouter à l'index
            if self.sentence_model and self.vector_index:
                embedding = self.sentence_model.encode([content])[0]
                self.vector_index.add(np.array([embedding]).astype('float32'))
                
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
    
    async def analyze_context_batch(self, symbols: List[str], queries: Dict[str, str]) -> Dict[str, RAGResult]:
        """
        Analyser le contexte pour une liste de symboles
        """
        results = {}
        
        for symbol in symbols:
            if symbol in queries:
                try:
                    rag_result = await self.analyze_context(symbol, queries[symbol])
                    results[symbol] = rag_result
                except Exception as e:
                    logger.error(f"RAG analysis failed for {symbol}: {e}")
                    # Créer un résultat fallback
                    results[symbol] = RAGResult(
                        context_score=0.5,
                        confidence=0.3,
                        relevant_info=["Context analysis unavailable"],
                        sources=[],
                        reasoning=["RAG analysis failed"],
                        query_type="error",
                        timestamp=datetime.now()
                    )
        
        return results
    
    async def analyze_context(self, symbol: str, query: str) -> RAGResult:
        """
        Analyser le contexte pour un symbole spécifique
        
        Combine:
        1. Recherche sémantique dans les documents
        2. Récupération d'informations externes
        3. Analyse de pertinence
        """
        try:
            # 1. Recherche sémantique
            semantic_results = await self._semantic_search(query)
            
            # 2. Recherche TF-IDF
            tfidf_results = await self._tfidf_search(query)
            
            # 3. Récupération d'informations externes (simulée)
            external_info = await self._fetch_external_info(symbol, query)
            
            # 4. Fusionner et analyser les résultats
            final_result = self._fuse_rag_results(
                semantic_results, tfidf_results, external_info, query
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"RAG analysis failed for {symbol}: {e}")
            return RAGResult(
                context_score=0.0,
                confidence=0.0,
                relevant_info=[],
                sources=[],
                reasoning=[str(e)],
                query_type="error",
                timestamp=datetime.now()
            )
    
    async def _semantic_search(self, query: str) -> Dict[str, Any]:
        """Recherche sémantique avec embeddings"""
        if not self.sentence_model or not self.vector_index:
            return {'results': [], 'scores': []}
        
        try:
            # Encoder la requête
            query_embedding = self.sentence_model.encode([query])[0]
            
            # Rechercher dans l'index
            k = min(self.top_k_results, len(self.document_store))
            distances, indices = self.vector_index.search(
                np.array([query_embedding]).astype('float32'), k
            )
            
            # Récupérer les documents pertinents
            results = []
            scores = []
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.document_store):
                    doc_id = list(self.document_store.keys())[idx]
                    doc_content = self.document_store[doc_id]['content']
                    
                    # Calculer le score de similarité
                    score = 1 / (1 + distance)  # Convertir distance en similarité
                    
                    results.append({
                        'doc_id': doc_id,
                        'content': doc_content,
                        'score': score
                    })
                    scores.append(score)
            
            return {'results': results, 'scores': scores}
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {'results': [], 'scores': []}
    
    async def _tfidf_search(self, query: str) -> Dict[str, Any]:
        """Recherche TF-IDF classique"""
        if not self.tfidf_vectorizer or not self.document_store:
            return {'results': [], 'scores': []}
        
        try:
            # Préparer les documents
            documents = [doc['content'] for doc in self.document_store.values()]
            doc_ids = list(self.document_store.keys())
            
            # Vectoriser les documents et la requête
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                # Première utilisation - entraîner le vectorizer
                self.tfidf_vectorizer.fit(documents)
            
            doc_vectors = self.tfidf_vectorizer.transform(documents)
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculer les similarités
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            # Récupérer les meilleurs résultats
            top_indices = np.argsort(similarities)[::-1][:self.top_k_results]
            
            results = []
            scores = []
            
            for idx in top_indices:
                if similarities[idx] > 0:
                    results.append({
                        'doc_id': doc_ids[idx],
                        'content': documents[idx],
                        'score': similarities[idx]
                    })
                    scores.append(similarities[idx])
            
            return {'results': results, 'scores': scores}
            
        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}")
            return {'results': [], 'scores': []}
    
    async def _fetch_external_info(self, symbol: str, query: str) -> Dict[str, Any]:
        """Récupérer des informations externes (simulé)"""
        try:
            # Pour l'instant, simuler des informations externes
            # Dans une version réelle, cela appellerait des APIs externes
            
            external_info = {
                'market_context': f"Current market conditions for {symbol} show moderate volatility",
                'recent_developments': f"Recent news for {symbol} includes earnings announcements",
                'analyst_sentiment': f"Analyst consensus for {symbol} is generally positive",
                'sector_trends': f"Sector trends affecting {symbol} include technological innovation"
            }
            
            return {
                'info': external_info,
                'confidence': 0.6,  # Confiance modérée pour les données simulées
                'sources': ['simulated_market_data', 'simulated_analyst_reports']
            }
            
        except Exception as e:
            logger.error(f"External info fetch failed: {e}")
            return {'info': {}, 'confidence': 0.0, 'sources': []}
    
    def _fuse_rag_results(self, semantic_results: Dict[str, Any], 
                         tfidf_results: Dict[str, Any], 
                         external_info: Dict[str, Any], 
                         query: str) -> RAGResult:
        """Fusionner tous les résultats RAG"""
        
        # Combiner les résultats de recherche
        all_results = semantic_results['results'] + tfidf_results['results']
        
        # Dédupliquer et scorer
        unique_results = {}
        for result in all_results:
            doc_id = result['doc_id']
            if doc_id not in unique_results or result['score'] > unique_results[doc_id]['score']:
                unique_results[doc_id] = result
        
        # Trier par score
        sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
        
        # Extraire les informations pertinentes
        relevant_info = []
        sources = []
        
        for result in sorted_results[:3]:  # Top 3 résultats
            relevant_info.append(result['content'][:200] + "...")  # Tronquer
            sources.append(result['doc_id'])
        
        # Ajouter les informations externes
        if external_info['info']:
            for key, value in external_info['info'].items():
                relevant_info.append(f"{key.replace('_', ' ').title()}: {value}")
            sources.extend(external_info['sources'])
        
        # Calculer le score de contexte
        if sorted_results:
            context_score = np.mean([r['score'] for r in sorted_results[:3]])
        else:
            context_score = 0.5
        
        # Calculer la confiance globale
        semantic_conf = np.mean(semantic_results['scores']) if semantic_results['scores'] else 0.0
        tfidf_conf = np.mean(tfidf_results['scores']) if tfidf_results['scores'] else 0.0
        external_conf = external_info['confidence']
        
        global_confidence = (semantic_conf * 0.4 + tfidf_conf * 0.3 + external_conf * 0.3)
        
        # Générer le reasoning
        reasoning = []
        
        if context_score > 0.7:
            reasoning.append("High relevance found in document search")
        elif context_score > 0.4:
            reasoning.append("Moderate relevance in document search")
        else:
            reasoning.append("Low relevance - limited contextual information")
        
        if global_confidence > 0.6:
            reasoning.append("Confident in contextual analysis")
        else:
            reasoning.append("Limited confidence in contextual analysis")
        
        if external_info['info']:
            reasoning.append("External market information incorporated")
        
        # Déterminer le type de requête
        query_type = self._classify_query(query)
        
        return RAGResult(
            context_score=context_score,
            confidence=global_confidence,
            relevant_info=relevant_info,
            sources=list(set(sources)),  # Dédupliquer
            reasoning=reasoning,
            query_type=query_type,
            timestamp=datetime.now()
        )
    
    def _classify_query(self, query: str) -> str:
        """Classifier le type de requête"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['risk', 'danger', 'threat', 'loss']):
            return "risk_analysis"
        elif any(word in query_lower for word in ['opportunity', 'growth', 'potential', 'buy']):
            return "opportunity_analysis"
        elif any(word in query_lower for word in ['trend', 'forecast', 'prediction', 'future']):
            return "trend_analysis"
        elif any(word in query_lower for word in ['news', 'recent', 'latest', 'current']):
            return "current_events"
        else:
            return "general_analysis"
    
    def get_relevant_context(self, symbol: str, context_type: str = "trading") -> Dict[str, Any]:
        """
        Obtenir du contexte pertinent pour un symbole
        Méthode utilitaire pour l'orchestrateur
        """
        try:
            # Construire une requête basée sur le type de contexte
            query_templates = {
                "trading": f"trading analysis and market conditions for {symbol}",
                "risk": f"risk factors and potential threats for {symbol}",
                "opportunity": f"investment opportunities and growth potential for {symbol}",
                "trend": f"market trends and price direction for {symbol}"
            }
            
            query = query_templates.get(context_type, f"general analysis for {symbol}")
            
            # Exécuter l'analyse RAG
            result = asyncio.run(self.analyze_context(symbol, query))
            
            return {
                'context_score': result.context_score,
                'confidence': result.confidence,
                'relevant_info': result.relevant_info,
                'sources': result.sources,
                'reasoning': result.reasoning
            }
            
        except Exception as e:
            logger.error(f"Context retrieval failed for {symbol}: {e}")
            return {
                'context_score': 0.0,
                'confidence': 0.0,
                'relevant_info': [],
                'sources': [],
                'reasoning': [str(e)]
            }
    
    def save_models(self):
        """Sauvegarder les modèles RAG"""
        try:
            # Sauvegarder le vectorizer TF-IDF
            if self.tfidf_vectorizer:
                import joblib
                joblib.dump(self.tfidf_vectorizer, os.path.join(self.model_path, "rag_tfidf.pkl"))
            
            # Sauvegarder l'index FAISS
            if self.vector_index:
                faiss.write_index(self.vector_index, os.path.join(self.model_path, "rag_faiss.index"))
            
            # Sauvegarder le document store
            if self.document_store:
                with open(os.path.join(self.model_path, "rag_documents.json"), 'w') as f:
                    # Convertir les datetime en string pour JSON
                    serializable_docs = {}
                    for doc_id, doc_data in self.document_store.items():
                        serializable_docs[doc_id] = {
                            'content': doc_data['content'],
                            'timestamp': doc_data['timestamp'].isoformat(),
                            'metadata': doc_data['metadata']
                        }
                    json.dump(serializable_docs, f, indent=2)
            
            logger.info("💾 RAG models saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to save RAG models: {e}")
    
    def load_models(self):
        """Charger les modèles RAG sauvegardés"""
        try:
            # Charger le vectorizer TF-IDF
            tfidf_path = os.path.join(self.model_path, "rag_tfidf.pkl")
            if os.path.exists(tfidf_path):
                import joblib
                self.tfidf_vectorizer = joblib.load(tfidf_path)
            
            # Charger l'index FAISS
            faiss_path = os.path.join(self.model_path, "rag_faiss.index")
            if os.path.exists(faiss_path):
                self.vector_index = faiss.read_index(faiss_path)
            
            # Charger le document store
            docs_path = os.path.join(self.model_path, "rag_documents.json")
            if os.path.exists(docs_path):
                with open(docs_path, 'r') as f:
                    serializable_docs = json.load(f)
                    # Convertir les strings en datetime
                    for doc_id, doc_data in serializable_docs.items():
                        doc_data['timestamp'] = datetime.fromisoformat(doc_data['timestamp'])
                    self.document_store = serializable_docs
                    
        except Exception as e:
            logger.error(f"❌ Failed to load RAG models: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtenir des informations sur les modèles RAG"""
        return {
            'sentence_model_available': self.sentence_model is not None,
            'vector_index_available': self.vector_index is not None,
            'tfidf_available': self.tfidf_vectorizer is not None,
            'rag_available': RAG_AVAILABLE,
            'document_count': len(self.document_store),
            'model_path': self.model_path
        }
