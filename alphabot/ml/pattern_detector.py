#!/usr/bin/env python3
"""
ML Pattern Detector - AlphaBot Deep Learning Pattern Recognition
Détecteur de patterns utilisant des réseaux de neurones pour identifier
des configurations de marché non-linéaires et complexes
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pickle
import os
from pathlib import Path

# Imports ML (optionnels)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import joblib
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PatternResult:
    """Résultat de détection de pattern"""
    pattern_type: str
    confidence: float
    score: float
    description: str
    reasoning: List[str]
    timeframe: str
    expected_move: str  # UP, DOWN, SIDEWAYS


class MLPatternDetector:
    """
    Détecteur de patterns utilisant Deep Learning et Machine Learning
    
    Capable de détecter:
    - Patterns techniques classiques (tête-épaules, double top/bottom)
    - Patterns de prix non-linéaires via LSTM
    - Patterns de volume via CNN
    - Patterns multi-timeframes
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "./alphabot/ml/models/"
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        
        # Modèles ML/DL
        self.lstm_model = None
        self.cnn_model = None
        self.rf_classifier = None
        self.gb_classifier = None
        
        # Scalers
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = StandardScaler()
        
        # Patterns techniques classiques
        self.technical_patterns = {
            'head_and_shoulders': self._detect_head_and_shoulders,
            'double_top': self._detect_double_top,
            'double_bottom': self._detect_double_bottom,
            'triangle': self._detect_triangle,
            'wedge': self._detect_wedge,
            'flag': self._detect_flag
        }
        
        # Initialiser les modèles
        self._initialize_models()
        
        logger.info("🧠 ML Pattern Detector initialized")
    
    def _initialize_models(self):
        """Initialiser ou charger les modèles ML/DL"""
        try:
            # Charger les modèles existants ou en créer de nouveaux
            lstm_path = os.path.join(self.model_path, "lstm_pattern_model.h5")
            cnn_path = os.path.join(self.model_path, "cnn_pattern_model.h5")
            rf_path = os.path.join(self.model_path, "rf_classifier.pkl")
            gb_path = os.path.join(self.model_path, "gb_classifier.pkl")
            
            if TF_AVAILABLE:
                # Charger ou créer modèle LSTM
                if os.path.exists(lstm_path):
                    self.lstm_model = load_model(lstm_path)
                    logger.info("✅ LSTM model loaded")
                else:
                    self.lstm_model = self._create_lstm_model()
                    logger.info("🆕 LSTM model created")
                
                # Charger ou créer modèle CNN
                if os.path.exists(cnn_path):
                    self.cnn_model = load_model(cnn_path)
                    logger.info("✅ CNN model loaded")
                else:
                    self.cnn_model = self._create_cnn_model()
                    logger.info("🆕 CNN model created")
            
            # Charger ou créer Random Forest
            if os.path.exists(rf_path):
                self.rf_classifier = joblib.load(rf_path)
                logger.info("✅ Random Forest loaded")
            else:
                self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                logger.info("🆕 Random Forest created")
            
            # Charger ou créer Gradient Boosting
            if os.path.exists(gb_path):
                self.gb_classifier = joblib.load(gb_path)
                logger.info("✅ Gradient Boosting loaded")
            else:
                self.gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
                logger.info("🆕 Gradient Boosting created")
                
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            # Créer des modèles fallback simples
            self._create_fallback_models()
    
    def _create_lstm_model(self) -> Sequential:
        """Créer un modèle LSTM pour la détection de patterns temporels"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(30, 5)),  # 30 jours, 5 features
            Dropout(0.2),
            LSTM(25, return_sequences=False),
            Dropout(0.2),
            Dense(20, activation='relu'),
            Dense(3, activation='softmax')  # UP, DOWN, SIDEWAYS
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_cnn_model(self) -> Sequential:
        """Créer un modèle CNN pour la détection de patterns de volume"""
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=(20, 3)),  # 20 jours, 3 features
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(3, activation='softmax')  # UP, DOWN, SIDEWAYS
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_fallback_models(self):
        """Créer des modèles fallback simples si les modèles DL échouent"""
        if not TF_AVAILABLE:
            # Utiliser seulement les modèles sklearn
            self.rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            self.gb_classifier = GradientBoostingClassifier(n_estimators=50, random_state=42)
            logger.info("⚠️ Using fallback sklearn models only")
    
    async def detect_patterns_batch(self, price_data: Dict[str, pd.Series]) -> Dict[str, PatternResult]:
        """
        Détection de patterns par batch pour plusieurs symboles
        Méthode principale appelée par l'orchestrateur hybride
        """
        try:
            results = {}
            
            for symbol, prices in price_data.items():
                try:
                    # Détecter les patterns pour ce symbole
                    pattern_result = await self._detect_single_symbol_patterns(symbol, prices)
                    results[symbol] = pattern_result
                except Exception as e:
                    logger.error(f"Pattern detection failed for {symbol}: {e}")
                    # Créer un résultat par défaut
                    results[symbol] = PatternResult(
                        pattern_type="none",
                        confidence=0.0,
                        score=0.5,
                        description="Pattern detection failed",
                        reasoning=[f"Error: {str(e)}"],
                        timeframe="daily",
                        expected_move="SIDEWAYS"
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Batch pattern detection failed: {e}")
            return {}
    
    async def _detect_single_symbol_patterns(self, symbol: str, prices: pd.Series) -> PatternResult:
        """
        Détecter les patterns pour un seul symbole
        Combine analyse technique classique et ML/DL
        """
        try:
            # 1. Patterns techniques classiques
            technical_patterns = {}
            for pattern_name, detection_func in self.technical_patterns.items():
                try:
                    pattern_result = detection_func(prices)
                    if pattern_result['detected']:
                        technical_patterns[pattern_name] = pattern_result
                except Exception as e:
                    logger.warning(f"Technical pattern {pattern_name} failed for {symbol}: {e}")
            
            # 2. Prédictions Deep Learning (si disponibles)
            lstm_pred = {'prediction': 'SIDEWAYS', 'confidence': 0.5}
            cnn_pred = {'prediction': 'SIDEWAYS', 'confidence': 0.5}
            
            if TF_AVAILABLE and self.lstm_model and len(prices) >= 30:
                try:
                    lstm_pred = await self._predict_lstm_pattern(prices)
                except Exception as e:
                    logger.warning(f"LSTM prediction failed for {symbol}: {e}")
            
            if TF_AVAILABLE and self.cnn_model and len(prices) >= 20:
                try:
                    cnn_pred = await self._predict_cnn_pattern(prices)
                except Exception as e:
                    logger.warning(f"CNN prediction failed for {symbol}: {e}")
            
            # 3. Prédictions ensemble ML
            ensemble_pred = {'prediction': 'SIDEWAYS', 'confidence': 0.5}
            try:
                ensemble_pred = await self._predict_ensemble_pattern(symbol, prices, lstm_pred, cnn_pred)
            except Exception as e:
                logger.warning(f"Ensemble prediction failed for {symbol}: {e}")
            
            # 4. Fusionner tous les résultats
            final_result = self._fuse_pattern_results(technical_patterns, lstm_pred, cnn_pred, ensemble_pred)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Single symbol pattern detection failed for {symbol}: {e}")
            return PatternResult(
                pattern_type="error",
                confidence=0.0,
                score=0.5,
                description="Pattern detection error",
                reasoning=[f"Error: {str(e)}"],
                timeframe="daily",
                expected_move="SIDEWAYS"
            )
    
    async def _predict_lstm_pattern(self, prices: pd.Series) -> Dict[str, Any]:
        """Prédiction de pattern avec modèle LSTM"""
        try:
            # Préparer les données pour LSTM
            if len(prices) < 30:
                return {'prediction': 'SIDEWAYS', 'confidence': 0.5}
            
            # Créer les features
            recent_prices = prices.tail(30).values
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volume_features = np.ones(len(returns))  # Placeholder
            
            # Combiner features: prix, returns, volume (normalisé)
            features = np.column_stack([
                recent_prices[1:] / np.max(recent_prices),  # Prix normalisés
                returns,
                volume_features
            ])
            
            # Ajouter 2 features supplémentaires pour atteindre 5
            while features.shape[1] < 5:
                features = np.column_stack([features, np.zeros(len(features))])
            
            # Reshape pour LSTM: (samples, timesteps, features)
            features = features.reshape(1, features.shape[0], features.shape[1])
            
            # Faire la prédiction
            prediction = self.lstm_model.predict(features, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Mapper les classes
            class_map = {0: 'SIDEWAYS', 1: 'UP', 2: 'DOWN'}
            
            return {
                'prediction': class_map[predicted_class],
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return {'prediction': 'SIDEWAYS', 'confidence': 0.5}
    
    async def _predict_cnn_pattern(self, prices: pd.Series) -> Dict[str, Any]:
        """Prédiction de pattern avec modèle CNN"""
        try:
            # Préparer les données pour CNN
            if len(prices) < 20:
                return {'prediction': 'SIDEWAYS', 'confidence': 0.5}
            
            # Créer les features (prix, volume, volatilité)
            recent_prices = prices.tail(20).values
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = pd.Series(returns).rolling(5).std().fillna(0).values
            
            # Combiner features
            features = np.column_stack([
                recent_prices[1:] / np.max(recent_prices),  # Prix normalisés
                returns,
                volatility
            ])
            
            # Reshape pour CNN: (samples, timesteps, features)
            features = features.reshape(1, features.shape[0], features.shape[1])
            
            # Faire la prédiction
            prediction = self.cnn_model.predict(features, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Mapper les classes
            class_map = {0: 'SIDEWAYS', 1: 'UP', 2: 'DOWN'}
            
            return {
                'prediction': class_map[predicted_class],
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"CNN prediction failed: {e}")
            return {'prediction': 'SIDEWAYS', 'confidence': 0.5}
    
    async def _predict_ensemble_pattern(self, symbol: str, prices: pd.Series, 
                                     lstm_pred: Dict[str, Any], cnn_pred: Dict[str, Any]) -> Dict[str, Any]:
        """Prédiction avec ensemble ML (Random Forest + Gradient Boosting)"""
        try:
            # Préparer les features pour les modèles sklearn
            features = self._prepare_sklearn_features(symbol, prices, lstm_pred, cnn_pred)
            
            if features is None:
                return {'prediction': 'SIDEWAYS', 'confidence': 0.5}
            
            # Faire les prédictions
            rf_pred = self.rf_classifier.predict([features])[0]
            gb_pred = self.gb_classifier.predict([features])[0]
            
            # Obtenir les probabilités
            rf_proba = self.rf_classifier.predict_proba([features])[0]
            gb_proba = self.gb_classifier.predict_proba([features])[0]
            
            # Fusionner les prédictions
            # Mapper les classes numériques aux directions
            class_map = {0: 'SIDEWAYS', 1: 'UP', 2: 'DOWN'}
            
            # Voting pondéré
            rf_confidence = np.max(rf_proba)
            gb_confidence = np.max(gb_proba)
            
            if rf_pred == gb_pred:
                final_prediction = class_map[rf_pred]
                final_confidence = (rf_confidence + gb_confidence) / 2
            else:
                # En cas de désaccord, prendre celui avec la plus haute confiance
                if rf_confidence > gb_confidence:
                    final_prediction = class_map[rf_pred]
                    final_confidence = rf_confidence
                else:
                    final_prediction = class_map[gb_pred]
                    final_confidence = gb_confidence
            
            return {
                'prediction': final_prediction,
                'confidence': float(final_confidence)
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return {'prediction': 'SIDEWAYS', 'confidence': 0.5}
    
    def _prepare_sklearn_features(self, symbol: str, prices: pd.Series, 
                                lstm_pred: Dict[str, Any], cnn_pred: Dict[str, Any]) -> Optional[np.ndarray]:
        """Préparer les features pour les modèles sklearn"""
        try:
            # Features techniques
            recent_data = prices.to_frame(name='Close')
            recent_data['Returns'] = recent_data['Close'].pct_change()
            recent_data['RSI'] = self._calculate_rsi(recent_data['Close'])
            
            # Utiliser les 10 dernières valeurs
            recent_data = recent_data.tail(10).dropna()
            
            if len(recent_data) < 5:
                return None
            
            features = []
            
            # Features de base
            features.append(recent_data['Close'].iloc[-1])
            features.append(recent_data['Returns'].iloc[-1])
            features.append(recent_data['RSI'].iloc[-1])
            
            # Features des prédictions DL
            lstm_conf = lstm_pred.get('confidence', 0.5)
            cnn_conf = cnn_pred.get('confidence', 0.5)
            
            # Encoder les prédictions
            lstm_encoded = [1 if lstm_pred.get('prediction') == 'UP' else 0,
                           1 if lstm_pred.get('prediction') == 'DOWN' else 0,
                           1 if lstm_pred.get('prediction') == 'SIDEWAYS' else 0]
            
            cnn_encoded = [1 if cnn_pred.get('prediction') == 'UP' else 0,
                          1 if cnn_pred.get('prediction') == 'DOWN' else 0,
                          1 if cnn_pred.get('prediction') == 'SIDEWAYS' else 0]
            
            features.extend(lstm_encoded)
            features.extend(cnn_encoded)
            features.append(lstm_conf)
            features.append(cnn_conf)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Sklearn feature preparation failed: {e}")
            return None
    
    async def detect_patterns_batch(self, symbols: List[str], price_data: Dict[str, pd.DataFrame]) -> Dict[str, PatternResult]:
        """
        Détecter les patterns pour une liste de symboles
        """
        results = {}
        
        for symbol in symbols:
            if symbol in price_data:
                try:
                    pattern_result = await self.detect_patterns(symbol, price_data[symbol])
                    results[symbol] = pattern_result
                except Exception as e:
                    logger.error(f"Pattern detection failed for {symbol}: {e}")
                    # Créer un résultat fallback
                    results[symbol] = PatternResult(
                        pattern_type="unknown",
                        confidence=0.5,
                        score=0.5,
                        description="Pattern detection unavailable",
                        reasoning=["ML analysis failed"],
                        timeframe="daily",
                        expected_move="SIDEWAYS"
                    )
        
        return results
    
    async def detect_patterns(self, symbol: str, data: pd.DataFrame) -> PatternResult:
        """
        Détecter les patterns pour un symbole spécifique
        
        Combine:
        1. Patterns techniques classiques
        2. Prédictions LSTM (temporel)
        3. Prédictions CNN (volume)
        4. Ensemble Random Forest + Gradient Boosting
        """
        try:
            if len(data) < 50:
                return PatternResult(
                    pattern_type="insufficient_data",
                    confidence=0.0,
                    score=0.0,
                    description="Insufficient data for pattern detection",
                    reasoning=["Need at least 50 data points"],
                    timeframe="daily",
                    expected_move="SIDEWAYS"
                )
            
            # 1. Patterns techniques classiques
            technical_patterns = await self._detect_technical_patterns(data)
            
            # 2. Prédictions LSTM
            lstm_prediction = await self._predict_lstm(data)
            
            # 3. Prédictions CNN
            cnn_prediction = await self._predict_cnn(data)
            
            # 4. Ensemble ML
            ensemble_prediction = await self._ensemble_predict(data, lstm_prediction, cnn_prediction)
            
            # Fusionner tous les résultats
            final_result = self._fuse_pattern_results(
                technical_patterns, lstm_prediction, cnn_prediction, ensemble_prediction
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Pattern detection failed for {symbol}: {e}")
            return PatternResult(
                pattern_type="error",
                confidence=0.0,
                score=0.0,
                description="Pattern detection error",
                reasoning=[str(e)],
                timeframe="daily",
                expected_move="SIDEWAYS"
            )
    
    async def _detect_technical_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Détecter les patterns techniques classiques"""
        patterns_found = {}
        
        for pattern_name, detection_func in self.technical_patterns.items():
            try:
                result = detection_func(data)
                if result['detected']:
                    patterns_found[pattern_name] = result
            except Exception as e:
                logger.warning(f"Technical pattern {pattern_name} detection failed: {e}")
        
        return patterns_found
    
    def _detect_head_and_shoulders(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Détecter pattern tête-épaules"""
        # Implémentation simplifiée
        prices = data['Close'].values[-30:]  # Derniers 30 jours
        
        if len(prices) < 15:
            return {'detected': False}
        
        # Chercher 3 pics avec le pic central plus haut
        peaks = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
        
        if len(peaks) >= 3:
            # Vérifier si le pic central est le plus haut
            middle_idx = len(peaks) // 2
            if peaks[middle_idx][1] > peaks[middle_idx-1][1] and peaks[middle_idx][1] > peaks[middle_idx+1][1]:
                return {
                    'detected': True,
                    'confidence': 0.7,
                    'expected_move': 'DOWN',
                    'reasoning': 'Head and shoulders pattern detected - bearish reversal signal'
                }
        
        return {'detected': False}
    
    def _detect_double_top(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Détecter double top"""
        prices = data['Close'].values[-20:]
        
        if len(prices) < 10:
            return {'detected': False}
        
        # Chercher deux pics similaires
        peaks = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
        
        if len(peaks) >= 2:
            # Vérifier si les deux derniers pics sont similaires en hauteur
            price_diff = abs(peaks[-1][1] - peaks[-2][1]) / peaks[-2][1]
            if price_diff < 0.05:  # Moins de 5% de différence
                return {
                    'detected': True,
                    'confidence': 0.6,
                    'expected_move': 'DOWN',
                    'reasoning': 'Double top pattern detected - bearish reversal signal'
                }
        
        return {'detected': False}
    
    def _detect_double_bottom(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Détecter double bottom"""
        prices = data['Close'].values[-20:]
        
        if len(prices) < 10:
            return {'detected': False}
        
        # Chercher deux creux similaires
        troughs = []
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append((i, prices[i]))
        
        if len(troughs) >= 2:
            # Vérifier si les deux derniers creux sont similaires en hauteur
            price_diff = abs(troughs[-1][1] - troughs[-2][1]) / troughs[-2][1]
            if price_diff < 0.05:  # Moins de 5% de différence
                return {
                    'detected': True,
                    'confidence': 0.6,
                    'expected_move': 'UP',
                    'reasoning': 'Double bottom pattern detected - bullish reversal signal'
                }
        
        return {'detected': False}
    
    def _detect_triangle(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Détecter pattern triangle"""
        prices = data['Close'].values[-30:]
        
        if len(prices) < 15:
            return {'detected': False}
        
        # Détecter la convergence des hauts et des bas
        highs = []
        lows = []
        
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                highs.append((i, prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                lows.append((i, prices[i]))
        
        if len(highs) >= 3 and len(lows) >= 3:
            # Vérifier la convergence (les hauts baissent, les bas montent)
            high_trend = (highs[-1][1] - highs[0][1]) / highs[0][1]
            low_trend = (lows[-1][1] - lows[0][1]) / lows[0][1]
            
            if high_trend < -0.02 and low_trend > 0.02:  # Convergence significative
                return {
                    'detected': True,
                    'confidence': 0.65,
                    'expected_move': 'BREAKOUT',
                    'reasoning': 'Triangle pattern detected - consolidation phase'
                }
        
        return {'detected': False}
    
    def _detect_wedge(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Détecter pattern wedge"""
        # Implémentation similaire au triangle mais avec pente plus prononcée
        return self._detect_triangle(data)  # Simplifié pour l'instant
    
    def _detect_flag(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Détecter pattern flag"""
        prices = data['Close'].values[-15:]
        
        if len(prices) < 10:
            return {'detected': False}
        
        # Détecter une consolidation après un fort mouvement
        volatility = np.std(prices[-5:]) / np.mean(prices[-5:])
        if volatility < 0.02:  # Faible volatilité récente
            return {
                'detected': True,
                'confidence': 0.5,
                'expected_move': 'CONTINUATION',
                'reasoning': 'Flag pattern detected - continuation signal'
            }
        
        return {'detected': False}
    
    async def _predict_lstm(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Prédiction avec modèle LSTM"""
        if not TF_AVAILABLE or self.lstm_model is None:
            return {'prediction': 'SIDEWAYS', 'confidence': 0.5}
        
        try:
            # Préparer les données
            features = self._prepare_lstm_features(data)
            
            if features is None:
                return {'prediction': 'SIDEWAYS', 'confidence': 0.5}
            
            # Faire la prédiction
            prediction = self.lstm_model.predict(features, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            class_map = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
            
            return {
                'prediction': class_map[predicted_class],
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return {'prediction': 'SIDEWAYS', 'confidence': 0.5}
    
    async def _predict_cnn(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Prédiction avec modèle CNN"""
        if not TF_AVAILABLE or self.cnn_model is None:
            return {'prediction': 'SIDEWAYS', 'confidence': 0.5}
        
        try:
            # Préparer les données
            features = self._prepare_cnn_features(data)
            
            if features is None:
                return {'prediction': 'SIDEWAYS', 'confidence': 0.5}
            
            # Faire la prédiction
            prediction = self.cnn_model.predict(features, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            class_map = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
            
            return {
                'prediction': class_map[predicted_class],
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"CNN prediction failed: {e}")
            return {'prediction': 'SIDEWAYS', 'confidence': 0.5}
    
    async def _ensemble_predict(self, data: pd.DataFrame, lstm_pred: Dict[str, Any], cnn_pred: Dict[str, Any]) -> Dict[str, Any]:
        """Prédiction ensemble avec Random Forest et Gradient Boosting"""
        try:
            # Préparer les features pour les modèles sklearn
            features = self._prepare_sklearn_features(data, lstm_pred, cnn_pred)
            
            if features is None:
                return {'prediction': 'SIDEWAYS', 'confidence': 0.5}
            
            # Faire les prédictions
            rf_pred = self.rf_classifier.predict_proba([features])[0]
            gb_pred = self.gb_classifier.predict_proba([features])[0]
            
            # Combiner les prédictions
            ensemble_pred = (rf_pred + gb_pred) / 2
            predicted_class = np.argmax(ensemble_pred)
            confidence = np.max(ensemble_pred)
            
            class_map = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
            
            return {
                'prediction': class_map[predicted_class],
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return {'prediction': 'SIDEWAYS', 'confidence': 0.5}
    
    def _prepare_lstm_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Préparer les features pour le LSTM"""
        try:
            # Utiliser les derniers 30 jours
            recent_data = data.tail(30)
            
            if len(recent_data) < 30:
                return None
            
            # Créer features: prix, volume, returns, volatilité
            features = pd.DataFrame()
            features['price'] = recent_data['Close']
            features['volume'] = recent_data['Volume']
            features['returns'] = recent_data['Close'].pct_change()
            features['volatility'] = features['returns'].rolling(5).std()
            features['rsi'] = self._calculate_rsi(recent_data['Close'])
            
            # Normaliser
            features = features.fillna(0)
            features_scaled = self.price_scaler.fit_transform(features)
            
            return features_scaled.reshape(1, 30, 5)
            
        except Exception as e:
            logger.error(f"LSTM feature preparation failed: {e}")
            return None
    
    def _prepare_cnn_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Préparer les features pour le CNN"""
        try:
            # Utiliser les derniers 20 jours
            recent_data = data.tail(20)
            
            if len(recent_data) < 20:
                return None
            
            # Créer features volume-based
            features = pd.DataFrame()
            features['volume'] = recent_data['Volume']
            features['volume_sma'] = features['volume'].rolling(5).mean()
            features['volume_ratio'] = features['volume'] / features['volume_sma']
            features['price_change'] = recent_data['Close'].pct_change()
            
            # Normaliser
            features = features.fillna(0)
            features_scaled = self.volume_scaler.fit_transform(features)
            
            return features_scaled.reshape(1, 20, 3)
            
        except Exception as e:
            logger.error(f"CNN feature preparation failed: {e}")
            return None
    
    def _prepare_sklearn_features(self, data: pd.DataFrame, lstm_pred: Dict[str, Any], cnn_pred: Dict[str, Any]) -> Optional[np.ndarray]:
        """Préparer les features pour les modèles sklearn"""
        try:
            # Features techniques
            recent_data = data.tail(10)
            
            features = []
            
            # Features de base
            features.append(recent_data['Close'].iloc[-1])
            features.append(recent_data['Close'].pct_change(5).iloc[-1])
            features.append(recent_data['Close'].pct_change(10).iloc[-1])
            features.append(self._calculate_rsi(recent_data['Close']).iloc[-1])
            features.append(recent_data['Volume'].pct_change().iloc[-1])
            
            # Features des prédictions DL
            lstm_conf = lstm_pred.get('confidence', 0.5)
            cnn_conf = cnn_pred.get('confidence', 0.5)
            
            # Encoder les prédictions
            lstm_encoded = [1 if lstm_pred.get('prediction') == 'UP' else 0,
                           1 if lstm_pred.get('prediction') == 'DOWN' else 0,
                           1 if lstm_pred.get('prediction') == 'SIDEWAYS' else 0]
            
            cnn_encoded = [1 if cnn_pred.get('prediction') == 'UP' else 0,
                          1 if cnn_pred.get('prediction') == 'DOWN' else 0,
                          1 if cnn_pred.get('prediction') == 'SIDEWAYS' else 0]
            
            features.extend(lstm_encoded)
            features.extend(cnn_encoded)
            features.append(lstm_conf)
            features.append(cnn_conf)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Sklearn feature preparation failed: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculer le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _fuse_pattern_results(self, technical_patterns: Dict[str, Any], 
                             lstm_pred: Dict[str, Any], cnn_pred: Dict[str, Any],
                             ensemble_pred: Dict[str, Any]) -> PatternResult:
        """Fusionner tous les résultats de détection de patterns"""
        
        # Compter les votes
        votes = {'UP': 0, 'DOWN': 0, 'SIDEWAYS': 0}
        confidences = []
        reasoning = []
        
        # Votes des patterns techniques
        for pattern_name, result in technical_patterns.items():
            if result['detected']:
                move = result['expected_move']
                votes[move] += result['confidence']
                confidences.append(result['confidence'])
                reasoning.append(f"{pattern_name}: {result['reasoning']}")
        
        # Votes des modèles DL
        for pred in [lstm_pred, cnn_pred, ensemble_pred]:
            move = pred['prediction']
            votes[move] += pred['confidence']
            confidences.append(pred['confidence'])
        
        # Déterminer le résultat final
        final_move = max(votes, key=votes.get)
        total_confidence = votes[final_move]
        avg_confidence = np.mean(confidences) if confidences else 0.5
        
        # Score normalisé
        score = min(1.0, total_confidence / len([lstm_pred, cnn_pred, ensemble_pred]))
        
        # Description
        if technical_patterns:
            pattern_names = list(technical_patterns.keys())
            description = f"Detected patterns: {', '.join(pattern_names)}"
        else:
            description = "ML-based pattern detection"
        
        return PatternResult(
            pattern_type="hybrid",
            confidence=avg_confidence,
            score=score,
            description=description,
            reasoning=reasoning,
            timeframe="daily",
            expected_move=final_move
        )
    
    def save_models(self):
        """Sauvegarder les modèles entraînés"""
        try:
            if TF_AVAILABLE:
                if self.lstm_model:
                    self.lstm_model.save(os.path.join(self.model_path, "lstm_pattern_model.h5"))
                if self.cnn_model:
                    self.cnn_model.save(os.path.join(self.model_path, "cnn_pattern_model.h5"))
            
            if self.rf_classifier:
                joblib.dump(self.rf_classifier, os.path.join(self.model_path, "rf_classifier.pkl"))
            
            if self.gb_classifier:
                joblib.dump(self.gb_classifier, os.path.join(self.model_path, "gb_classifier.pkl"))
            
            logger.info("💾 ML models saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to save models: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtenir des informations sur les modèles"""
        return {
            'lstm_available': self.lstm_model is not None,
            'cnn_available': self.cnn_model is not None,
            'rf_available': self.rf_classifier is not None,
            'gb_available': self.gb_classifier is not None,
            'tensorflow_available': TF_AVAILABLE,
            'model_path': self.model_path,
            'technical_patterns': list(self.technical_patterns.keys())
        }
