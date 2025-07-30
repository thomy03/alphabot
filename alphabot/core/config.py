#!/usr/bin/env python3
"""
Configuration centralisée - AlphaBot Multi-Agent Trading System

Secure configuration management with environment variables,
validation, and production-ready security practices.
"""

import os
import logging
from functools import lru_cache
from typing import Optional, List, Literal
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr, validator


class Settings(BaseSettings):
    """Configuration sécurisée de l'application
    
    Toutes les clés sensibles sont gérées via des variables d'environnement
    et ne sont jamais loggées ou exposées en tant que chaînes simples.
    """
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[SecretStr] = Field(
        default=None, 
        env="REDIS_PASSWORD",
        description="Redis authentication password"
    )
    
    # API Keys (Secure)
    alpha_vantage_api_key: Optional[SecretStr] = Field(
        default=None, 
        env="ALPHA_VANTAGE_API_KEY",
        description="Alpha Vantage API key for market data"
    )
    finnhub_api_key: Optional[SecretStr] = Field(
        default=None, 
        env="FINNHUB_API_KEY",
        description="Finnhub API key for real-time data"
    )
    financial_modeling_prep_key: Optional[SecretStr] = Field(
        default=None, 
        env="FINANCIAL_MODELING_PREP_KEY",
        description="Financial Modeling Prep API key"
    )
    polygon_api_key: Optional[SecretStr] = Field(
        default=None,
        env="POLYGON_API_KEY", 
        description="Polygon.io API key for market data"
    )
    news_api_key: Optional[SecretStr] = Field(
        default=None,
        env="NEWS_API_KEY",
        description="News API key for sentiment analysis"
    )
    
    # Interactive Brokers
    ibkr_host: str = Field(default="localhost", env="IBKR_HOST")
    ibkr_port: int = Field(default=7497, env="IBKR_PORT")
    ibkr_client_id: int = Field(default=1, env="IBKR_CLIENT_ID")
    
    # Application Settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", 
        env="LOG_LEVEL",
        description="Logging level"
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development", 
        env="ENVIRONMENT",
        description="Application environment"
    )
    debug_mode: bool = Field(
        default=False,
        env="DEBUG_MODE", 
        description="Enable debug features (should be False in production)"
    )
    
    # Trading Settings
    max_position_size: float = Field(default=0.05, description="Max 5% par position")
    max_sector_exposure: float = Field(default=0.30, description="Max 30% par secteur")
    max_daily_var: float = Field(default=0.03, description="VaR 95% max 3%")
    
    # Signal Hub Settings
    signal_ttl_seconds: int = Field(default=300, description="TTL signaux par défaut")
    max_signal_history: int = Field(default=1000, description="Historique max signaux")
    
    @validator('environment')
    def validate_production_settings(cls, v, values):
        """Validate security settings for production environment"""
        if v == 'production':
            # Ensure debug mode is disabled in production
            if values.get('debug_mode', False):
                raise ValueError("Debug mode must be disabled in production")
            
            # Warn if using default Redis settings in production
            if values.get('redis_host') == 'localhost' and not values.get('redis_password'):
                logging.warning(
                    "Production environment detected with localhost Redis and no password. "
                    "Consider using secured Redis configuration."
                )
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v, values):
        """Validate log level based on environment"""
        env = values.get('environment', 'development')
        if env == 'production' and v == 'DEBUG':
            logging.warning(
                "DEBUG logging enabled in production. "
                "Consider using INFO or WARNING level for performance."
            )
        return v
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """Safely retrieve API key value"""
        key_field = getattr(self, key_name, None)
        if key_field and isinstance(key_field, SecretStr):
            return key_field.get_secret_value()
        return None
    
    def mask_sensitive_config(self) -> dict:
        """Return configuration with sensitive values masked for logging"""
        config_dict = self.dict()
        sensitive_fields = [
            'alpha_vantage_api_key', 'finnhub_api_key', 'financial_modeling_prep_key',
            'polygon_api_key', 'news_api_key', 'redis_password'
        ]
        
        for field in sensitive_fields:
            if field in config_dict and config_dict[field] is not None:
                config_dict[field] = "***MASKED***"
        
        return config_dict
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Don't allow extra fields to prevent configuration typos
        extra = "forbid"


@lru_cache()
def get_settings() -> Settings:
    """Obtenir les paramètres de configuration (singleton)"""
    return Settings()