#!/usr/bin/env python3
"""
Test basique des agents AlphaBot en mode dégradé
Vérifie que la structure de base fonctionne sans les dépendances externes
"""

import sys
import os
import traceback
from datetime import datetime

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.abspath('.'))

print("🧪 Test basique des agents AlphaBot")
print("=" * 50)

# Test 1: Import du template
try:
    from alphabot.agents.TEMPLATE_agent import AlphaBotAgentTemplate
    print("✅ Template Agent: Import OK")
except Exception as e:
    print(f"❌ Template Agent: {e}")
    traceback.print_exc()

# Test 2: Test Risk Agent (doit fonctionner car dépendances minimales)
try:
    from alphabot.agents.risk.risk_agent import RiskAgent
    print("✅ Risk Agent: Import OK")
    
    # Test d'initialisation
    agent = RiskAgent()
    print(f"✅ Risk Agent: Initialisation OK - {agent.agent_name}")
    
    # Test health check
    health = agent.health_check()
    print(f"✅ Risk Agent: Health check - {'OK' if health else 'FAIL'}")
    
except Exception as e:
    print(f"❌ Risk Agent: {e}")

# Test 3: Technical Agent (peut échouer sans numpy)
try:
    from alphabot.agents.technical.technical_agent import TechnicalAgent
    print("✅ Technical Agent: Import OK")
    
    agent = TechnicalAgent()
    print(f"✅ Technical Agent: Initialisation OK - {agent.agent_name}")
    
except ImportError as e:
    print(f"⚠️  Technical Agent: Dépendance manquante - {e}")
except Exception as e:
    print(f"❌ Technical Agent: {e}")

# Test 4: Sentiment Agent (peut échouer sans transformers)
try:
    from alphabot.agents.sentiment.sentiment_agent import SentimentAgent
    print("✅ Sentiment Agent: Import OK")
    
    agent = SentimentAgent()
    print(f"✅ Sentiment Agent: Initialisation OK - {agent.agent_name}")
    
    # Test en mode fallback
    result = agent._analyze_sentiment({
        'text': 'Company reports strong earnings growth',
        'ticker': 'TEST'
    })
    
    if result['status'] == 'success':
        print(f"✅ Sentiment Agent: Test fallback OK - {result['sentiment']}")
    else:
        print(f"❌ Sentiment Agent: Test failed - {result}")
    
except ImportError as e:
    print(f"⚠️  Sentiment Agent: Dépendance manquante - {e}")
except Exception as e:
    print(f"❌ Sentiment Agent: {e}")

# Test 5: Structure des dossiers
print("\n📁 Structure des agents:")
agents_dir = "alphabot/agents"
if os.path.exists(agents_dir):
    for item in os.listdir(agents_dir):
        item_path = os.path.join(agents_dir, item)
        if os.path.isdir(item_path):
            files = [f for f in os.listdir(item_path) if f.endswith('.py')]
            print(f"   📂 {item}/: {len(files)} fichiers Python")

# Test 6: Configuration files
print("\n📋 Fichiers de configuration:")
config_files = ['pyproject.toml', 'Makefile', 'risk_policy.yaml', '.gitignore']
for config_file in config_files:
    if os.path.exists(config_file):
        print(f"   ✅ {config_file}")
    else:
        print(f"   ❌ {config_file} (manquant)")

print("\n🏁 Test terminé!")
print(f"Timestamp: {datetime.now().isoformat()}")