#!/usr/bin/env python3
"""
Interactive Brokers Setup Guide for Elite Paper Trading System
Guide de configuration IBKR pour déploiement paper trading
"""

import os
import sys
from pathlib import Path

def print_setup_guide():
    """Guide de configuration Interactive Brokers"""
    
    print("🎯 INTERACTIVE BROKERS PAPER TRADING SETUP")
    print("=" * 60)
    
    print("\n📋 REQUIREMENTS CHECKLIST:")
    print("□ Interactive Brokers account (paper trading enabled)")
    print("□ TWS (Trader Workstation) or IB Gateway installed")
    print("□ API access enabled in TWS/Gateway")
    print("□ Python packages: ib_insync, yfinance, pandas, numpy")
    
    print("\n🔧 STEP 1: Install Required Packages")
    print("Run these commands:")
    print("pip install ib_insync yfinance pandas numpy")
    print("pip install langgraph langchain transformers")
    print("pip install faiss-cpu rank-bm25 feedparser")
    print("pip install ta-lib cvxpy matplotlib plotly")
    
    print("\n🏦 STEP 2: Interactive Brokers Account Setup")
    print("1. Create IB account: https://www.interactivebrokers.com")
    print("2. Enable paper trading in Client Portal")
    print("3. Download TWS or IB Gateway")
    print("4. Enable API access in TWS: API → Settings → Enable ActiveX and Socket Clients")
    
    print("\n⚙️ STEP 3: TWS Configuration")
    print("Paper Trading Configuration:")
    print("• Host: 127.0.0.1 (localhost)")
    print("• Port: 7497 (paper trading)")  
    print("• Port: 7496 (live trading - DO NOT USE)")
    print("• Socket port: same as above")
    print("• Master API client ID: 0")
    print("• Read-Only API: NO (we need to place orders)")
    
    print("\n🔒 STEP 4: Security Settings")
    print("• Trusted IP addresses: 127.0.0.1")
    print("• API authentication: not required for localhost")
    print("• Use paper account ONLY for testing")
    
    print("\n🚀 STEP 5: Test Connection")
    print("Run the test script below to verify connection:")
    
    test_script = '''
#!/usr/bin/env python3
"""Test IBKR connection"""
try:
    from ib_insync import IB, util
    util.startLoop()
    
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)
    
    if ib.isConnected():
        print("✅ IBKR Connection successful!")
        
        # Get account info
        accounts = ib.managedAccounts()
        print(f"📊 Accounts: {accounts}")
        
        # Get portfolio
        portfolio = ib.portfolio()
        print(f"💼 Portfolio positions: {len(portfolio)}")
        
        ib.disconnect()
        print("✅ Test completed successfully")
    else:
        print("❌ Connection failed")
        
except ImportError:
    print("❌ ib_insync not installed: pip install ib_insync")
except Exception as e:
    print(f"❌ Connection error: {e}")
    print("Check TWS/Gateway is running and API is enabled")
'''
    
    # Save test script
    test_file = Path("test_ibkr_connection.py")
    test_file.write_text(test_script)
    print(f"💾 Test script saved as: {test_file.absolute()}")
    
    print("\n📊 STEP 6: Run Elite Paper Trading System")
    print("python elite_superintelligence_paper_trading.py")
    
    print("\n⚠️ IMPORTANT SAFETY NOTES:")
    print("• ALWAYS use paper trading (port 7497) for testing")
    print("• Never use live trading (port 7496) until fully tested")
    print("• Start with small position sizes")
    print("• Monitor trades carefully")
    print("• Keep TWS/Gateway running during trading")
    
    print("\n🐛 TROUBLESHOOTING:")
    print("Connection Failed:")
    print("• Check TWS/Gateway is running")
    print("• Verify port 7497 for paper trading")  
    print("• Enable API in TWS settings")
    print("• Check firewall/antivirus")
    
    print("\nAPI Errors:")
    print("• Increase client ID if 'already connected'")
    print("• Restart TWS/Gateway if persistent errors")
    print("• Check account permissions")
    
    print("\nOrder Errors:")
    print("• Verify paper trading account has sufficient funds")
    print("• Check market hours for the instrument")
    print("• Ensure contract is properly qualified")
    
    print("\n📞 SUPPORT:")
    print("• IB API Documentation: https://interactivebrokers.github.io/tws-api/")
    print("• ib_insync Documentation: https://ib-insync.readthedocs.io/")
    print("• IB Client Portal: https://www.interactivebrokers.com/portal")

def create_config_file():
    """Créer fichier de configuration"""
    
    config = {
        "ibkr": {
            "host": "127.0.0.1",
            "paper_port": 7497,
            "live_port": 7496,
            "client_id": 1,
            "timeout": 10
        },
        "trading": {
            "initial_capital": 100000,
            "max_positions": 20,
            "max_leverage": 1.4,
            "target_return": 0.40,
            "risk_free_rate": 0.02
        },
        "data": {
            "max_news_articles": 5,
            "news_cache_hours": 24,
            "bm25_corpus_limit": 5000,
            "data_cache_days": 7
        },
        "paths": {
            "data_dir": "./paper_trading_states/",
            "trades_dir": "./paper_trading_states/trades/",
            "logs_dir": "./paper_trading_states/logs/",
            "models_dir": "./paper_trading_states/models/"
        }
    }
    
    import json
    config_file = Path("ibkr_config.json")
    config_file.write_text(json.dumps(config, indent=2))
    print(f"💾 Configuration saved as: {config_file.absolute()}")
    
    return config

def main():
    """Main setup function"""
    print_setup_guide()
    print("\n" + "=" * 60)
    
    # Create config
    config = create_config_file()
    
    print("\n✅ Setup guide completed!")
    print("Next steps:")
    print("1. Install required packages")
    print("2. Configure TWS/Gateway with API enabled")
    print("3. Run: python test_ibkr_connection.py")
    print("4. Run: python elite_superintelligence_paper_trading.py")
    
    print(f"\n📁 Files created:")
    print("• test_ibkr_connection.py")
    print("• ibkr_config.json")
    print("• interactive_brokers_setup.py")

if __name__ == "__main__":
    main()