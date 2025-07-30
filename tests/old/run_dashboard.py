#!/usr/bin/env python3
"""
Lanceur du Dashboard AlphaBot Streamlit
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Lance le dashboard Streamlit"""
    
    print("🚀 Lancement du Dashboard AlphaBot...")
    
    # Vérifier que streamlit est installé
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} détecté")
    except ImportError:
        print("❌ Streamlit non installé")
        print("💡 Installation: pip install streamlit")
        return 1
    
    # Chemin vers l'app streamlit
    dashboard_path = Path(__file__).parent.parent / "alphabot" / "dashboard" / "streamlit_app.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard non trouvé: {dashboard_path}")
        return 1
    
    print(f"📊 Dashboard: {dashboard_path}")
    print(f"🌐 URL: http://localhost:8501")
    print(f"⚡ Auto-refresh activé (30s)")
    print("")
    print("🔗 Navigation:")
    print("   📊 Live Trading    - Monitoring temps réel")
    print("   📈 Backtest        - Résultats historiques") 
    print("   📋 Agent Status    - Statut des agents")
    print("   ⚙️  Configuration   - Paramètres système")
    print("")
    print("🛑 Pour arrêter: Ctrl+C")
    print("=" * 50)
    
    # Lancer streamlit
    try:
        cmd = [
            sys.executable, 
            "-m", "streamlit", 
            "run", 
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ]
        
        subprocess.run(cmd, cwd=dashboard_path.parent.parent.parent)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard arrêté par l'utilisateur")
        return 0
    except Exception as e:
        print(f"❌ Erreur lancement dashboard: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)