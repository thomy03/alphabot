#!/usr/bin/env python3
"""
Analyseur de registre de risques AlphaBot
Génère des rapports et visualisations à partir de risk_register.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_risk_register(file_path: str) -> pd.DataFrame:
    """Charge le registre de risques"""
    try:
        df = pd.read_csv(file_path)
        df['Next_Review'] = pd.to_datetime(df['Next_Review'])
        return df
    except Exception as e:
        print(f"Erreur chargement registre: {e}")
        return None


def analyze_risks(df: pd.DataFrame) -> dict:
    """Analyse les risques et génère des statistiques"""
    if df is None or df.empty:
        return {}
    
    analysis = {}
    
    # Statistiques générales
    analysis['total_risks'] = len(df)
    analysis['by_category'] = df['Category'].value_counts().to_dict()
    analysis['by_status'] = df['Status'].value_counts().to_dict()
    analysis['by_impact'] = df['Impact'].value_counts().to_dict()
    analysis['by_probability'] = df['Probability'].value_counts().to_dict()
    
    # Scores de risque
    analysis['avg_risk_score'] = df['Risk_Score'].mean()
    analysis['max_risk_score'] = df['Risk_Score'].max()
    analysis['high_risk_count'] = len(df[df['Risk_Score'] >= 6])
    
    # Risques nécessitant une révision urgente
    today = datetime.now()
    upcoming_reviews = df[df['Next_Review'] <= today + timedelta(days=30)]
    analysis['upcoming_reviews'] = len(upcoming_reviews)
    analysis['overdue_reviews'] = len(df[df['Next_Review'] < today])
    
    # Top risques
    analysis['top_risks'] = df.nlargest(5, 'Risk_Score')[['ID', 'Risk_Description', 'Risk_Score']].to_dict('records')
    
    return analysis


def create_risk_dashboard(df: pd.DataFrame, output_file: str = 'docs/risk_dashboard.png'):
    """Crée un dashboard visuel des risques"""
    
    if df is None or df.empty:
        print("Aucune donnée de risque disponible")
        return
    
    # Configuration du graphique
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🚨 Dashboard de Gestion des Risques - AlphaBot', fontsize=16, fontweight='bold')
    
    # 1. Distribution par catégorie
    category_counts = df['Category'].value_counts()
    colors = plt.cm.Set3(range(len(category_counts)))
    ax1.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', colors=colors)
    ax1.set_title('📊 Répartition par Catégorie')
    
    # 2. Matrice Impact vs Probabilité
    impact_map = {'Low': 1, 'Medium': 2, 'High': 3}
    prob_map = {'Low': 1, 'Medium': 2, 'High': 3}
    
    df['Impact_Num'] = df['Impact'].map(impact_map)
    df['Probability_Num'] = df['Probability'].map(prob_map)
    
    scatter = ax2.scatter(df['Probability_Num'], df['Impact_Num'], 
                         c=df['Risk_Score'], s=df['Risk_Score']*20, 
                         alpha=0.6, cmap='YlOrRd')
    ax2.set_xlabel('Probabilité')
    ax2.set_ylabel('Impact')
    ax2.set_title('🎯 Matrice Risques (taille = score)')
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(['Low', 'Medium', 'High'])
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(['Low', 'Medium', 'High'])
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Risk Score')
    
    # 3. Évolution des statuts
    status_counts = df['Status'].value_counts()
    status_colors = {'Active': '#dc3545', 'Planned': '#ffc107', 'Monitored': '#17a2b8', 'Resolved': '#28a745'}
    colors = [status_colors.get(status, '#6c757d') for status in status_counts.index]
    
    bars = ax3.bar(status_counts.index, status_counts.values, color=colors)
    ax3.set_title('🔄 Statuts des Risques')
    ax3.set_ylabel('Nombre de risques')
    
    # Annotations
    for bar, count in zip(bars, status_counts.values):
        ax3.annotate(str(count), xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # 4. Timeline des révisions
    today = datetime.now()
    df['Days_to_Review'] = (df['Next_Review'] - today).dt.days
    
    # Catégoriser les révisions
    overdue = len(df[df['Days_to_Review'] < 0])
    this_week = len(df[(df['Days_to_Review'] >= 0) & (df['Days_to_Review'] <= 7)])
    this_month = len(df[(df['Days_to_Review'] > 7) & (df['Days_to_Review'] <= 30)])
    later = len(df[df['Days_to_Review'] > 30])
    
    timeline_data = [overdue, this_week, this_month, later]
    timeline_labels = ['En retard', 'Cette semaine', 'Ce mois', 'Plus tard']
    timeline_colors = ['#dc3545', '#fd7e14', '#ffc107', '#28a745']
    
    ax4.bar(timeline_labels, timeline_data, color=timeline_colors)
    ax4.set_title('📅 Timeline des Révisions')
    ax4.set_ylabel('Nombre de risques')
    ax4.tick_params(axis='x', rotation=45)
    
    # Annotations
    for i, count in enumerate(timeline_data):
        ax4.annotate(str(count), xy=(i, count),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Ajustements
    plt.tight_layout()
    
    # Créer le dossier de sortie si nécessaire
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Dashboard risques généré: {output_file}")
    
    return output_file


def generate_risk_report(df: pd.DataFrame, analysis: dict) -> str:
    """Génère un rapport textuel des risques"""
    
    if not analysis:
        return "Aucune analyse disponible"
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    report = f"""
🚨 RAPPORT D'ANALYSE DES RISQUES - AlphaBot
{'='*60}
📅 Date du rapport: {today}

📊 VUE D'ENSEMBLE:
   Total des risques     : {analysis['total_risks']}
   Score moyen          : {analysis['avg_risk_score']:.1f}/9
   Risques élevés (≥6)  : {analysis['high_risk_count']}
   Révisions en retard  : {analysis['overdue_reviews']}
   Révisions urgentes   : {analysis['upcoming_reviews']} (30 jours)

📋 RÉPARTITION PAR CATÉGORIE:
"""
    
    for category, count in analysis['by_category'].items():
        pct = (count / analysis['total_risks']) * 100
        report += f"   {category:<12} : {count:2d} risques ({pct:4.1f}%)\n"
    
    report += f"\n🎯 RÉPARTITION PAR STATUT:\n"
    for status, count in analysis['by_status'].items():
        emoji = {'Active': '🔴', 'Planned': '🟡', 'Monitored': '🔵', 'Resolved': '🟢'}.get(status, '⚪')
        report += f"   {emoji} {status:<10} : {count:2d} risques\n"
    
    report += f"\n⚠️  TOP 5 RISQUES PRIORITAIRES:\n"
    for i, risk in enumerate(analysis['top_risks'], 1):
        report += f"   {i}. [{risk['ID']}] Score {risk['Risk_Score']} - {risk['Risk_Description'][:60]}...\n"
    
    # Recommandations
    report += f"\n💡 RECOMMANDATIONS:\n"
    
    if analysis['overdue_reviews'] > 0:
        report += f"   🚨 URGENT: {analysis['overdue_reviews']} révisions en retard\n"
    
    if analysis['high_risk_count'] > 5:
        report += f"   ⚠️  Trop de risques élevés ({analysis['high_risk_count']}) - prioriser la mitigation\n"
    
    if analysis['by_status'].get('Active', 0) > 10:
        report += f"   📋 Beaucoup de risques actifs - revoir les stratégies de mitigation\n"
    
    # Actions immédiates
    if df is not None:
        urgent_risks = df[df['Risk_Score'] >= 8]
        if not urgent_risks.empty:
            report += f"\n🚨 ACTIONS IMMÉDIATES REQUISES:\n"
            for _, risk in urgent_risks.iterrows():
                report += f"   • [{risk['ID']}] {risk['Risk_Description']} (Score: {risk['Risk_Score']})\n"
                report += f"     → Owner: {risk['Owner']} | Révision: {risk['Next_Review'].strftime('%Y-%m-%d')}\n"
    
    # Tendances
    technical_risks = analysis['by_category'].get('Technical', 0)
    financial_risks = analysis['by_category'].get('Financial', 0)
    
    report += f"\n📈 TENDANCES:\n"
    if technical_risks > financial_risks:
        report += f"   • Focus technique: {technical_risks} risques tech vs {financial_risks} financiers\n"
    else:
        report += f"   • Risques équilibrés entre technique et financier\n"
    
    resolved_pct = (analysis['by_status'].get('Resolved', 0) / analysis['total_risks']) * 100
    if resolved_pct > 20:
        report += f"   • Bonne résolution: {resolved_pct:.1f}% des risques résolus\n"
    else:
        report += f"   • Peu de résolutions: seulement {resolved_pct:.1f}% résolus\n"
    
    report += f"\n📋 PROCHAINES ACTIONS:\n"
    report += f"   1. Réviser les {analysis['upcoming_reviews']} risques prioritaires\n"
    report += f"   2. Mettre à jour les stratégies de mitigation\n"
    report += f"   3. Planifier révision trimestrielle complète\n"
    
    report += f"\n---\n📊 Rapport généré automatiquement par scripts/risk_analysis.py\n"
    
    return report


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Analyseur de registre de risques AlphaBot")
    parser.add_argument("--register", default="risk_register.csv", help="Fichier registre CSV")
    parser.add_argument("--output-chart", default="docs/risk_dashboard.png", help="Dashboard PNG")
    parser.add_argument("--output-report", default="docs/risk_analysis.md", help="Rapport Markdown")
    parser.add_argument("--show", action="store_true", help="Afficher le dashboard")
    parser.add_argument("--format", choices=['txt', 'md'], default='md', help="Format du rapport")
    
    args = parser.parse_args()
    
    # Vérifier que le fichier existe
    if not Path(args.register).exists():
        print(f"❌ Registre de risques non trouvé: {args.register}")
        return 1
    
    # Charger les données
    df = load_risk_register(args.register)
    if df is None:
        return 1
    
    print(f"📊 Chargé {len(df)} risques depuis {args.register}")
    
    # Analyse
    analysis = analyze_risks(df)
    
    # Générer le dashboard
    try:
        chart_file = create_risk_dashboard(df, args.output_chart)
        
        # Générer le rapport
        report = generate_risk_report(df, analysis)
        
        # Sauvegarder le rapport
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(args.output_report, 'w', encoding='utf-8') as f:
            if args.format == 'md':
                f.write(f"# {report}")
            else:
                f.write(report)
        
        print(f"📝 Rapport généré: {args.output_report}")
        print(report)
        
        # Affichage
        if args.show:
            plt.show()
        
        print("✅ Analyse terminée avec succès!")
        return 0
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())