#!/usr/bin/env python3
"""
Générateur de diagramme de Gantt pour AlphaBot
Lit planning.yml et génère un PNG avec matplotlib
"""

import yaml
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import argparse
from pathlib import Path
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_planning(planning_file: str) -> dict:
    """Charge le fichier de planning YAML"""
    try:
        with open(planning_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Erreur chargement planning: {e}")
        return None


def parse_date(date_str: str) -> datetime:
    """Parse une date au format YYYY-MM-DD"""
    return datetime.strptime(date_str, "%Y-%m-%d")


def get_status_color(status: str) -> str:
    """Retourne la couleur selon le statut"""
    colors = {
        'completed': '#28a745',    # Vert
        'in_progress': '#ffc107',  # Jaune
        'pending': '#6c757d',      # Gris
        'blocked': '#dc3545'       # Rouge
    }
    return colors.get(status, '#6c757d')


def create_gantt_chart(planning: dict, output_file: str = 'gantt_chart.png'):
    """Crée le diagramme de Gantt"""
    
    if not planning:
        print("Aucune donnée de planning disponible")
        return
    
    # Configuration du graphique
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Graphique principal - Phases
    phases = planning.get('phases', [])
    phase_names = []
    phase_starts = []
    phase_durations = []
    phase_colors = []
    
    for i, phase in enumerate(phases):
        start_date = parse_date(phase['start'])
        end_date = parse_date(phase['end'])
        duration = (end_date - start_date).days
        
        phase_names.append(phase['name'])
        phase_starts.append(start_date)
        phase_durations.append(duration)
        phase_colors.append(get_status_color(phase['status']))
    
    # Barres horizontales pour les phases
    y_positions = range(len(phase_names))
    bars = ax1.barh(y_positions, phase_durations, left=phase_starts, 
                    color=phase_colors, alpha=0.8, height=0.6)
    
    # Annotations des phases
    for i, (bar, phase) in enumerate(zip(bars, phases)):
        # Statut et durée
        status_emoji = {'completed': '✅', 'in_progress': '🔄', 'pending': '⏳'}.get(phase['status'], '❓')
        duration_weeks = phase_durations[i] // 7
        
        # Texte au centre de la barre
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                f"{status_emoji} {duration_weeks}w", 
                ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Configuration axes principales
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(phase_names)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlabel('Timeline 2025')
    ax1.set_title(f"{planning.get('title', 'AlphaBot Roadmap')}\n"
                  f"📊 {len([p for p in phases if p['status'] == 'completed'])}/{len(phases)} phases terminées", 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Format des dates
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_minor_locator(mdates.WeekdayLocator())
    
    # Graphique KPIs
    kpis = planning.get('kpis', {})
    if kpis:
        kpi_data = []
        
        # KPIs performance
        for kpi in kpis.get('performance', []):
            status = '✅' if 'TBD' not in kpi.get('current', 'TBD') else '⏳'
            kpi_data.append(f"{kpi['name']}: {kpi['target']} {status}")
        
        # KPIs techniques
        for kpi in kpis.get('technical', []):
            status = '✅' if '✅' in kpi.get('current', '') else '⏳'
            kpi_data.append(f"{kpi['name']}: {kpi['target']} {status}")
        
        # Affichage KPIs
        ax2.text(0.02, 0.8, "🎯 KPIs Projet:", fontweight='bold', fontsize=12, transform=ax2.transAxes)
        for i, kpi_text in enumerate(kpi_data):
            ax2.text(0.02, 0.6 - i*0.15, f"• {kpi_text}", fontsize=10, transform=ax2.transAxes)
    
    # Info ressources
    resources = planning.get('resources', {})
    if resources:
        resource_text = f"👥 Team: {resources.get('team_size', 'N/A')} | " \
                       f"💰 Budget: {resources.get('budget', 'N/A')} | " \
                       f"💻 Hardware: {resources.get('hardware', 'N/A')}"
        ax2.text(0.02, 0.1, resource_text, fontsize=9, style='italic', transform=ax2.transAxes)
    
    # Légende statuts
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#28a745', alpha=0.8, label='✅ Terminé'),
        plt.Rectangle((0,0),1,1, facecolor='#ffc107', alpha=0.8, label='🔄 En cours'),
        plt.Rectangle((0,0),1,1, facecolor='#6c757d', alpha=0.8, label='⏳ Planifié')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Masquer axes du graphique KPIs
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Ligne verticale "aujourd'hui"
    today = datetime.now()
    ax1.axvline(x=today, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Aujourd\'hui')
    
    # Ajustements finaux
    plt.tight_layout()
    
    # Sauvegarde
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Diagramme de Gantt généré: {output_file}")
    
    return output_file


def generate_status_report(planning: dict) -> str:
    """Génère un rapport de statut textuel"""
    if not planning:
        return "Aucune donnée disponible"
    
    phases = planning.get('phases', [])
    total_phases = len(phases)
    completed_phases = len([p for p in phases if p['status'] == 'completed'])
    in_progress_phases = len([p for p in phases if p['status'] == 'in_progress'])
    
    # Calcul du pourcentage d'avancement
    progress_percent = (completed_phases / total_phases * 100) if total_phases > 0 else 0
    
    report = f"""
📊 RAPPORT DE STATUT - {planning.get('title', 'AlphaBot')}
{'='*60}

🏁 AVANCEMENT GLOBAL: {progress_percent:.1f}%
   Phases terminées  : {completed_phases}/{total_phases}
   Phases en cours   : {in_progress_phases}
   
📋 DÉTAIL PAR PHASE:
"""
    
    for phase in phases:
        status_emoji = {'completed': '✅', 'in_progress': '🔄', 'pending': '⏳', 'blocked': '❌'}.get(phase['status'], '❓')
        start_date = parse_date(phase['start'])
        end_date = parse_date(phase['end'])
        duration_weeks = (end_date - start_date).days // 7
        
        report += f"   {status_emoji} {phase['name']}: {phase['start']} → {phase['end']} ({duration_weeks}w)\n"
    
    # KPIs
    kpis = planning.get('kpis', {})
    if kpis:
        report += f"\n🎯 KPIS:\n"
        for category, kpi_list in kpis.items():
            for kpi in kpi_list:
                current = kpi.get('current', 'TBD')
                status = '✅' if 'TBD' not in current and '✅' in current else '⏳'
                report += f"   {status} {kpi['name']}: {kpi['target']} (actuel: {current})\n"
    
    report += f"\n📅 Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    return report


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Générateur de Gantt AlphaBot")
    parser.add_argument("--planning", default="planning.yml", help="Fichier de planning YAML")
    parser.add_argument("--output", default="docs/gantt_chart.png", help="Fichier de sortie PNG")
    parser.add_argument("--report", action="store_true", help="Générer aussi un rapport textuel")
    parser.add_argument("--show", action="store_true", help="Afficher le graphique")
    
    args = parser.parse_args()
    
    # Vérifier que le fichier planning existe
    if not Path(args.planning).exists():
        print(f"❌ Fichier de planning non trouvé: {args.planning}")
        return 1
    
    # Créer le dossier de sortie si nécessaire
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger le planning
    planning = load_planning(args.planning)
    if not planning:
        return 1
    
    # Générer le diagramme
    try:
        output_file = create_gantt_chart(planning, args.output)
        
        # Rapport textuel
        if args.report:
            report = generate_status_report(planning)
            report_file = str(Path(args.output).with_suffix('.txt'))
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📝 Rapport généré: {report_file}")
            print(report)
        
        # Affichage
        if args.show:
            plt.show()
        
        print("✅ Génération terminée avec succès!")
        return 0
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())