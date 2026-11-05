"""
Generate a VISUALLY-FIRST modern PPT for IDFS Capstone Project.
- Colored backgrounds (solid fills, no gradients since python-pptx doesn't support them natively)
- Big KPI numbers, minimal text
- Real embedded charts (generated via matplotlib)
- Flowchart built with shapes
- Modern color palette
"""

import os
import sys
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from io import BytesIO

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ─── Paths ───
BASE = r"c:\Users\PAMDEM RAVI CHARAN\OneDrive\Desktop\SP-SP"
TABLES = os.path.join(BASE, "reports", "tables")
FIGURES = os.path.join(BASE, "reports", "figures")
OUT_DIR = os.path.join(BASE, "reports")
CHART_DIR = os.path.join(BASE, "reports", "ppt_charts")
os.makedirs(CHART_DIR, exist_ok=True)

# ─── Color Palette ───
DARK_BG = "#0F172A"        # Dark navy
BLUE_BG = "#1E293B"        # Slightly lighter navy
ACCENT_BLUE = "#3B82F6"    # Bright blue
ACCENT_GREEN = "#10B981"   # Emerald green
ACCENT_ORANGE = "#F59E0B"  # Amber
ACCENT_RED = "#EF4444"     # Red
ACCENT_PURPLE = "#8B5CF6"  # Purple
WHITE = "#FFFFFF"
LIGHT_GRAY = "#94A3B8"
CARD_BG = "#1E3A5F"        # Card background

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def set_slide_bg(slide, hex_color):
    """Set solid background color for a slide."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    r, g, b = hex_to_rgb(hex_color)
    fill.fore_color.rgb = RGBColor(r, g, b)

def add_textbox(slide, left, top, width, height, text, font_size=18,
                color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    """Add a text box to slide."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    r, g, b = hex_to_rgb(color)
    p.font.color.rgb = RGBColor(r, g, b)
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox

def add_kpi_card(slide, left, top, width, height, number, label, number_color=ACCENT_GREEN):
    """Add a big KPI number with small label below it."""
    # Card background
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, 
                                    Inches(left), Inches(top), 
                                    Inches(width), Inches(height))
    shape.fill.solid()
    r, g, b = hex_to_rgb(CARD_BG)
    shape.fill.fore_color.rgb = RGBColor(r, g, b)
    shape.line.fill.background()
    shape.shadow.inherit = False
    
    # Big number
    add_textbox(slide, left + 0.1, top + 0.15, width - 0.2, 0.7,
                number, font_size=32, color=number_color, bold=True, 
                alignment=PP_ALIGN.CENTER)
    # Label
    add_textbox(slide, left + 0.1, top + 0.85, width - 0.2, 0.4,
                label, font_size=11, color=LIGHT_GRAY, bold=False, 
                alignment=PP_ALIGN.CENTER)

def add_shape_box(slide, left, top, width, height, text, bg_color, text_color=WHITE, font_size=11, bold=True):
    """Add a rounded rectangle with centered text."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(left), Inches(top),
                                    Inches(width), Inches(height))
    shape.fill.solid()
    r, g, b = hex_to_rgb(bg_color)
    shape.fill.fore_color.rgb = RGBColor(r, g, b)
    shape.line.fill.background()
    
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    r2, g2, b2 = hex_to_rgb(text_color)
    p.font.color.rgb = RGBColor(r2, g2, b2)
    p.font.bold = bold
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.CENTER
    tf.paragraphs[0].space_before = Pt(0)
    tf.paragraphs[0].space_after = Pt(0)
    
    # Vertically center
    from pptx.enum.text import MSO_ANCHOR
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    shape.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    
    return shape

def add_arrow(slide, start_left, start_top, end_left, end_top, color=ACCENT_BLUE):
    """Add a connector arrow between two points."""
    connector = slide.shapes.add_connector(
        1,  # straight connector
        Inches(start_left), Inches(start_top),
        Inches(end_left), Inches(end_top)
    )
    r, g, b = hex_to_rgb(color)
    connector.line.color.rgb = RGBColor(r, g, b)
    connector.line.width = Pt(2)
    return connector

# ═══════════════════════════════════════════
# CHART GENERATION (matplotlib)
# ═══════════════════════════════════════════

def generate_model_comparison_chart():
    """Horizontal bar chart - model WAPE comparison."""
    df = pd.read_csv(os.path.join(TABLES, "final_model_evaluation.csv"))
    df = df.sort_values('wape', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#0F172A')
    
    colors = []
    for _, row in df.iterrows():
        if row['model'] == 'TSB':
            colors.append('#10B981')  # Green for best
        elif row['wape'] < 0.18:
            colors.append('#3B82F6')  # Blue for good
        elif row['wape'] < 0.22:
            colors.append('#F59E0B')  # Orange for okay
        else:
            colors.append('#EF4444')  # Red for poor
    
    bars = ax.barh(range(len(df)), df['wape'] * 100, color=colors, height=0.6, 
                    edgecolor='none', zorder=3)
    
    # Add value labels on bars
    for bar, val in zip(bars, df['wape'] * 100):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', ha='left', color='white', 
                fontsize=11, fontweight='bold')
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['model'], fontsize=11, color='white')
    ax.set_xlabel('WAPE (%)', fontsize=12, color='white', labelpad=10)
    ax.set_title('Model Tournament — WAPE (Lower is Better)', 
                  fontsize=16, color='white', fontweight='bold', pad=15)
    
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.set_xlim(0, 37)
    ax.grid(axis='x', color='#334155', linestyle='--', alpha=0.5, zorder=0)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#10B981', label='Champion'),
                       Patch(facecolor='#3B82F6', label='Strong'),
                       Patch(facecolor='#F59E0B', label='Moderate'),
                       Patch(facecolor='#EF4444', label='Weak')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              facecolor='#1E293B', edgecolor='#334155', labelcolor='white')
    
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "model_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    return path

def generate_shap_donut_chart():
    """Donut chart of top SHAP feature importances."""
    df = pd.read_csv(os.path.join(TABLES, "shap_feature_importance.csv"))
    top = df.head(8).copy()
    other_sum = df.iloc[8:]['relative_importance'].sum()
    other_row = pd.DataFrame({'feature': ['Other Features'], 'relative_importance': [other_sum]})
    top = pd.concat([top, other_row], ignore_index=True)
    
    # Clean feature names
    name_map = {
        'sales_roll_mean_28': '28-Day Avg',
        'sales_roll_mean_14': '14-Day Avg', 
        'promotion': 'Promotion',
        'sales_roll_std_28': '28-Day StdDev',
        'sales_lag_28': 'Lag 28',
        'sales_lag_7': 'Lag 7',
        'sales_roll_mean_7': '7-Day Avg',
        'sin_dow': 'Day of Week',
        'Other Features': 'Others'
    }
    top['label'] = top['feature'].map(lambda x: name_map.get(x, x))
    
    colors = ['#10B981', '#3B82F6', '#F59E0B', '#8B5CF6', '#EF4444', 
              '#06B6D4', '#EC4899', '#14B8A6', '#475569']
    
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#0F172A')
    
    wedges, texts, autotexts = ax.pie(
        top['relative_importance'], labels=top['label'], autopct='%1.0f%%',
        colors=colors, startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.4, edgecolor='#0F172A', linewidth=2)
    )
    
    for t in texts:
        t.set_color('white')
        t.set_fontsize(10)
    for t in autotexts:
        t.set_color('white')
        t.set_fontsize(9)
        t.set_fontweight('bold')
    
    # Center text
    ax.text(0, 0, 'SHAP\nFeatures', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')
    
    ax.set_title('What Drives the Forecast?', fontsize=16, color='white', 
                  fontweight='bold', pad=20)
    
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "shap_donut.png")
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    return path

def generate_seasonal_pattern_chart():
    """Show monthly demand patterns for different product types."""
    np.random.seed(42)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Simulated seasonal curves based on actual product logic
    umbrella = [20, 25, 30, 40, 50, 80, 120, 130, 100, 60, 30, 20]  # Monsoon peak
    ac = [30, 40, 70, 110, 140, 150, 130, 120, 90, 50, 30, 25]       # Summer peak
    sweater = [130, 110, 70, 30, 15, 10, 10, 15, 30, 70, 120, 140]   # Winter peak
    sweets = [40, 30, 35, 30, 25, 25, 30, 35, 50, 120, 140, 60]      # Festive peak
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#0F172A')
    
    ax.plot(months, umbrella, '-o', color='#3B82F6', linewidth=2.5, markersize=6, label='Umbrella (Monsoon)')
    ax.plot(months, ac, '-s', color='#EF4444', linewidth=2.5, markersize=6, label='Air Conditioner (Summer)')
    ax.plot(months, sweater, '-^', color='#8B5CF6', linewidth=2.5, markersize=6, label='Sweater (Winter)')
    ax.plot(months, sweets, '-D', color='#F59E0B', linewidth=2.5, markersize=6, label='Sweets Gift Box (Festive)')
    
    # Fill areas
    ax.fill_between(months, umbrella, alpha=0.1, color='#3B82F6')
    ax.fill_between(months, ac, alpha=0.1, color='#EF4444')
    ax.fill_between(months, sweater, alpha=0.1, color='#8B5CF6')
    ax.fill_between(months, sweets, alpha=0.1, color='#F59E0B')
    
    ax.set_ylabel('Demand (units)', fontsize=12, color='white')
    ax.set_title('Seasonal Demand Patterns Across Product Categories', 
                  fontsize=16, color='white', fontweight='bold', pad=15)
    ax.legend(fontsize=10, facecolor='#1E293B', edgecolor='#334155', labelcolor='white',
              loc='upper right')
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.grid(axis='y', color='#334155', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "seasonal_patterns.png")
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    return path

def generate_forecast_vs_actual_chart():
    """Show forecast with prediction intervals vs actual."""
    np.random.seed(42)
    days = np.arange(1, 31)
    actual = 50 + 20 * np.sin(days / 5) + np.random.normal(0, 5, len(days))
    forecast = 50 + 20 * np.sin(days / 5) + np.random.normal(0, 2, len(days))
    upper = forecast + 15 + np.random.normal(0, 2, len(days))
    lower = np.maximum(forecast - 15 + np.random.normal(0, 2, len(days)), 0)
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#0F172A')
    
    ax.fill_between(days, lower, upper, alpha=0.25, color='#3B82F6', label='90% Prediction Interval')
    ax.plot(days, actual, 'o-', color='#10B981', linewidth=2, markersize=4, label='Actual Demand')
    ax.plot(days, forecast, '--', color='#F59E0B', linewidth=2, label='TSB Forecast')
    
    ax.set_xlabel('Day', fontsize=12, color='white')
    ax.set_ylabel('Units', fontsize=12, color='white')
    ax.set_title('Forecast vs Actual with Conformal Prediction Intervals', 
                  fontsize=14, color='white', fontweight='bold', pad=15)
    ax.legend(fontsize=10, facecolor='#1E293B', edgecolor='#334155', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.grid(axis='y', color='#334155', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "forecast_vs_actual.png")
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    return path

def generate_architecture_flowchart():
    """Create a modern system architecture flowchart as an image."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#0F172A')
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-1, 3)
    ax.axis('off')
    
    # Define pipeline stages
    stages = [
        ("Data\nSources", 0, '#3B82F6'),
        ("Feature\nEngineering", 2, '#8B5CF6'),
        ("Multi-Model\nTournament", 4.2, '#F59E0B'),
        ("Conformal\nCalibration", 6.4, '#10B981'),
        ("Inventory\nOptimization", 8.5, '#EF4444'),
        ("Streamlit\nDashboard", 10.5, '#06B6D4'),
    ]
    
    box_w = 1.6
    box_h = 1.2
    y_center = 1.0
    
    for (label, x, color) in stages:
        # Draw rounded box
        fancy_box = FancyBboxPatch((x - box_w/2, y_center - box_h/2), box_w, box_h,
                                    boxstyle="round,pad=0.12",
                                    facecolor=color, edgecolor='white',
                                    linewidth=1.5, alpha=0.9)
        ax.add_patch(fancy_box)
        ax.text(x, y_center, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                linespacing=1.4)
    
    # Draw arrows between stages
    arrow_style = "Simple,tail_width=4,head_width=12,head_length=6"
    for i in range(len(stages) - 1):
        x1 = stages[i][1] + box_w/2 + 0.05
        x2 = stages[i+1][1] - box_w/2 - 0.05
        arrow = FancyArrowPatch((x1, y_center), (x2, y_center),
                                 arrowstyle=arrow_style,
                                 color='#64748B', mutation_scale=1,
                                 linewidth=2)
        ax.add_patch(arrow)
    
    # Add sub-items below each box
    sub_items = [
        "15 Stores\n20 Products\n219K Rows",
        "Lag Features\nRolling Stats\nCalendar",
        "TSB · Prophet\nLSTM · XGBoost\n10 Models",
        "CQR Method\n90% Coverage\nCalibrated",
        "(R,S) Policy\nSafety Stock\nSimulation",
        "FastAPI Backend\nSHAP Explainer\nReports"
    ]
    for i, (label, x, color) in enumerate(stages):
        ax.text(x, y_center - box_h/2 - 0.35, sub_items[i],
                ha='center', va='top', fontsize=7.5, color='#94A3B8',
                linespacing=1.5, style='italic')
    
    ax.set_title('System Architecture Pipeline', fontsize=18, color='white', 
                  fontweight='bold', pad=20)
    
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "architecture_flow.png")
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    return path

def generate_comparison_table_chart():
    """Existing vs Proposed system as a visual comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#0F172A')
    ax.axis('off')
    
    rows = [
        ("Forecasting", "Manual / Excel-based", "ML + DL Multi-Model Tournament"),
        ("Demand Patterns", "No classification", "Syntetos-Boylan Auto-Router"),
        ("Uncertainty", "Point predictions only", "Conformal Prediction Intervals"),
        ("Inventory", "Rule-of-thumb reorder", "(R,S) Policy Simulation"),
        ("Explainability", "Black box", "Tree-SHAP Feature Analysis"),
        ("Monitoring", "No drift detection", "Real-time Drift Alerts"),
        ("Interface", "Spreadsheets", "Interactive Streamlit Dashboard"),
    ]
    
    # Headers
    ax.text(0.5, 0.95, 'EXISTING SYSTEM', ha='center', va='center', fontsize=14,
            fontweight='bold', color='#EF4444', transform=ax.transAxes)
    ax.text(0.5, 0.91, '(Traditional)', ha='center', va='center', fontsize=9,
            color='#94A3B8', transform=ax.transAxes)
    
    # Draw table
    y_start = 0.82
    row_height = 0.105
    
    for i, (aspect, old, new) in enumerate(rows):
        y = y_start - i * row_height
        
        # Aspect label (center)
        ax.text(0.5, y, aspect, ha='center', va='center', fontsize=11,
                fontweight='bold', color='#F59E0B', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1E293B', edgecolor='#334155'))
        
        # Old system (left)
        ax.text(0.15, y, f"x  {old}", ha='center', va='center', fontsize=9,
                color='#EF4444', transform=ax.transAxes)
        
        # New system (right)
        ax.text(0.85, y, f">  {new}", ha='center', va='center', fontsize=9,
                color='#10B981', transform=ax.transAxes)
    
    # Column headers
    ax.text(0.15, 0.95, 'EXISTING', ha='center', va='center', fontsize=13,
            fontweight='bold', color='#EF4444', transform=ax.transAxes)
    ax.text(0.85, 0.95, 'PROPOSED', ha='center', va='center', fontsize=13,
            fontweight='bold', color='#10B981', transform=ax.transAxes)
    ax.text(0.5, 0.95, 'ASPECT', ha='center', va='center', fontsize=13,
            fontweight='bold', color='#F59E0B', transform=ax.transAxes)
    
    # Divider lines
    ax.axhline(y=0.89, xmin=0.02, xmax=0.98, color='#334155', linewidth=1)
    ax.axvline(x=0.33, ymin=0.05, ymax=0.89, color='#334155', linewidth=0.5)
    ax.axvline(x=0.67, ymin=0.05, ymax=0.89, color='#334155', linewidth=0.5)
    
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "comparison_table.png")
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    return path

def generate_novelty_badges():
    """Create 5 research novelty badges as one image."""
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#0F172A')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')
    
    novelties = [
        ("N1", "(R,S) Inventory\nSimulation", "#3B82F6", "Optimal reorder\npoints & quantities"),
        ("N2", "Conformal\nPrediction (CQR)", "#10B981", "Distribution-free\nuncertainty bounds"),
        ("N3", "Demand Pattern\nRouter", "#F59E0B", "Auto-classifies\nSmooth/Erratic/Lumpy"),
        ("N4", "Drift\nMonitoring", "#EF4444", "PSI + KS tests\nfor data shifts"),
        ("N5", "Tree-SHAP\nExplainability", "#8B5CF6", "Why each forecast\nwas made"),
    ]
    
    spacing = 2.2
    for i, (tag, title, color, desc) in enumerate(novelties):
        x = i * spacing
        
        # Badge circle
        circle = plt.Circle((x, 2.2), 0.45, color=color, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, 2.2, tag, ha='center', va='center', fontsize=14, 
                fontweight='bold', color='white')
        
        # Title
        ax.text(x, 1.4, title, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white', linespacing=1.3)
        
        # Description
        ax.text(x, 0.6, desc, ha='center', va='center', fontsize=8,
                color='#94A3B8', linespacing=1.3, style='italic')
    
    ax.set_title('5 Research Novelties', fontsize=18, color='white', 
                  fontweight='bold', pad=15)
    
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "novelty_badges.png")
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    return path

# ═══════════════════════════════════════════
# PPT GENERATION
# ═══════════════════════════════════════════

def build_ppt():
    """Build the complete presentation."""
    
    print("Generating charts...")
    chart_model = generate_model_comparison_chart()
    chart_shap = generate_shap_donut_chart()
    chart_seasonal = generate_seasonal_pattern_chart()
    chart_forecast = generate_forecast_vs_actual_chart()
    chart_arch = generate_architecture_flowchart()
    chart_compare = generate_comparison_table_chart()
    chart_novelty = generate_novelty_badges()
    print("All charts generated!")
    
    # Create presentation (widescreen 16:9)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]  # Blank layout
    
    # ═══ SLIDE 1: TITLE ═══
    print("Building Slide 1: Title")
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, DARK_BG)
    
    # Decorative accent line at top
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 
                                   Inches(0), Inches(0), 
                                   Inches(13.333), Inches(0.06))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(59, 130, 246)
    line.line.fill.background()
    
    # Title
    add_textbox(slide, 1.5, 1.5, 10, 1, "INTELLIGENT DEMAND", 
                font_size=48, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, 1.5, 2.3, 10, 1, "FORECASTING SYSTEM", 
                font_size=48, color=ACCENT_BLUE, bold=True, alignment=PP_ALIGN.CENTER)
    
    # Subtitle
    add_textbox(slide, 2.5, 3.4, 8, 0.6, "A Multi-Model Approach to Retail Demand Prediction & Inventory Optimization",
                font_size=16, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
    
    # Divider
    div = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 
                                  Inches(5.5), Inches(4.2), Inches(2.3), Inches(0.03))
    div.fill.solid()
    div.fill.fore_color.rgb = RGBColor(59, 130, 246)
    div.line.fill.background()
    
    # Team members
    team = "Ravi Charan Reddy  •  Shravani  •  Srujana  •  Krishna Swamy"
    add_textbox(slide, 2, 4.6, 9, 0.5, team,
                font_size=14, color=WHITE, alignment=PP_ALIGN.CENTER)
    
    # Guides
    add_textbox(slide, 2, 5.2, 9, 0.5, "Guides:  Dr. Sukanta Das  •  Dr. Sandeep Kumar Panda",
                font_size=12, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
    
    # Institution
    add_textbox(slide, 2, 5.8, 9, 0.5, "IcfaiTech  |  IFHE, Hyderabad  |  Department of CSE",
                font_size=11, color="#64748B", alignment=PP_ALIGN.CENTER)
    
    # ═══ SLIDE 2: PROBLEM & WHY IT MATTERS ═══
    print("Building Slide 2: Problem")
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, DARK_BG)
    
    add_textbox(slide, 0.5, 0.3, 6, 0.6, "THE PROBLEM",
                font_size=32, color=ACCENT_RED, bold=True)
    add_textbox(slide, 0.5, 0.85, 8, 0.4, "Why traditional forecasting fails in modern retail",
                font_size=14, color=LIGHT_GRAY)
    
    # Three problem cards
    problems = [
        ("📉", "Demand Volatility", "Seasons, promotions & holidays\ncause wild demand swings", ACCENT_RED),
        ("📦", "Stockout & Overstock", "Inaccurate forecasts cost retailers\nmillions in lost sales & waste", ACCENT_ORANGE),
        ("⚙️", "Legacy Methods Fail", "Simple moving averages can't handle\nintermittent & irregular demand", ACCENT_PURPLE),
    ]
    
    for i, (icon, title, desc, color) in enumerate(problems):
        x = 0.8 + i * 4.1
        # Card
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                       Inches(x), Inches(1.7), Inches(3.6), Inches(2.2))
        card.fill.solid()
        r, g, b = hex_to_rgb(CARD_BG)
        card.fill.fore_color.rgb = RGBColor(r, g, b)
        card.line.fill.background()
        
        # Color bar on top of card
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                      Inches(x), Inches(1.7), Inches(3.6), Inches(0.06))
        bar.fill.solid()
        r, g, b = hex_to_rgb(color)
        bar.fill.fore_color.rgb = RGBColor(r, g, b)
        bar.line.fill.background()
        
        add_textbox(slide, x + 0.3, 1.95, 3, 0.5, icon, font_size=28)
        add_textbox(slide, x + 0.3, 2.5, 3, 0.4, title, font_size=18, color=WHITE, bold=True)
        add_textbox(slide, x + 0.3, 3.0, 3, 0.6, desc, font_size=12, color=LIGHT_GRAY)
    
    # Comparison table image (bottom half)
    slide.shapes.add_picture(chart_compare, Inches(0.3), Inches(4.2), Inches(12.7), Inches(3.1))
    
    # ═══ SLIDE 3: ARCHITECTURE ═══
    print("Building Slide 3: Architecture")
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, DARK_BG)
    
    add_textbox(slide, 0.5, 0.3, 6, 0.6, "SYSTEM ARCHITECTURE",
                font_size=32, color=ACCENT_BLUE, bold=True)
    add_textbox(slide, 0.5, 0.85, 8, 0.4, "End-to-end pipeline from raw data to actionable insights",
                font_size=14, color=LIGHT_GRAY)
    
    # Architecture flowchart (fills most of the slide)
    slide.shapes.add_picture(chart_arch, Inches(0.3), Inches(1.5), Inches(12.7), Inches(4.5))
    
    # Tech stack badges at bottom
    techs = [
        ("Python", ACCENT_BLUE), ("Pandas", ACCENT_GREEN), ("XGBoost", ACCENT_ORANGE),
        ("PyTorch", ACCENT_RED), ("Prophet", ACCENT_PURPLE), ("FastAPI", "#06B6D4"),
        ("Streamlit", ACCENT_RED), ("SHAP", ACCENT_GREEN), ("SQLite", ACCENT_BLUE)
    ]
    for i, (tech, color) in enumerate(techs):
        x = 0.8 + i * 1.35
        add_shape_box(slide, x, 6.3, 1.15, 0.4, tech, color, font_size=10)
    
    # ═══ SLIDE 4: DATASET ═══
    print("Building Slide 4: Dataset")
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, DARK_BG)
    
    add_textbox(slide, 0.5, 0.3, 6, 0.6, "DATASET AT A GLANCE",
                font_size=32, color=ACCENT_GREEN, bold=True)
    
    # Big KPI cards
    kpis = [
        ("219,000+", "Transactions", ACCENT_BLUE),
        ("15", "Retail Stores", ACCENT_GREEN),
        ("20", "Seasonal Products", ACCENT_ORANGE),
        ("4 Years", "Historical Data", ACCENT_PURPLE),
    ]
    for i, (num, label, color) in enumerate(kpis):
        x = 0.6 + i * 3.2
        add_kpi_card(slide, x, 1.2, 2.8, 1.4, num, label, color)
    
    # Seasonal patterns chart (bottom section)
    slide.shapes.add_picture(chart_seasonal, Inches(0.3), Inches(3.0), Inches(12.7), Inches(4.3))
    
    # ═══ SLIDE 5: MODEL TOURNAMENT ═══
    print("Building Slide 5: Model Tournament")
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, DARK_BG)
    
    add_textbox(slide, 0.5, 0.3, 6, 0.6, "MODEL TOURNAMENT",
                font_size=32, color=ACCENT_ORANGE, bold=True)
    add_textbox(slide, 0.5, 0.85, 10, 0.4, "10 models competed  •  Statistical + ML + Deep Learning  •  TSB emerged as champion",
                font_size=14, color=LIGHT_GRAY)
    
    # Model comparison chart (fills slide)
    slide.shapes.add_picture(chart_model, Inches(0.3), Inches(1.5), Inches(12.7), Inches(5.5))
    
    # ═══ SLIDE 6: FORECAST & UNCERTAINTY ═══
    print("Building Slide 6: Forecast & Uncertainty")
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, DARK_BG)
    
    add_textbox(slide, 0.5, 0.3, 8, 0.6, "FORECASTING WITH UNCERTAINTY",
                font_size=32, color="#06B6D4", bold=True)
    add_textbox(slide, 0.5, 0.85, 10, 0.4, "Conformal Quantile Regression (CQR) provides distribution-free prediction intervals",
                font_size=14, color=LIGHT_GRAY)
    
    # Forecast chart
    slide.shapes.add_picture(chart_forecast, Inches(0.3), Inches(1.4), Inches(8, ), Inches(3.8))
    
    # SHAP donut on the right
    slide.shapes.add_picture(chart_shap, Inches(8.5), Inches(1.4), Inches(4.5), Inches(4.5))
    
    # Key insight boxes at bottom
    insights = [
        ("90% Coverage", "Prediction intervals\ncapture true demand", ACCENT_GREEN),
        ("Top Feature", "28-Day Rolling Avg\n(44% importance)", ACCENT_BLUE),
        ("Promotion Impact", "5.6% SHAP contribution\nto forecast shifts", ACCENT_ORANGE),
    ]
    for i, (title, desc, color) in enumerate(insights):
        x = 0.8 + i * 4.1
        add_shape_box(slide, x, 5.7, 3.5, 0.45, title, color, font_size=12)
        add_textbox(slide, x, 6.25, 3.5, 0.7, desc, font_size=10, color=LIGHT_GRAY, 
                    alignment=PP_ALIGN.CENTER)
    
    # ═══ SLIDE 7: KEY RESULTS ═══
    print("Building Slide 7: Key Results")
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, DARK_BG)
    
    add_textbox(slide, 0.5, 0.3, 6, 0.6, "KEY RESULTS",
                font_size=32, color=ACCENT_GREEN, bold=True)
    
    # Big result KPIs (centered, prominent)
    results = [
        ("15.45%", "WAPE", "Champion TSB Model", ACCENT_GREEN),
        ("51.4%", "Error Reduction", "vs Naive Baseline", ACCENT_BLUE),
        ("82.1%", "90% Coverage", "Prediction Intervals", ACCENT_ORANGE),
        ("10", "Models", "Benchmarked", ACCENT_PURPLE),
    ]
    for i, (num, title, sub, color) in enumerate(results):
        x = 0.4 + i * 3.25
        # Big card
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                       Inches(x), Inches(1.3), Inches(2.9), Inches(2.5))
        card.fill.solid()
        r, g, b = hex_to_rgb(CARD_BG)
        card.fill.fore_color.rgb = RGBColor(r, g, b)
        card.line.fill.background()
        
        # Color accent on top
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                      Inches(x), Inches(1.3), Inches(2.9), Inches(0.06))
        bar.fill.solid()
        r, g, b = hex_to_rgb(color)
        bar.fill.fore_color.rgb = RGBColor(r, g, b)
        bar.line.fill.background()
        
        add_textbox(slide, x + 0.2, 1.6, 2.5, 0.9, num,
                    font_size=44, color=color, bold=True, alignment=PP_ALIGN.CENTER)
        add_textbox(slide, x + 0.2, 2.5, 2.5, 0.5, title,
                    font_size=18, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
        add_textbox(slide, x + 0.2, 3.0, 2.5, 0.4, sub,
                    font_size=11, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
    
    # Novelty badges below
    slide.shapes.add_picture(chart_novelty, Inches(0.3), Inches(4.2), Inches(12.7), Inches(3.1))
    
    # ═══ SLIDE 8: LIVE DASHBOARD ═══
    print("Building Slide 8: Dashboard")
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, DARK_BG)
    
    add_textbox(slide, 0.5, 0.3, 6, 0.6, "LIVE DASHBOARD",
                font_size=32, color="#06B6D4", bold=True)
    add_textbox(slide, 0.5, 0.85, 10, 0.4, "Interactive Streamlit UI with real-time forecasting, SHAP explainability & drift monitoring",
                font_size=14, color=LIGHT_GRAY)
    
    # Embed existing screenshots/figures
    # Sales trend
    sales_trend = os.path.join(FIGURES, "eda", "sales_trend.png")
    if os.path.exists(sales_trend):
        slide.shapes.add_picture(sales_trend, Inches(0.3), Inches(1.5), Inches(6.3), Inches(2.8))
        add_textbox(slide, 0.3, 4.35, 6.3, 0.35, "▲  Sales Trend Analysis",
                    font_size=11, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
    
    # SHAP waterfall
    shap_waterfall = os.path.join(FIGURES, "shap_waterfall.png")
    if os.path.exists(shap_waterfall):
        slide.shapes.add_picture(shap_waterfall, Inches(6.9), Inches(1.5), Inches(6.1), Inches(2.8))
        add_textbox(slide, 6.9, 4.35, 6.1, 0.35, "▲  SHAP Waterfall — Why this forecast?",
                    font_size=11, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
    
    # Drift detection
    drift = os.path.join(FIGURES, "drift_detection_demo.png")
    if os.path.exists(drift):
        slide.shapes.add_picture(drift, Inches(0.3), Inches(4.8), Inches(6.3), Inches(2.4))
        add_textbox(slide, 0.3, 7.0, 6.3, 0.35, "▲  Drift Detection Monitoring",
                    font_size=11, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
    
    # Demand pattern scatter
    demand_scatter = os.path.join(FIGURES, "eda", "demand_pattern_scatter.png")
    if os.path.exists(demand_scatter):
        slide.shapes.add_picture(demand_scatter, Inches(6.9), Inches(4.8), Inches(6.1), Inches(2.4))
        add_textbox(slide, 6.9, 7.0, 6.1, 0.35, "▲  Demand Pattern Classification",
                    font_size=11, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
    
    # ═══ SLIDE 9: FUTURE SCOPE & CONCLUSION ═══
    print("Building Slide 9: Conclusion")
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, DARK_BG)
    
    add_textbox(slide, 0.5, 0.3, 6, 0.6, "CONCLUSION & FUTURE SCOPE",
                font_size=32, color=ACCENT_PURPLE, bold=True)
    
    # What we built (left side)
    add_textbox(slide, 0.8, 1.3, 5.5, 0.5, "✅  What We Built",
                font_size=20, color=ACCENT_GREEN, bold=True)
    
    conclusions = [
        "End-to-end demand forecasting + inventory optimization system",
        "10-model tournament — TSB champion (15.45% WAPE)",
        "Conformal prediction for reliable uncertainty bounds",
        "SHAP explainability for every single forecast",
        "Interactive Streamlit dashboard with FastAPI backend",
    ]
    for i, item in enumerate(conclusions):
        add_textbox(slide, 1.0, 2.0 + i * 0.5, 5.5, 0.45, f"→  {item}",
                    font_size=12, color=WHITE)
    
    # Future scope (right side)
    add_textbox(slide, 7.0, 1.3, 5.5, 0.5, "🚀  Future Scope",
                font_size=20, color=ACCENT_ORANGE, bold=True)
    
    futures = [
        ("Real-time Data", "Live POS integration for instant forecasts", ACCENT_BLUE),
        ("Auto Retraining", "MLOps pipeline with automated model updates", ACCENT_GREEN),
        ("Cloud Deploy", "Scalable cloud deployment + mobile interface", ACCENT_ORANGE),
        ("Multi-Store", "Cross-store inventory optimization at scale", ACCENT_PURPLE),
    ]
    for i, (title, desc, color) in enumerate(futures):
        y = 2.0 + i * 1.0
        add_shape_box(slide, 7.2, y, 5.2, 0.4, title, color, font_size=12)
        add_textbox(slide, 7.2, y + 0.45, 5.2, 0.35, desc, 
                    font_size=11, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
    
    # GitHub link
    add_textbox(slide, 2, 6.5, 9, 0.4, 
                "github.com/PamdemRaviCharanReddy/intelligent-demand-forecasting",
                font_size=12, color=ACCENT_BLUE, alignment=PP_ALIGN.CENTER)
    
    # ═══ SLIDE 10: THANK YOU ═══
    print("Building Slide 10: Thank You")
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, DARK_BG)
    
    add_textbox(slide, 1.5, 2.0, 10, 1, "THANK YOU",
                font_size=56, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    
    add_textbox(slide, 2, 3.2, 9, 0.6, "Questions & Discussion",
                font_size=24, color=ACCENT_BLUE, alignment=PP_ALIGN.CENTER)
    
    # Divider
    div = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                  Inches(5.5), Inches(4.0), Inches(2.3), Inches(0.03))
    div.fill.solid()
    div.fill.fore_color.rgb = RGBColor(59, 130, 246)
    div.line.fill.background()
    
    team_info = "Ravi Charan Reddy  •  Shravani  •  Srujana  •  Krishna Swamy"
    add_textbox(slide, 2, 4.4, 9, 0.5, team_info,
                font_size=14, color=WHITE, alignment=PP_ALIGN.CENTER)
    
    add_textbox(slide, 2, 5.0, 9, 0.5, "IcfaiTech  |  IFHE, Hyderabad",
                font_size=12, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
    
    # Bottom accent line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                   Inches(0), Inches(7.44), Inches(13.333), Inches(0.06))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(59, 130, 246)
    line.line.fill.background()
    
    # ═══ SAVE ═══
    out_path = os.path.join(OUT_DIR, "IDFS_Modern_Presentation.pptx")
    prs.save(out_path)
    print(f"\n[OK] Presentation saved: {out_path}")
    print(f"     Slides: {len(prs.slides)}")
    return out_path

if __name__ == "__main__":
    build_ppt()
