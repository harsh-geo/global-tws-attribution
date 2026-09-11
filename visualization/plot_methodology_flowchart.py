"""
plot_methodology_flowchart.py
==============================
Generates Figure 3: High-resolution publication flowchart diagram of the
Twin Machine Learning Attribution Framework for Terrestrial Water Storage (TWS).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_flowchart():
    # Setup Figure (15 x 8.4 inches, 300 DPI)
    fig = plt.figure(figsize=(15.0, 8.4), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 8.4)
    ax.axis('off')
    
    # -------------------------------------------------------------
    # Typography & Styling
    # -------------------------------------------------------------
    FONT_TITLE = {'family': 'sans-serif', 'weight': 'bold', 'size': 11}
    FONT_HEAD = {'family': 'sans-serif', 'weight': 'bold', 'size': 9.5}
    FONT_BODY_BOLD = {'family': 'sans-serif', 'weight': 'bold', 'size': 8.5}
    FONT_BODY = {'family': 'sans-serif', 'weight': 'normal', 'size': 8.2}
    FONT_SUB = {'family': 'sans-serif', 'weight': 'normal', 'size': 7.6}
    FONT_MATH = {'family': 'sans-serif', 'weight': 'bold', 'size': 8.8}

    # -------------------------------------------------------------
    # 1. Background Category Cards (Subgraphs)
    # -------------------------------------------------------------
    # Container 1: Inputs
    c1 = FancyBboxPatch((0.4, 0.4), 3.5, 7.5, boxstyle="round,pad=0.1,rounding_size=0.2",
                        facecolor='#F4F8FC', edgecolor='#B9D5ED', linewidth=1.4, zorder=1)
    ax.add_patch(c1)
    ax.text(2.15, 7.62, "Input Flux Predictors", ha='center', va='center', **FONT_TITLE, color='#1B4965')
    ax.text(2.15, 7.34, "(0.5° Monthly, Standardized to cm/month)", ha='center', va='center', **FONT_SUB, color='#3B6785')

    # Container 2: Twin ML Models
    c2 = FancyBboxPatch((4.3, 0.4), 4.8, 7.5, boxstyle="round,pad=0.1,rounding_size=0.2",
                        facecolor='#FFFDF8', edgecolor='#FCD8B4', linewidth=1.4, zorder=1)
    ax.add_patch(c2)
    ax.text(6.7, 7.62, "Twin Machine Learning Framework", ha='center', va='center', **FONT_TITLE, color='#8C4300')
    ax.text(6.7, 7.34, r"(Target: Storage Derivative $TWSC = dTWS/dt$, Deseasonalized)", ha='center', va='center', **FONT_SUB, color='#A85D14')

    # Container 3: Diagnostics & Attribution
    c3 = FancyBboxPatch((9.5, 0.4), 5.1, 7.5, boxstyle="round,pad=0.1,rounding_size=0.2",
                        facecolor='#F5FAF6', edgecolor='#BDE3C8', linewidth=1.4, zorder=1)
    ax.add_patch(c3)
    ax.text(12.05, 7.62, "Attribution Metrics & Causal Diagnostics", ha='center', va='center', **FONT_TITLE, color='#1E5631')
    ax.text(12.05, 7.34, "(Generalization Gain, Collinearity Isolation & Directionality)", ha='center', va='center', **FONT_SUB, color='#2F7344')

    # -------------------------------------------------------------
    # 2. Left Column: Input Variable Boxes
    # -------------------------------------------------------------
    inputs = [
        ("Precipitation (P)", "ERA5 Reanalysis (m/mo -> cm/mo)", '#E6F0FA', '#357ABD', 6.25),
        ("Evapotranspiration (ET)", "GLEAM v3.x (mm/mo -> cm/mo, Flipped)", '#E6F0FA', '#357ABD', 5.00),
        ("Runoff / Discharge (Q)", "ERA5 Reanalysis (m/mo -> cm/mo)", '#E6F0FA', '#357ABD', 3.75),
        ("Groundwater Abs. (GW_abs)", "PCR-GLOBWB 2.0 (m/mo -> cm/mo)", '#FFF2E6', '#E67E22', 2.50),
        ("Surface Water Abs. (SW_abs)", "PCR-GLOBWB 2.0 (m/mo -> cm/mo)", '#FFF2E6', '#E67E22', 1.25)
    ]
    
    input_ports = []
    for title, sub, bg, border, y in inputs:
        box = FancyBboxPatch((0.6, y - 0.45), 3.1, 0.9, boxstyle="round,pad=0.08,rounding_size=0.12",
                             facecolor=bg, edgecolor=border, linewidth=1.2, zorder=2)
        ax.add_patch(box)
        ax.text(2.15, y + 0.13, title, ha='center', va='center', **FONT_BODY_BOLD, color='#1A2B4C')
        ax.text(2.15, y - 0.18, sub, ha='center', va='center', **FONT_SUB, color='#4A5B70')
        input_ports.append((3.7, y))

    # -------------------------------------------------------------
    # 3. Middle Column: Twin ML Model Cards
    # -------------------------------------------------------------
    # Model 1: Natural Baseline M_nat
    m1_box = FancyBboxPatch((4.5, 4.0), 4.4, 3.1, boxstyle="round,pad=0.08,rounding_size=0.15",
                            facecolor='#FFFFFF', edgecolor='#2B7BBA', linewidth=1.8, zorder=2)
    ax.add_patch(m1_box)
    ax.text(6.7, 6.78, "Natural Baseline Model (M_nat)", ha='center', va='center', **FONT_HEAD, color='#164268')
    ax.text(6.7, 6.45, r"$TWSC_{pred} = f_{nat}( P, ET, Q )$", ha='center', va='center', **FONT_MATH, color='#2B7BBA')
    
    m1_desc = (
        "• Regressor: Random Forest Ensemble (500 Trees)\n"
        "• Constrained Subspace: NumPredictorsToSample = 1\n"
        "• Physics Alignment: Finite Centered Difference dTWS/dt\n"
        "• Evaluation: 3-Year Contiguous Block CV\n"
        "• Purpose: Establishes Climate-Forced Baseline"
    )
    ax.text(4.7, 5.25, m1_desc, ha='left', va='center', **FONT_SUB, color='#2C3E50', linespacing=1.45)

    # Model 2: Full Anthropogenic M_anthro
    m2_box = FancyBboxPatch((4.5, 0.7), 4.4, 2.95, boxstyle="round,pad=0.08,rounding_size=0.15",
                            facecolor='#FFFFFF', edgecolor='#D9531E', linewidth=1.8, zorder=2)
    ax.add_patch(m2_box)
    ax.text(6.7, 3.32, "Full Anthropogenic Model (M_anthro)", ha='center', va='center', **FONT_HEAD, color='#8A2C08')
    ax.text(6.7, 3.00, r"$TWSC_{pred} = f_{anthro}( P, ET, Q, GW_{abs}, SW_{abs} )$", ha='center', va='center', **FONT_MATH, color='#D9531E')
    
    m2_desc = (
        "• Regressor: Random Forest Ensemble (500 Trees)\n"
        "• Constrained Subspace: NumPredictorsToSample = 1\n"
        "• Driver Integration: Human Abstraction Sinks\n"
        "• Importance: OOB Permuted Predictor Delta Error\n"
        "• Deep Benchmark: Parallel Twin LSTM Network"
    )
    ax.text(4.7, 1.88, m2_desc, ha='left', va='center', **FONT_SUB, color='#2C3E50', linespacing=1.45)

    # -------------------------------------------------------------
    # 4. Right Column: Diagnostics & Attribution Boxes
    # -------------------------------------------------------------
    eval_boxes = [
        ("Variance Explained Gain (ΔR²)",
         r"$\Delta R^2 = R^2_{anthro} - R^2_{nat}$" + "\n"
         "Isolates variance gain across 103 river basins\n"
         "(78 basins show positive human explanatory gain)",
         '#E8F5E9', '#388E3C', 6.25),
        
        ("3-Year Block CV Gain (ΔNSE)",
         r"$\Delta NSE = NSE_{anthro} - NSE_{nat}$" + "\n"
         "Strictly prevents temporal autocorrelation leakage\n"
         "(Mean NSE gain: +25.2%, 67/103 basins improved)",
         '#E8F5E9', '#388E3C', 4.70),
        
        ("Event-Level Causal SHAP Values",
         "Shapley Additive exPlanations (Game Theory)\n"
         "Resolves monthly directional forcing;\n"
         "High abstraction deterministically drives negative TWSC",
         '#F3E5F5', '#7B1FA2', 3.05),
         
        ("Spatial Transferability Test",
         "Trained on 52 Pristine basins (No Irrigation)\n"
         "Evaluated out-of-domain on 51 Irrigated basins;\n"
         "Positive prediction bias proves missing human sink",
         '#FFF8E1', '#F57C00', 1.35)
    ]
    
    eval_inputs = []
    for title, text, bg, border, y in eval_boxes:
        box = FancyBboxPatch((9.7, y - 0.58), 4.7, 1.15, boxstyle="round,pad=0.08,rounding_size=0.12",
                             facecolor=bg, edgecolor=border, linewidth=1.3, zorder=2)
        ax.add_patch(box)
        ax.text(12.05, y + 0.35, title, ha='center', va='center', **FONT_BODY_BOLD, color='#1F2D3D')
        ax.text(12.05, y - 0.15, text, ha='center', va='center', **FONT_SUB, color='#34495E', linespacing=1.3)
        eval_inputs.append((9.7, y))

    # -------------------------------------------------------------
    # 5. Clean, Professional Connective Arrows
    # -------------------------------------------------------------
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=13, lw=1.3)
    
    # Natural Inputs (P, ET, Q) -> M_nat
    for i, target_y in enumerate([5.8, 5.3, 4.8]):
        sx, sy = input_ports[i]
        ax.annotate("", xy=(4.5, target_y), xytext=(sx, sy),
                    arrowprops=dict(**arrow_kw, color='#2B7BBA', alpha=0.85,
                                    connectionstyle="arc3,rad=-0.04"), zorder=3)
        
    # All 5 Inputs -> M_anthro
    anthro_targets = [2.6, 2.3, 2.0, 1.6, 1.2]
    for i in range(5):
        sx, sy = input_ports[i]
        c = '#D9531E' if i >= 3 else '#7FB3D5'
        rad = 0.07 if i < 3 else -0.04
        ax.annotate("", xy=(4.5, anthro_targets[i]), xytext=(sx, sy),
                    arrowprops=dict(**arrow_kw, color=c, alpha=0.85,
                                    connectionstyle=f"arc3,rad={rad}"), zorder=3)

    # M_nat & M_anthro -> Delta R² (eval_inputs[0])
    ax.annotate("", xy=(9.7, 6.4), xytext=(8.9, 5.8),
                arrowprops=dict(**arrow_kw, color='#2B7BBA', alpha=0.9,
                                connectionstyle="arc3,rad=-0.08"), zorder=3)
    ax.annotate("", xy=(9.7, 6.1), xytext=(8.9, 2.6),
                arrowprops=dict(**arrow_kw, color='#D9531E', alpha=0.9,
                                connectionstyle="arc3,rad=0.12"), zorder=3)

    # M_nat & M_anthro -> Delta NSE (eval_inputs[1])
    ax.annotate("", xy=(9.7, 4.85), xytext=(8.9, 5.2),
                arrowprops=dict(**arrow_kw, color='#2B7BBA', alpha=0.9,
                                connectionstyle="arc3,rad=-0.04"), zorder=3)
    ax.annotate("", xy=(9.7, 4.55), xytext=(8.9, 2.2),
                arrowprops=dict(**arrow_kw, color='#D9531E', alpha=0.9,
                                connectionstyle="arc3,rad=0.08"), zorder=3)

    # M_anthro -> SHAP (eval_inputs[2])
    ax.annotate("", xy=(9.7, 3.05), xytext=(8.9, 1.8),
                arrowprops=dict(**arrow_kw, color='#7B1FA2', alpha=0.9,
                                connectionstyle="arc3,rad=0.04"), zorder=3)

    # M_nat (Pristine) -> Spatial Transferability (eval_inputs[3])
    ax.annotate("", xy=(9.7, 1.35), xytext=(8.9, 4.3),
                arrowprops=dict(**arrow_kw, color='#F57C00', alpha=0.9, linestyle='--',
                                connectionstyle="arc3,rad=0.22"), zorder=3)
    
    # Badge on pristine transfer arrow
    ax.text(9.28, 3.2, "Pristine Basin Training", ha='center', va='center',
            fontsize=7.2, fontweight='bold', color='#B45309', rotation=-65,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FEF3C7', edgecolor='#FCD34D', alpha=0.95))

    # Save outputs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    out_dir = os.path.join(project_root, 'outputs', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    
    out_png = os.path.join(out_dir, 'methodology_flowchart.png')
    out_pdf = os.path.join(out_dir, 'methodology_flowchart.pdf')
    
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[SUCCESS] Methodology flowchart saved to:\n  PNG: {out_png}\n  PDF: {out_pdf}")

if __name__ == '__main__':
    create_flowchart()
