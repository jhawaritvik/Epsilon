"""
Generate publication-quality diagrams for the Epsilon research paper.
Outputs to paper/overleaf_assets/

Clean, professional academic style with white background for print/PDF.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path
import matplotlib.patheffects as pe
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'overleaf_assets')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────
COLORS = {
    'bg':           '#FFFFFF',
    'controller':   '#1565C0',   # deep blue
    'research':     '#00838F',   # teal
    'design':       '#5E35B1',   # deep purple
    'execution':    '#2E7D32',   # green
    'evaluation':   '#C62828',   # deep red
    'memory_ev':    '#0288D1',   # light blue
    'memory_kn':    '#388E3C',   # green
    'memory_run':   '#F57C00',   # orange
    'text_light':   '#FFFFFF',
    'text_dark':    '#212121',
    'text_muted':   '#616161',
    'arrow':        '#455A64',
    'arrow_light':  '#90A4AE',
    'border':       '#B0BEC5',
    'success':      '#2E7D32',
    'fail':         '#D32F2F',
    'feedback':     '#E65100',
    'panel_bg':     '#F5F5F5',
    'panel_border': '#E0E0E0',
    'lock_bg':      '#FFCDD2',
    'lock_border':  '#EF9A9A',
    'supabase':     '#37474F',
}

# ── Font config ───────────────────────────────────────────────
FONT_TITLE = {'fontsize': 18, 'fontweight': 'bold', 'fontfamily': 'sans-serif'}
FONT_SUBTITLE = {'fontsize': 11, 'fontfamily': 'sans-serif', 'fontstyle': 'italic'}
FONT_BOX = {'fontsize': 12, 'fontweight': 'bold', 'fontfamily': 'sans-serif'}
FONT_SUB = {'fontsize': 9, 'fontfamily': 'sans-serif'}
FONT_LABEL = {'fontsize': 9, 'fontfamily': 'sans-serif', 'fontstyle': 'italic'}
FONT_SMALL = {'fontsize': 8, 'fontfamily': 'sans-serif'}

# ── Helpers ───────────────────────────────────────────────────
def _shadow():
    return [pe.withSimplePatchShadow(offset=(1.5, -1.5), shadow_rgbFace='#00000018')]

def draw_box(ax, x, y, w, h, label, sublabel=None, color='#1565C0',
             text_color='white', fontsize=12, sub_fontsize=9,
             border_style='-', border_color=None, border_width=1.8,
             alpha=1.0, radius=0.12, shadow=True, zorder=2):
    """Draw a rounded rectangle with centered text."""
    if border_color is None:
        border_color = color
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=color, edgecolor=border_color,
        linewidth=border_width, linestyle=border_style,
        alpha=alpha, zorder=zorder
    )
    if shadow:
        box.set_path_effects(_shadow())
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + h*0.15, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color,
                fontfamily='sans-serif', zorder=zorder+1)
        ax.text(x, y - h*0.2, sublabel, ha='center', va='center',
                fontsize=sub_fontsize, color=text_color, alpha=0.9,
                fontfamily='sans-serif', zorder=zorder+1)
    else:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color,
                fontfamily='sans-serif', zorder=zorder+1)

def draw_arrow(ax, x1, y1, x2, y2, color='#455A64', style='-|>',
               linewidth=1.8, connectionstyle='arc3,rad=0',
               label=None, label_color=None, label_fontsize=9,
               label_offset=(0.08, 0.0), zorder=1, linestyle='-'):
    """Draw an arrow between two points with optional label."""
    if label_color is None:
        label_color = color
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, linewidth=linewidth,
        connectionstyle=connectionstyle, zorder=zorder,
        mutation_scale=16, linestyle=linestyle
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=label_fontsize, color=label_color,
                ha='center', va='center', zorder=zorder+1,
                fontfamily='sans-serif', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                         edgecolor='none', alpha=0.85))

def draw_lock_icon(ax, x, y, size=0.18, color='#C62828'):
    """Draw a simple lock icon using matplotlib shapes."""
    # Lock body
    bw, bh = size * 1.2, size * 0.9
    body = FancyBboxPatch(
        (x - bw/2, y - bh/2 - size*0.15), bw, bh,
        boxstyle="round,pad=0.03",
        facecolor=color, edgecolor='#B71C1C',
        linewidth=1.0, zorder=5
    )
    ax.add_patch(body)
    # Lock shackle (arc)
    theta = np.linspace(0, np.pi, 30)
    r = size * 0.45
    xs = x + r * np.cos(theta)
    ys = y + size*0.2 + r * np.sin(theta)
    ax.plot(xs, ys, color='#B71C1C', linewidth=2.5, solid_capstyle='round', zorder=5)
    # Keyhole
    ax.plot(x, y - size*0.15, marker='o', markersize=3, color='white', zorder=6)

def setup_axes(figsize, xlim, ylim):
    """Create a figure with white background and clean axes."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


# ============================================================
# FIGURE 1: Agent Architecture
# ============================================================
def generate_agent_architecture():
    fig, ax = setup_axes((13, 9.5), (-4, 9), (-1.5, 8.2))
    
    # Title
    ax.text(2.5, 7.8, 'Epsilon Agent Architecture', color=COLORS['text_dark'],
            ha='center', va='center', **FONT_TITLE)
    
    # ─── Controller ───────────────────────────────────────
    draw_box(ax, 2.5, 6.6, 6.0, 0.85, 'Research Controller',
             'Central Orchestrator', COLORS['controller'], fontsize=13)
    
    # ─── Agent Row ────────────────────────────────────────
    agents = [
        (-0.5, 4.8, 'Research Agent', 'Literature Review', COLORS['research']),
        ( 2.5, 4.8, 'Design Agent',   'Hypothesis',        COLORS['design']),
        ( 5.5, 4.8, 'Execution Agent', 'Code & Run',       COLORS['execution']),
    ]
    for x, y, name, sub, col in agents:
        draw_box(ax, x, y, 2.6, 0.85, name, sub, col)
    
    # ─── Evaluation (isolated, dashed border) ─────────────
    # Isolation background panel
    iso_panel = FancyBboxPatch(
        (0.6, 1.6), 3.8, 1.75,
        boxstyle="round,pad=0.2",
        facecolor=COLORS['lock_bg'], edgecolor=COLORS['lock_border'],
        linewidth=1.5, linestyle='--', alpha=0.4, zorder=1
    )
    ax.add_patch(iso_panel)
    ax.text(4.2, 3.2, 'ISOLATED', fontsize=8, fontweight='bold',
            color=COLORS['evaluation'], alpha=0.7, ha='center', va='center',
            fontfamily='sans-serif')
    
    draw_box(ax, 2.5, 2.4, 3.2, 0.95, 'Evaluation Agent',
             'Statistical Validation', COLORS['evaluation'], border_width=2.2)
    draw_lock_icon(ax, 4.35, 2.4, size=0.2)
    
    # ─── Arrows: Controller → Agents ─────────────────────
    for ax_pos in [-0.5, 2.5, 5.5]:
        draw_arrow(ax, ax_pos, 6.15, ax_pos, 5.27, COLORS['arrow'], linewidth=2)
    
    # ─── Arrows: Agent flow (Research → Design → Execution) ──
    draw_arrow(ax, 0.85, 4.8, 1.15, 4.8, COLORS['arrow'], linewidth=1.8)
    draw_arrow(ax, 3.85, 4.8, 4.15, 4.8, COLORS['arrow'], linewidth=1.8)
    
    # Execution → Evaluation
    draw_arrow(ax, 4.7, 4.35, 3.5, 2.92, COLORS['arrow'], linewidth=1.8,
               label='results', label_color=COLORS['text_muted'],
               label_offset=(0.35, 0.15))
    
    # Self-loop on Execution Agent (retry cycle)
    self_loop = FancyArrowPatch(
        (6.8, 5.05), (6.8, 4.55),
        arrowstyle='->', color=COLORS['execution'], linewidth=1.6,
        connectionstyle='arc3,rad=-1.6', mutation_scale=13, zorder=3
    )
    ax.add_patch(self_loop)
    ax.text(7.6, 4.8, 'self-correction\n(≤3 retries)', fontsize=7.5,
            color=COLORS['execution'], ha='left', va='center',
            fontfamily='sans-serif', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                     edgecolor=COLORS['execution'], alpha=0.7, linewidth=0.8))
    
    # Feedback: Evaluation → Design (curved)
    draw_arrow(ax, 1.2, 2.6, 1.5, 4.35, COLORS['feedback'], linewidth=2,
               connectionstyle='arc3,rad=-0.35',
               label='feedback\n(if failed)', label_color=COLORS['fail'],
               label_offset=(-0.7, 0.0))
    
    # Success → Final Report
    draw_arrow(ax, 2.5, 1.9, 2.5, 0.55, COLORS['success'], linewidth=2,
               label='pass', label_color=COLORS['success'],
               label_offset=(0.35, 0.0))
    
    # ─── Final Report ─────────────────────────────────────
    draw_box(ax, 2.5, 0.0, 2.5, 0.7, 'Final Report', None,
             '#546E7A', fontsize=11)
    
    # ─── Memory Sidebar ──────────────────────────────────
    mem_panel = FancyBboxPatch(
        (-3.8, 0.8), 2.6, 3.8,
        boxstyle="round,pad=0.15",
        facecolor=COLORS['panel_bg'], edgecolor=COLORS['panel_border'],
        linewidth=1.2, zorder=0
    )
    ax.add_patch(mem_panel)
    
    ax.text(-2.5, 4.2, 'Memory System', fontsize=11, fontweight='bold',
            color=COLORS['text_dark'], ha='center', va='center',
            fontfamily='sans-serif')
    
    mem_tiers = [
        (-2.5, 3.45, 'Evidence Memory',   COLORS['memory_ev'],  'white'),
        (-2.5, 2.55, 'Knowledge Memory',   COLORS['memory_kn'],  'white'),
        (-2.5, 1.65, 'Run Memory',         COLORS['memory_run'], 'white'),
    ]
    for x, y, label, col, tc in mem_tiers:
        draw_box(ax, x, y, 2.2, 0.6, label, None, col,
                 text_color=tc, fontsize=9.5, shadow=False)
    
    # Scope labels (inside boxes, second line)
    scopes = [
        (-2.5, 3.22, 'per-run',       COLORS['bg']),
        (-2.5, 2.32, 'cross-run',     COLORS['bg']),
        (-2.5, 1.42, 'per-iteration', COLORS['bg']),
    ]
    for x, y, label, col in scopes:
        ax.text(x, y, label, fontsize=7, color=col, ha='center', va='center',
                fontfamily='sans-serif', fontstyle='italic', alpha=0.85)
    
    # Dashed connector to research agent
    draw_arrow(ax, -1.35, 3.6, -0.9, 4.35, COLORS['arrow_light'],
               linewidth=1.2, linestyle=':', style='-')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'agent_architecture.png'),
                dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("✅ agent_architecture.png generated")


# ============================================================
# FIGURE 2: The Epistemic Loop
# ============================================================
def generate_epistemic_loop():
    fig, ax = setup_axes((11, 8), (-1.5, 10), (-1, 7))
    
    # Title
    ax.text(4.25, 6.5, 'The Epistemic Loop', color=COLORS['text_dark'],
            ha='center', va='center', **FONT_TITLE)
    ax.text(4.25, 6.05, 'Self-Correcting Research Workflow', color=COLORS['text_muted'],
            ha='center', va='center', **FONT_SUBTITLE)
    
    # ─── Stage boxes ──────────────────────────────────────
    bw, bh = 2.2, 0.9
    
    # Top row: Research → Design → Execute
    draw_box(ax, 0.5, 4.6, bw, bh, '1. Research', 'Context Gathering', COLORS['research'])
    draw_box(ax, 4.25, 4.6, bw, bh, '2. Design', 'Hypothesis', COLORS['design'])
    draw_box(ax, 8.0, 4.6, bw, bh, '3. Execute', 'Code & Run', COLORS['execution'])
    
    # Bottom right: Evaluate (isolated)
    iso_panel2 = FancyBboxPatch(
        (6.55, 1.3), 2.9, 1.6,
        boxstyle="round,pad=0.15",
        facecolor=COLORS['lock_bg'], edgecolor=COLORS['lock_border'],
        linewidth=1.2, linestyle='--', alpha=0.35, zorder=0
    )
    ax.add_patch(iso_panel2)
    ax.text(9.2, 2.75, 'ISOLATED', fontsize=7, fontweight='bold',
            color=COLORS['evaluation'], alpha=0.6, ha='center', va='center',
            fontfamily='sans-serif')
    
    draw_box(ax, 8.0, 2.1, bw, bh, '4. Evaluate', 'Stat. Validation', COLORS['evaluation'])
    draw_lock_icon(ax, 9.3, 2.1, size=0.17)
    
    # Bottom left: Feedback
    draw_box(ax, 0.5, 2.1, bw, bh, 'Feedback', 'Revision Directives', COLORS['feedback'])
    
    # Bottom center: Final Report
    draw_box(ax, 4.25, 0.0, 2.4, 0.7, 'Final Report', None, '#546E7A', fontsize=11)
    
    # ─── Arrows ───────────────────────────────────────────
    # Top row flow
    draw_arrow(ax, 1.65, 4.6, 3.1, 4.6, COLORS['arrow'], linewidth=2.2)
    draw_arrow(ax, 5.4, 4.6, 6.85, 4.6, COLORS['arrow'], linewidth=2.2)
    
    # Execute → Evaluate (down)
    draw_arrow(ax, 8.0, 4.1, 8.0, 2.6, COLORS['arrow'], linewidth=2.2)
    
    # Evaluate → Feedback (fail path)
    draw_arrow(ax, 6.85, 2.1, 1.65, 2.1, COLORS['fail'], linewidth=2.2,
               label='Failed', label_color=COLORS['fail'], label_fontsize=10)
    
    # Feedback → Design (loop back up, curved)
    draw_arrow(ax, 0.5, 2.6, 3.8, 4.1, COLORS['feedback'], linewidth=2,
               connectionstyle='arc3,rad=-0.2',
               label='iterate', label_color=COLORS['feedback'],
               label_offset=(-0.1, 0.2))
    
    # Evaluate → Report (success path)
    draw_arrow(ax, 7.2, 1.6, 5.5, 0.4, COLORS['success'], linewidth=2.2,
               label='Passed', label_color=COLORS['success'], label_fontsize=10,
               label_offset=(0.15, 0.15))
    
    # Iteration badge
    badge = FancyBboxPatch(
        (-1.1, 3.0), 1.6, 0.55,
        boxstyle="round,pad=0.1",
        facecolor=COLORS['panel_bg'], edgecolor=COLORS['panel_border'],
        linewidth=1.0, zorder=2
    )
    ax.add_patch(badge)
    ax.text(-0.3, 3.27, 'Iteration i / N', fontsize=9, color=COLORS['text_muted'],
            ha='center', va='center', fontfamily='sans-serif', fontstyle='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'epistemic_loop.png'),
                dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("✅ epistemic_loop.png generated")


# ============================================================
# FIGURE 3: Memory Architecture
# ============================================================
def generate_memory_architecture():
    fig, ax = setup_axes((11, 7), (-1.5, 10), (-0.5, 6))
    
    # Title
    ax.text(4.25, 5.6, 'Three-Tier Memory Architecture', color=COLORS['text_dark'],
            ha='center', va='center', **FONT_TITLE)
    
    # ─── Tier boxes ───────────────────────────────────────
    tier_w, tier_h = 5.5, 0.9
    cx = 3.5
    
    # Evidence Memory
    draw_box(ax, cx, 4.3, tier_w, tier_h, 'Evidence Memory',
             'Raw findings  |  Read-before-write dedup',
             COLORS['memory_ev'], fontsize=13, sub_fontsize=10)
    
    # Arrow + crystallization label
    ax.annotate('', xy=(cx, 3.45), xytext=(cx, 3.75),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['arrow'], lw=1.5))
    ax.text(cx, 3.55, 'Crystallization Filter', fontsize=9,
            color=COLORS['text_muted'], ha='center', va='center',
            fontfamily='sans-serif', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#E8EAF6',
                     edgecolor='#C5CAE9', linewidth=0.8))
    
    # Knowledge Memory
    draw_box(ax, cx, 2.7, tier_w, tier_h, 'Knowledge Memory',
             'Validated conclusions  |  Append-only',
             COLORS['memory_kn'], fontsize=13, sub_fontsize=10)
    
    # Separator
    ax.plot([cx - tier_w/2 + 0.3, cx + tier_w/2 - 0.3], [1.95, 1.95],
            color=COLORS['panel_border'], linewidth=1, linestyle='--', alpha=0.5)
    
    # Run Memory
    draw_box(ax, cx, 1.2, tier_w, tier_h, 'Run Memory',
             'Complete audit trail  |  Query by failure type',
             COLORS['memory_run'], fontsize=13, sub_fontsize=10)
    
    # ─── Scope labels (left side) ─────────────────────────
    scope_x = cx - tier_w/2 - 0.4
    scopes = [
        (scope_x, 4.3, 'Per-run',       COLORS['memory_ev'],  '●'),
        (scope_x, 2.7, 'Cross-run',     COLORS['memory_kn'],  '●'),
        (scope_x, 1.2, 'Per-iteration', COLORS['memory_run'], '●'),
    ]
    for x, y, label, col, marker in scopes:
        ax.text(x, y, f'{marker} {label}', fontsize=9, color=col,
                ha='right', va='center', fontfamily='sans-serif',
                fontweight='bold')
    
    # ─── Backend box (right side) ─────────────────────────
    be_x = 8.2
    # Panel
    be_panel = FancyBboxPatch(
        (be_x - 1.0, 0.9), 2.0, 3.7,
        boxstyle="round,pad=0.15",
        facecolor=COLORS['panel_bg'], edgecolor=COLORS['panel_border'],
        linewidth=1.2, zorder=0
    )
    ax.add_patch(be_panel)
    
    draw_box(ax, be_x, 2.7, 1.6, 0.7, 'Supabase', 'PostgreSQL',
             COLORS['supabase'], fontsize=11, sub_fontsize=8, shadow=False)
    
    # Connection lines
    for tier_y in [4.3, 2.7, 1.2]:
        draw_arrow(ax, cx + tier_w/2 + 0.05, tier_y, be_x - 0.85, 2.7,
                   COLORS['arrow_light'], linewidth=1.2, linestyle=':',
                   style='-')
    
    # ─── Data flow annotations ────────────────────────────
    ax.text(be_x, 4.4, 'Persistent\nStorage', fontsize=8,
            color=COLORS['text_muted'], ha='center', va='center',
            fontfamily='sans-serif', fontstyle='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'memory_architecture.png'),
                dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("✅ memory_architecture.png generated")


# ============================================================
# FIGURE 4: Threat Model
# ============================================================
# Threat model is now a native LaTeX table in the paper,
# no image generation needed.

def _generate_threat_model_removed():
    """Removed — threat model is now a LaTeX table."""
    fig, ax = setup_axes((11, 6), (-0.5, 10.5), (-1, 5.5))
    
    # Title
    ax.text(5, 5.1, 'Epistemic Threat Model', color=COLORS['text_dark'],
            ha='center', va='center', **FONT_TITLE)
    
    # ─── Left panel: Mitigated ────────────────────────────
    left_panel = FancyBboxPatch(
        (0, 0.2), 4.5, 4.2,
        boxstyle="round,pad=0.2",
        facecolor='#E8F5E9', edgecolor='#A5D6A7',
        linewidth=1.5, zorder=0
    )
    ax.add_patch(left_panel)
    
    ax.text(2.25, 4.0, 'MITIGATED', fontsize=13, fontweight='bold',
            color=COLORS['success'], ha='center', va='center',
            fontfamily='sans-serif')
    ax.plot([0.5, 4.0], [3.65, 3.65], color='#A5D6A7', linewidth=1)
    
    mitigated = [
        ('Metric switching',      'Frozen at design time'),
        ('Hypothesis rewriting',  'Append-only constraint'),
        ('Optional stopping',     'Fixed iteration plan'),
        ('Selective reporting',   'All results recorded'),
    ]
    for i, (threat, mechanism) in enumerate(mitigated):
        y = 3.2 - i * 0.75
        # Checkmark badge
        badge_bg = FancyBboxPatch(
            (0.35, y - 0.28), 4.0, 0.56,
            boxstyle="round,pad=0.08",
            facecolor='white', edgecolor='#C8E6C9',
            linewidth=1, alpha=0.8, zorder=1
        )
        ax.add_patch(badge_bg)
        ax.text(0.65, y, '✓', fontsize=14, fontweight='bold',
                color=COLORS['success'], ha='center', va='center',
                fontfamily='sans-serif', zorder=2)
        ax.text(1.0, y + 0.08, threat, fontsize=10, fontweight='bold',
                color=COLORS['text_dark'], ha='left', va='center',
                fontfamily='sans-serif', zorder=2)
        ax.text(1.0, y - 0.16, mechanism, fontsize=8,
                color=COLORS['text_muted'], ha='left', va='center',
                fontfamily='sans-serif', fontstyle='italic', zorder=2)
    
    # ─── Right panel: Not Mitigated ───────────────────────
    right_panel = FancyBboxPatch(
        (5.5, 0.2), 4.5, 4.2,
        boxstyle="round,pad=0.2",
        facecolor='#FBE9E7', edgecolor='#FFAB91',
        linewidth=1.5, zorder=0
    )
    ax.add_patch(right_panel)
    
    ax.text(7.75, 4.0, 'NOT MITIGATED', fontsize=13, fontweight='bold',
            color=COLORS['fail'], ha='center', va='center',
            fontfamily='sans-serif')
    ax.plot([6.0, 9.5], [3.65, 3.65], color='#FFAB91', linewidth=1)
    
    not_mitigated = [
        ('Poor hypothesis quality', 'Depends on LLM reasoning'),
        ('Dataset bias',            'Pre-existing data issue'),
        ('Adversarial injection',   'Prompt-level attack vector'),
    ]
    for i, (threat, reason) in enumerate(not_mitigated):
        y = 3.2 - i * 0.75
        badge_bg2 = FancyBboxPatch(
            (5.85, y - 0.28), 4.0, 0.56,
            boxstyle="round,pad=0.08",
            facecolor='white', edgecolor='#FFCDD2',
            linewidth=1, alpha=0.8, zorder=1
        )
        ax.add_patch(badge_bg2)
        ax.text(6.15, y, '✗', fontsize=14, fontweight='bold',
                color=COLORS['fail'], ha='center', va='center',
                fontfamily='sans-serif', zorder=2)
        ax.text(6.5, y + 0.08, threat, fontsize=10, fontweight='bold',
                color=COLORS['text_dark'], ha='left', va='center',
                fontfamily='sans-serif', zorder=2)
        ax.text(6.5, y - 0.16, reason, fontsize=8,
                color=COLORS['text_muted'], ha='left', va='center',
                fontfamily='sans-serif', fontstyle='italic', zorder=2)
    
    # ─── Divider ──────────────────────────────────────────
    ax.text(5.0, 2.5, 'VS', fontsize=11, fontweight='bold',
            color=COLORS['border'], ha='center', va='center',
            fontfamily='sans-serif')
    
    # ─── Footer ───────────────────────────────────────────
    ax.text(5.0, -0.5, 'Epsilon enforces structural constraints, not behavioral outcomes.',
            fontsize=10, color=COLORS['text_muted'], ha='center', va='center',
            fontfamily='sans-serif', fontstyle='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'threat_model.png'),
                dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("✅ threat_model.png generated")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Generating paper diagrams...")
    generate_agent_architecture()
    generate_epistemic_loop()
    generate_memory_architecture()
    # generate_threat_model() — now a native LaTeX table
    print(f"\nAll diagrams saved to: {OUTPUT_DIR}")
    print("Files:")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
            print(f"  {f} ({size/1024:.1f} KB)")
