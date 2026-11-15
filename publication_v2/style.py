"""One figure style for the paper: journal column widths, Arial-metric sans serif, validated colour-blind-safe colours.
Palettes checked with the dataviz validator (light surface): arms pass all checks; sources pass, with a contrast
warning for the two light slots, which are therefore always direct-labelled or backed by a table."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use('Agg')

OUT = Path(__file__).parent
MM = 1 / 25.4
W1, W2 = 89 * MM, 183 * MM  # single / double column (inches)

# Palette A: Nature Publishing Group (NPG) hues, navy / vermilion for the two arms.
C = {'noH': '#2E4C9E', 'H': '#E64B35', 'age': '#9A9A9A', 'L1': '#4D4D4D', 'L2': '#6F86D6', 'L2b': '#00A087',
     'ink': '#1D1D1F', 'mute': '#6B6B6B', 'grid': '#E6E6E3', 'chance': '#C8C8C8'}
SRC = {'own_num': '#2E4C9E', 'own_maps': '#6F86D6', 'fellow_num': '#00A087', 'fellow_maps': '#7CBF4A',
       'H': '#E64B35', 'age': '#8C8C8C', 'flags': '#BDBDBD'}
SRC_LABEL = {'own_num': 'Own eye, numerics', 'own_maps': 'Own eye, maps', 'fellow_num': 'Fellow-derived, numerics',
             'fellow_maps': 'Fellow-derived, maps', 'age': 'Age', 'flags': 'Map availability',
             'H': 'Fellow treatment history'}
ABCD = {'A': '#2E4C9E', 'B': '#00A087', 'C': '#E64B35', 'other': '#BDBDBD'}
SEQ = LinearSegmentedColormap.from_list('kc_seq', ['#F7F8FB', '#CBD3EC', '#8C9FD6', '#4A64B3', '#1B2C66'])
DIV = 'RdBu_r'
MAP_CMAP = {'T': 'cividis', 'K': 'magma', 'EF': 'RdBu_r', 'EB': 'RdBu_r'}


def apply():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'DejaVu Sans'],
        'font.size': 7.5, 'axes.titlesize': 8.5, 'axes.titleweight': 'bold', 'axes.labelsize': 7.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'axes.edgecolor': '#555555',
        'axes.labelcolor': C['ink'], 'text.color': C['ink'], 'xtick.color': '#555555', 'ytick.color': '#555555',
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.grid': False, 'grid.color': C['grid'], 'grid.linewidth': 0.6, 'lines.linewidth': 1.6,
        'savefig.dpi': 300, 'figure.dpi': 150, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.titlelocation': 'left', 'axes.titlepad': 6})


def panel_letter(ax, s, x=-0.14, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left')


def save(fig, name, local=False):
    """figures/{name}.pdf + .png (300 dpi), or local/ for individual-level figures (gitignored). Returns the PNG."""
    d = OUT / ('local' if local else 'figures')
    d.mkdir(exist_ok=True)
    fig.savefig(d / f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate': None})  # reproducible bytes
    fig.savefig(d / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return d / f'{name}.png'
