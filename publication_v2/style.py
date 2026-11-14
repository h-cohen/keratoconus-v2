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

C = {'noH': '#2A6FB0', 'H': '#C77700', 'age': '#9A9A9A', 'L1': '#4D4D4D', 'L2': '#7A9CC6', 'L2b': '#7FB7A4',
     'ink': '#1D1D1F', 'mute': '#6B6B6B', 'grid': '#E6E6E3', 'chance': '#C8C8C8'}
SRC = {'own_num': '#1F6FB4', 'own_maps': '#5BA8DE', 'fellow_num': '#C2561A', 'fellow_maps': '#E3A33B',
       'H': '#8E5BB5', 'age': '#8C8C8C', 'flags': '#BDBDBD'}
SRC_LABEL = {'own_num': 'Own eye, numerics', 'own_maps': 'Own eye, maps', 'fellow_num': 'Fellow-derived, numerics',
             'fellow_maps': 'Fellow-derived, maps', 'age': 'Age', 'flags': 'Map availability',
             'H': 'Fellow treatment history'}
ABCD = {'A': '#1F6FB4', 'B': '#8E5BB5', 'C': '#C2561A', 'other': '#BDBDBD'}
SEQ = LinearSegmentedColormap.from_list('kc_seq', ['#F6F7F9', '#C9D7EA', '#86A9D2', '#3F72AE', '#123A6B'])
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
