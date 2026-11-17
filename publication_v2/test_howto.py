"""Every report figure carries a 'how to read it' note; img() embeds it under the picture."""
from pathlib import Path

import howto

FIG = Path(__file__).parent / 'figures'


def test_every_report_figure_has_a_note():
    missing = [p.stem for p in FIG.glob('*.png') if howto.note(p.stem) is None]
    assert not missing, missing


def test_all_scans_copy_uses_same_note():
    assert howto.note('shap_beeswarm_noH_allscans') == howto.note('shap_beeswarm_noH')


def test_img_appends_note_and_can_skip_it():
    import build_report as B
    p = next(FIG.glob('ladder.png'))
    assert 'How to read it' in B.img(p)
    assert 'How to read it' not in B.img(p, howto=False)
