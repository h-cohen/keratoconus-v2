"""E00: clean baselines on the development set (reference points every later experiment must beat)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import evaluate, numeric_features  # noqa: E402
from models import legacy_multimodal, logreg, xgb_tab  # noqa: E402

F = numeric_features()
NOAGE = [f for f in F if f != 'age']

evaluate('E00_age_lr', logreg(['age']), 'age only, logistic regression')
evaluate('E00_clin_lr', logreg(F, C=0.1), 'all 55 numerics, L2 logistic C=0.1')
evaluate('E00_clin_xgb', xgb_tab(F), 'all 55 numerics, XGB (v15 params, inner ES)')
evaluate('E00_clin_noage_xgb', xgb_tab(NOAGE), '54 numerics without age, XGB')
evaluate('E00_mm_legacy', legacy_multimodal(F), 'v15 multimodal (frozen R18, k=30) + 55 numerics, clean CV')
evaluate('E00_mm_noage', legacy_multimodal(NOAGE), 'v15 multimodal without age')
evaluate('E00_img_only', legacy_multimodal([]), 'frozen R18 k=30 only')
