"""'How to read it' notes shown under each figure of the report. Keyed by figure-name prefix; the all-scans copies
(suffix _allscans) use the same note. First matching prefix wins, so longer prefixes come first."""
import html

ORIENT = ('Orientation: superior up, nasal right; left eyes are mirrored so both eyes share one orientation. '
          'Rings are 0–1, 1–2, 2–3 and 3–4 mm from the map centre; with four quadrants that gives 16 sectors per map.')

NOTES = [
    ('diagram_filtering',
     'Read each panel from top to bottom. Every step shows how many rows remain and how many a rule removed, with the '
     'reason. Panel a is the study cohort: these scans are scored and evaluated. Panel b is the extra exams: they have '
     'no label at the visit, so they are used for training only and never scored.'),
    ('diagram_dataflow',
     'Read from left to right. One visit gives both eyes\' numerics, the four maps and age. Encoding turns these into '
     'column blocks (the sizes are in the boxes). The design matrix is preprocessed inside the training fold only. '
     'Three learners are tuned inside that fold, and their test-fold scores are rank-blended into one referral score. '
     'No step uses the test patients while fitting.'),
    ('roc_age',
     'One panel per arm, one curve per age group; the legend gives referred/eyes and the AUC of each group. Each curve '
     'plots sensitivity against 1 − specificity as the threshold moves. Up and to the left is better; the diagonal is '
     'chance. Curves close together mean the model ranks about equally well in every age group. A group with few '
     'referred eyes has a jagged curve and a wide CI.'),
    ('roc_',
     'Each curve plots sensitivity (y) against 1 − specificity (x) as the decision threshold sweeps over the score. '
     'Up and to the left is better; the diagonal is chance. The AUC in the legend is the probability that a random '
     'referred eye scores above a random not-referred eye. The band is the 95% patient-cluster bootstrap CI (solid '
     'curves only). The panel titles give the scope: first visit is the primary population.'),
    ('ladder',
     'Each row is one model; its column set grows from the top row to the bottom row. Dot = first-visit AUC, bar = '
     '95% CI. Blue is noH, red is H, and the dotted line is age alone. To compare two rows, use the paired Δ printed '
     'on the right, not the overlap of the bars. The Δ is computed on the same eyes, so its CI is much narrower. If '
     'the Δ interval excludes 0, the step adds real discrimination.'),
    ('stage_auc',
     'One panel per model. Each bar is the AUC computed only among eyes of that stage (or age group), with a 95% CI '
     'whisker. Under each bar: the range, the stage and referred/eyes. A hatched bar has fewer than 10 referred or '
     'fewer than 10 not-referred eyes: it shows a point estimate only, so do not interpret it. n/a means the group has '
     'no referred eye. A within-stage AUC cannot use "advanced stage → referral", so it is the harder test. Compare one '
     'stage across the panels.'),
    ('age_adjusted',
     'Open circle = crude AUC over all first-visit eyes. Filled dot = age-adjusted AUC: only referred and '
     'not-referred eyes of the same age group are compared. The numbers on the right read crude → adjusted. The drop '
     'is the part of the discrimination that comes from age. Age alone falls to 0.500 by construction. A filled dot '
     'well to the right of 0.5 means the model separates eyes beyond what age explains.'),
    ('age_cohort',
     'a: age of first-visit eyes, split by referral. b: referral rate in each age group, with a 95% Wilson CI. '
     'c: how each age group splits across stage A. Together they show how strongly referral depends on age, which is '
     'why age alone already gives a useful AUC.'),
    ('shap_sources',
     'Bar length = mean |SHAP| of a group of columns. A group\'s SHAP is summed per eye, made absolute, then averaged. '
     'It is how much that group moves the referral score on average, in either direction. Blue is noH, red is H. '
     'Panel a groups the columns by information source, panel b by encoded family. A longer bar means the model uses '
     'that group more. It shows neither the direction of the effect nor a cause.'),
    ('shap_beeswarm',
     'One row per feature, ranked by mean |SHAP|; the top row is the most used. Each dot is one eye. Its horizontal '
     'position is how much this feature pushed that eye\'s score: right raises referral, left lowers it. Colour is the '
     'eye\'s value of the feature as a percentile (red high, blue low). Red on the right means higher values raise '
     'referral; red on the left means higher values lower it. A wide spread means a strong effect. The chip is the '
     'Belin ABCD category (A anterior curvature, B posterior surface, C thickness); grey means none. SHAP units are '
     'member-standardised: compare sizes within one figure, not as probabilities.'),
    ('shap_dependence',
     'One panel per leading feature. x = feature value, y = its SHAP (push on the score), each dot one eye coloured by '
     'stage A; black line = rolling median. Where the line crosses 0 the feature switches from lowering to raising '
     'the score. A steep step is a threshold-like effect; a flat stretch means no further effect. If all stage colours '
     'follow the same line, the effect does not depend on stage.'),
    ('shap_by_stage',
     'Rows are information sources and columns are stages (or age groups). Each cell is that source\'s share of the '
     'total |SHAP| among eyes of that column, so a column sums to 100%. Read down a column to see what the model leans '
     'on at that stage. Read along a row to see a source gain or lose weight as the disease advances. "—" marks a '
     'group with fewer than 10 eyes.'),
    ('shap_age',
     'x = age, y = SHAP of the age column, each dot one eye coloured by referral; the line is a rolling median. Above '
     '0, age raises the referral score; below 0 it lowers it. The age where the line crosses 0 is where age stops '
     'pushing toward referral.'),
    ('cornea_importance',
     'Each dartboard is one map seen from the front. ' + ORIENT + ' Colour = mean |SHAP| of the sector\'s value: '
     'darker means the model uses that sector more (direction not shown). Rows: own eye, fellow eye, own − fellow '
     'difference. A black outline marks the top 3 sectors of each panel. The heat strips below hold the non-sector '
     'measurements of each map (thinnest point, location of the maximum, asymmetries); their columns are own, fellow '
     'and own − fellow.'),
    ('cornea_signed',
     'Same layout as the importance dartboards, but colour shows direction: Spearman ρ between the sector\'s value and '
     'its SHAP. Red = a higher value raises the referral score; blue = a higher value lowers it; white = no consistent '
     'direction. The outline marks the 3 sectors with the largest |ρ| in each panel. Read it next to the importance '
     'figure: a strong colour on a sector the model hardly uses matters little.'),
    ('cornea_mean_maps',
     'One row per map. The first two columns are the average map of referred and of not-referred eyes, on one shared '
     'colour scale. The third column is their difference (referred − not referred): red is higher in referred eyes, '
     'blue is lower, white is equal. Outlines are the top 3 own-eye sectors by SHAP importance (noH). Check whether '
     'the sectors the model uses lie where the two groups differ. The differences are small (a few µm, or tenths of '
     'a dioptre) because they are averages over many eyes.'),
    ('robustness',
     'a: rank of each leading feature in each of the 25 CV fits (5 folds × 5 repeats). Circle = median rank, line = '
     '5–95% range, on a log scale with rank 1 on the right. A short line means every fit relies on the feature; a long '
     'line means its rank depends on the split. Colour = Belin category, as in the beeswarm. b: each source\'s share '
     'of total importance, one dot per fit, with the median as a bar. Tight clusters mean the overall picture does '
     'not depend on how the data were split.'),
    ('pairs_gap',
     'Each dot is one matched pair: a referred eye and the closest not-referred eye of another patient (same stage A, '
     'similar Km F and age). x = score gap (referred − not referred) under the single-eye model L1; y = the same gap '
     'under the final model. Above the horizontal line, the final model ranks the pair correctly. Right of the '
     'vertical line, L1 does. The shaded upper-left quadrant holds pairs only the final model gets right; compare its '
     'count with "only L1 correct". Hollow dots are pairs the final model gets wrong.'),
    ('operating',
     'a: precision–recall. For each sensitivity (x), y is the share of flagged eyes that were truly referred. The '
     'dotted line is the prevalence, which random flagging would achieve; AP is the area under the curve. b, c: the '
     'final score of not-referred (light) and referred (dark) eyes, one panel per arm. The dashed line is the '
     'threshold that catches 90% of referred eyes. Eyes right of it are flagged, so light area right of the line is '
     'false positives. Scores are within-fold ranks, so the threshold is a rank, not a probability. It is set on the '
     'same scores, so it is descriptive, not validated.'),
    ('reading_map',
     'The 10 most-used corneal sectors over all four maps, numbered by rank (1 = most used). Colour = mean |SHAP| '
     'summed over own eye, fellow eye and own − fellow. ' + ORIENT + ' The list gives, for each sector: the map, ring '
     'and quadrant; which side carries most of it (own, fellow or own − fellow); and the direction (higher or lower → '
     'referral, with ρ). ρ near ±1 means a consistent direction; ρ near 0 means non-monotone.'),
    ('r_gallery',
     'The 12 corneal map measurements with the highest mean |SHAP|, one panel each, in rank order. x = the measured '
     'value (µm, D or mm) for the side named under the axis: own eye, fellow eye or own − fellow. y = SHAP. Navy dots '
     'are referred eyes, grey not referred; the black line is a rolling median. The red dashed line and number mark '
     'where the trend crosses 0: there the measurement switches from lowering to raising the score. The locator '
     'shows the sector in navy, or "whole map" for a non-sector measurement. The crossing value describes the model; '
     'it is not a validated clinical cut-off.'),
    ('atlas_',
     'Rows are subgroups (referral status, stage or age group); columns are the four maps. Colour = each sector\'s '
     'share (%) of that group\'s total sector importance. Each row is therefore normalised, so groups compare on one '
     'scale even when one group\'s SHAP is larger overall (the total is printed at the left). Outlines mark the top 3 '
     'sectors per panel. Look for outlines that move between rows: that is where the model\'s attention shifts with '
     'stage or age. Groups with fewer than 10 eyes are not drawn.'),
]


def note(stem):
    """The 'how to read it' text for a figure name, or None."""
    stem = stem.removesuffix('_allscans')
    return next((t for k, t in NOTES if stem.startswith(k)), None)


def box(stem):
    """HTML block for the note, or '' when the figure has none."""
    t = note(stem)
    return f'<div class="howto"><b>How to read it.</b> {html.escape(t, quote=False)}</div>' if t else ''
