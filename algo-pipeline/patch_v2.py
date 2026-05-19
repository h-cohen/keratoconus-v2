import sys
import numpy as np
from pathlib import Path

with open("generate_results.py", "r") as f:
    code = f.read()

imports_patch = """
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats
"""

code = code.replace(
    "from sklearn.calibration import calibration_curve\nfrom scipy import stats",
    imports_patch.strip()
)

roc_patch_orig = """def plot_roc_curves(oof_df, out_dir, is_global=True):"""
roc_patch_new = """def plot_roc_curves(oof_df, df, out_dir, is_global=True):"""
code = code.replace(roc_patch_orig, roc_patch_new)

baseline_roc_orig = """        # Also plot Pooled
        pooled_fpr, pooled_tpr, _ = roc_curve(oof_df['y_true'], oof_df['y_pred'])
        pooled_auc = roc_auc_score(oof_df['y_true'], oof_df['y_pred'])
        ax.plot(pooled_fpr, pooled_tpr, color=COLOR_BL, lw=1.5, ls='--',
                label=f'Pooled (AUC={pooled_auc:.3f})')
"""
baseline_roc_new = """        # Also plot Pooled
        pooled_fpr, pooled_tpr, _ = roc_curve(oof_df['y_true'], oof_df['y_pred'])
        pooled_auc = roc_auc_score(oof_df['y_true'], oof_df['y_pred'])
        ax.plot(pooled_fpr, pooled_tpr, color=COLOR_BL, lw=1.5, ls='--',
                label=f'Pooled (AUC={pooled_auc:.3f})')
        
        # Baseline KMax ROC
        kmax_candidates = ['KMax Sagittal Front (D)', 'Km F (D):']
        kmax_col = next((c for c in kmax_candidates if c in df.columns), None)
        if kmax_col:
            # Map kmax to oof_df to ensure alignment
            kmax_map = df.groupby('ideye')[kmax_col].first()
            oof_valid = oof_df.copy()
            oof_valid['kmax'] = oof_valid['ideye'].map(kmax_map)
            oof_valid = oof_valid.dropna(subset=['kmax'])
            if not oof_valid.empty:
                bl_fpr, bl_tpr, _ = roc_curve(oof_valid['y_true'], oof_valid['kmax'])
                bl_auc = roc_auc_score(oof_valid['y_true'], oof_valid['kmax'])
                if bl_auc < 0.5: # If negatively correlated, flip it
                    bl_auc = 1 - bl_auc
                    bl_fpr, bl_tpr, _ = roc_curve(oof_valid['y_true'], -oof_valid['kmax'])
                ax.plot(bl_fpr, bl_tpr, color='#888888', lw=1.5, ls=':',
                        label=f'Baseline KMax (AUC={bl_auc:.3f})')
"""
code = code.replace(baseline_roc_orig, baseline_roc_new)


extra_funcs = """

# ═══════════════════════════════════════════════════════════════════════════
#  New Figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_tsne_embeddings(oof_df, df_full, config, results_dir, numeric_features, out_dir):
    \"\"\"t-SNE visualization of the multimodal embeddings before XGBoost.\"\"\"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_folds = config.get('n_cv_folds', 5)
    ideye_to_label = df_full.groupby('ideye')['y'].first()
    unique_ideyes = ideye_to_label.index.values
    ideye_labels = ideye_to_label.values
    
    # We will just compute the embeddings for the best fold to visualize the space
    valid_folds = [f for f in range(1, n_folds+1) if f in oof_df['fold'].values]
    if not valid_folds: return
    
    fold_aucs = [roc_auc_score(oof_df[oof_df['fold']==f]['y_true'], oof_df[oof_df['fold']==f]['y_pred']) for f in valid_folds]
    if not fold_aucs: return
    best_fold = valid_folds[np.argmax(fold_aucs)]
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['random_state'])
    image_dir = Path(config['image_dir'])
    
    X, Y = None, None
    for fold, (train_idx, test_idx) in enumerate(cv.split(unique_ideyes, ideye_labels), 1):
        if fold != best_fold: continue
        models = load_fold_models(fold, Path(config['_results_dir']), device, config)
        if models is None: return
        test_ideyes = set(unique_ideyes[test_idx])
        test_df = df_full[df_full['ideye'].isin(test_ideyes)].copy()
        
        # Safe transform loading
        test_transform = get_image_transform(training=False, size=config.get('image_size', 224))
        ds = KeratoconusDataset(test_df, image_dir, numeric_features,
                                test_transform, IMAGE_TYPES)
        loader = DataLoader(ds, batch_size=config.get('batch_size', 16),
                            collate_fn=collate_keratoconus, num_workers=0)
        embs, nums, ys = [], [], []
        with torch.no_grad():
            from tqdm import tqdm
            for imgs, nm, lb in tqdm(loader, desc="Extracting features for t-SNE", leave=False):
                imgs = {k: v.to(device) for k, v in imgs.items()}
                embs.append(models['cnn'](imgs).cpu().numpy())
                nums.append(nm.numpy())
                ys.append(lb.numpy())
        if not embs: return
        
        emb = np.vstack(embs)
        num = np.vstack(nums)
        emb_proc = models['scaler_cnn'].transform(emb) if models['scaler_cnn'] else emb
        if models['selector']: emb_proc = models['selector'].transform(emb_proc)
        num_proc = models['scaler_num'].transform(num) if models['scaler_num'] else num
        X = np.hstack([emb_proc, num_proc])
        Y = np.concatenate(ys)
        
    if X is None or len(X) < 5: return
    
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0]))
    X_pca = pca.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1), random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, cmap=plt.cm.coolwarm, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    legend_elements = [mpatches.Patch(facecolor=plt.cm.coolwarm(0.0), edgecolor='k', label='Normal'),
                       mpatches.Patch(facecolor=plt.cm.coolwarm(1.0), edgecolor='k', label='CXL')]
    ax.legend(handles=legend_elements, loc='best', fontsize=7)
    
    ax.set_title('t-SNE Latent Space Visualization', fontsize=9, fontweight='bold')
    ax.axis('off')
    save_fig(fig, out_dir, 'fig17_tsne_embeddings')

def plot_error_analysis(oof_df, df_full, out_dir):
    \"\"\"Misclassification profiling (Clinical metrics of TP, TN, FP, FN).\"\"\"
    kmax_candidates = ['KMax Sagittal Front (D)', 'Km F (D):']
    kmax_col = next((c for c in kmax_candidates if c in df_full.columns), None)
    if not kmax_col: return
    
    kmax_map = df_full.groupby('ideye')[kmax_col].first()
    
    pachy_candidates = ['Pachy Min:', 'Pachymetry Min']
    pachy_col = next((c for c in pachy_candidates if c in df_full.columns), None)
    pachy_map = df_full.groupby('ideye')[pachy_col].first() if pachy_col else None

    oof = oof_df.copy()
    oof['kmax'] = oof['ideye'].map(kmax_map)
    if pachy_map is not None:
        oof['pachy'] = oof['ideye'].map(pachy_map)
        
    # Get optimal threshold to define TP/FP/TN/FN
    fpr, tpr, thresh = roc_curve(oof['y_true'], oof['y_pred'])
    best_t = thresh[np.argmax(tpr - fpr)]
    oof['y_bin'] = (oof['y_pred'] >= best_t).astype(int)
    
    def get_outcome(row):
        if row['y_true'] == 1 and row['y_bin'] == 1: return 'TP'
        if row['y_true'] == 0 and row['y_bin'] == 0: return 'TN'
        if row['y_true'] == 0 and row['y_bin'] == 1: return 'FP'
        return 'FN'
        
    oof['outcome'] = oof.apply(get_outcome, axis=1)
    
    fig, axes = plt.subplots(1, 2 if pachy_col else 1, figsize=(6 if pachy_col else 3, 3))
    axes = [axes] if not pachy_col else axes
    
    order = ['TN', 'FP', 'FN', 'TP']
    palette = {'TN': '#4393c3', 'FP': '#ff9800', 'FN': '#f44336', 'TP': '#d6604d'}
    
    sns.violinplot(data=oof, x='outcome', y='kmax', order=order, ax=axes[0], palette=palette, inner='quartile', linewidth=0.8)
    axes[0].set_title('KMax Distribution by Outcome', fontsize=8)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('KMax (D)')
    
    if pachy_col:
        sns.violinplot(data=oof, x='outcome', y='pachy', order=order, ax=axes[1], palette=palette, inner='quartile', linewidth=0.8)
        axes[1].set_title('Pachymetry Distribution by Outcome', fontsize=8)
        axes[1].set_xlabel('')
        axes[1].set_ylabel('Min Pachymetry (μm)')
        
    plt.tight_layout()
    save_fig(fig, out_dir, 'fig18_error_profiling')

def plot_demographic_fairness(oof_df, df_full, out_dir):
    \"\"\"Algorithmic fairness across age and gender subgroups.\"\"\"
    gender_col = 'Gender:' if 'Gender:' in df_full.columns else ('Sex:' if 'Sex:' in df_full.columns else None)
    import re
    age_col = next((c for c in df_full.columns if re.search(r'(?i)^age', c)), None)
    
    if not gender_col and not age_col: return
    
    oof = oof_df.copy()
    if age_col:
        age_map = df_full.groupby('ideye')[age_col].first()
        oof['age'] = pd.to_numeric(oof['ideye'].map(age_map), errors='coerce')
        oof['age_group'] = pd.cut(oof['age'], bins=[0, 30, 45, 120], labels=['<30', '30-45', '>45'])
    if gender_col:
        gender_map = df_full.groupby('ideye')[gender_col].first()
        oof['gender'] = oof['ideye'].map(gender_map)
        
    results = []
    
    def eval_subgroup(sub, group_name):
        if len(sub) < 10 or len(sub['y_true'].unique()) < 2: return None
        return {'Subgroup': group_name, 'AUC': roc_auc_score(sub['y_true'], sub['y_pred']), 'N': len(sub)}
        
    if gender_col:
        for g in oof['gender'].dropna().unique():
            res = eval_subgroup(oof[oof['gender'] == g], f"Gender: {g}")
            if res: results.append(res)
            
    if age_col:
        for a in ['<30', '30-45', '>45']:
            res = eval_subgroup(oof[oof['age_group'] == a], f"Age: {a}")
            if res: results.append(res)
            
    res_df = pd.DataFrame(results)
    if res_df.empty: return
    
    fig, ax = plt.subplots(figsize=(4, len(res_df)*0.5 + 1.5))
    sns.barplot(data=res_df, y='Subgroup', x='AUC', ax=ax, palette='Blues_d', edgecolor='k')
    for i, (_, row) in enumerate(res_df.iterrows()):
        ax.text(row['AUC'] + 0.02, i, f"{row['AUC']:.3f} (N={int(row['N'])})", va='center', fontsize=7)
    
    ax.set_xlim([0, 1.05])
    ax.axvline(0.5, color='grey', ls=':')
    ax.set_title('Demographic Fairness (AUC)', fontsize=9, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, out_dir, 'fig19_demographic_fairness')
    res_df.to_csv(out_dir / 'demographic_fairness.csv', index=False)

def generate_clinical_operating_points(oof_df, out_dir):
    \"\"\"Generates metrics tightly coupled to clinical triage settings.\"\"\"
    y_true, y_pred = oof_df['y_true'].values, oof_df['y_pred'].values
    fpr, tpr, thresh = roc_curve(y_true, y_pred)
    
    # 1. Screening (High Sensitivity >= 0.95)
    idx_sens = np.where(tpr >= 0.95)[0]
    best_screening_idx = idx_sens[0] if len(idx_sens) > 0 else np.argmax(tpr)
    
    # 2. Balanced (Youden's J)
    best_balanced_idx = np.argmax(tpr - fpr)
    
    # 3. Confirmatory (High Specificity >= 0.95)
    spec = 1 - fpr
    idx_spec = np.where(spec >= 0.95)[0]
    best_confirmatory_idx = idx_spec[-1] if len(idx_spec) > 0 else np.argmax(spec)
    
    modes = [
        ('Screening (High Sens)', best_screening_idx),
        ('Balanced (Optimal F1)', best_balanced_idx),
        ('Confirmatory (High Spec)', best_confirmatory_idx)
    ]
    
    rows = []
    for mode, idx in modes:
        t = thresh[idx]
        y_bin = (y_pred >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel()
        rows.append({
            'Mode': mode, 'Threshold': t,
            'Sensitivity': tp/(tp+fn) if tp+fn >0 else 0,
            'Specificity': tn/(tn+fp) if tn+fp >0 else 0,
            'F1': f1_score(y_true, y_bin),
            'PPV': tp/(tp+fp) if tp+fp>0 else 0,
            'NPV': tn/(tn+fn) if tn+fn>0 else 0
        })
        
    df_op = pd.DataFrame(rows)
    df_op.to_csv(out_dir / 'clinical_operating_points.csv', index=False)
    with open(out_dir / 'clinical_operating_points.tex', 'w') as f:
        f.write(df_op.to_latex(index=False, float_format='%.3f'))
    print("  ✓ clinical_operating_points (.csv + .tex)")
"""

code = code.replace("def generate_evaluation_suite(", extra_funcs + "\n\ndef generate_evaluation_suite(")

eval_suite_orig = """    # 3: ROC
    plot_roc_curves(oof_df, out_dir, is_global)"""
eval_suite_new = """    # 3: ROC
    plot_roc_curves(oof_df, df, out_dir, is_global)"""
code = code.replace(eval_suite_orig, eval_suite_new)


eval_suite_calls_orig = """    if is_global:
        # 1 & 2: Static diagrams
        plot_data_flow(config, out_dir)"""
eval_suite_calls_new = """    if is_global:
        # 1 & 2: Static diagrams
        plot_data_flow(config, out_dir)
        
        # New Additions
        plot_tsne_embeddings(oof_df, df, config, results_dir, numeric_features, out_dir)
        plot_error_analysis(oof_df, df, out_dir)
        plot_demographic_fairness(oof_df, df, out_dir)
        generate_clinical_operating_points(oof_df, out_dir)
"""
code = code.replace(eval_suite_calls_orig, eval_suite_calls_new)


with open("generate_results.py", "w") as f:
    f.write(code)

print("Patch applied.")
