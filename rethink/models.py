"""Reusable fold-safe model builders. Each returns fit_predict(df_tr, cnn_tr, df_te, cnn_te, seed)."""
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def logreg(cols, C=1.0):
    def fp(dtr, ctr, dte, cte, seed):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        m.fit(dtr[cols].values, dtr.y.values)
        return m.predict_proba(dte[cols].values)[:, 1]
    return fp


def _xgb_fit(X, y, groups, seed, **kw):
    """XGBoost with early stopping on an inner patient-grouped split of the TRAINING fold."""
    p = dict(n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.7, colsample_bytree=0.6)
    p.update(kw)
    tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(X, y, groups))
    m = xgb.XGBClassifier(**p, scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
                          eval_metric='auc', early_stopping_rounds=20, random_state=seed, n_jobs=2)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    return m


def xgb_tab(cols, **kw):
    def fp(dtr, ctr, dte, cte, seed):
        m = _xgb_fit(dtr[cols].values, dtr.y.values, dtr.pid.values, seed, **kw)
        return m.predict_proba(dte[cols].values)[:, 1]
    return fp


def legacy_multimodal(cols, k=30, **kw):
    """v15: scale CNN, SelectKBest(k) on CNN within fold, concat scaled numerics, XGBoost."""
    def fp(dtr, ctr, dte, cte, seed):
        sc = StandardScaler().fit(ctr)
        sel = SelectKBest(f_classif, k=k).fit(sc.transform(ctr), dtr.y.values)
        a_tr, a_te = sel.transform(sc.transform(ctr)), sel.transform(sc.transform(cte))
        if cols:
            sn = StandardScaler().fit(dtr[cols].values)
            a_tr = np.hstack([a_tr, sn.transform(dtr[cols].values)])
            a_te = np.hstack([a_te, sn.transform(dte[cols].values)])
        m = _xgb_fit(a_tr, dtr.y.values, dtr.pid.values, seed, **kw)
        return m.predict_proba(a_te)[:, 1]
    return fp
