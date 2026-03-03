import joblib
import numpy as np
selector = joblib.load('results/v15_frozen_xgboost_20260302_075810/models/fold_2/selector.joblib')
support = selector.get_support(indices=True)
print("support:", support)
print("corneal_thickness:", sum(support < 512))
print("curvature_front:", sum((support >= 512) & (support < 1024)))
print("elevation_front:", sum((support >= 1024) & (support < 1536)))
print("elevation_back:", sum(support >= 1536))
