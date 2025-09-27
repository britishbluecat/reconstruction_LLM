from importlib.machinery import SourceFileLoader
mod = SourceFileLoader("cost_model", "cost_model.py").load_module()
mod.write_scored_csv(
    "data/20250928 - case_studies_tokyo_cleansed_int.csv",
    "data/20250928 - case_studies_tokyo_cleansed_int.with_cp.csv",
    n_splits=5, seed=42, n_estimators=200
)