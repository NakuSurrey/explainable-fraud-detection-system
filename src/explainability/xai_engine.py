"""
Phase 7: Explainable AI (XAI) Generation -- THE TRANSLATOR
===========================================================
This module fits SHAP TreeExplainer and LIME TabularExplainer on the
trained model, computes SHAP values for the test set, and saves all
explainer objects to disk so downstream phases (API, Dashboard) can
load them instantly without re-computing.

Reads:
    - models/best_model.pkl          (Phase 5 artifact)
    - data/processed/X_train.csv     (background data for SHAP)
    - data/processed/X_test.csv      (compute SHAP values on test set)
    - data/processed/y_test.csv      (for selecting example transactions)
    - data/processed/feature_names.json  (feature labels)
    - models/metrics.json            (optimal threshold)

Saves:
    - models/shap_explainer.pkl      (fitted TreeExplainer object)
    - models/shap_values.pkl         (pre-computed SHAP values for test set)
    - models/lime_explainer.pkl      (fitted LimeTabularExplainer object)
    - reports/xai_report.json        (summary stats, top features, example explanations)
    - reports/xai_report.txt         (human-readable report for auditors)

Usage:
    python -m src.explainability.xai_engine
"""

import sys
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress non-critical warnings during SHAP/LIME computation
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import (
    get_logger,
    load_config,
    resolve_path,
    log_phase_start,
    log_phase_end,
    check_phase_completed,
)

logger = get_logger(__name__)


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def load_model(config: dict):
    """Load the best trained model from Phase 5."""
    model_path = resolve_path(config["model"]["best_model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model not found at {model_path}. "
            "Run Phase 5 (model_training.py) first."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded best model from {model_path}")
    logger.info(f"Model type: {type(model).__name__}")
    return model


def load_data(config: dict):
    """Load training data (background) and test data for SHAP computation."""
    preprocess_cfg = config["preprocessing"]

    X_train_path = resolve_path(preprocess_cfg["train_path"])
    X_test_path = resolve_path(preprocess_cfg["test_path"])
    y_test_path = resolve_path(preprocess_cfg["y_test_path"])

    for p, name in [(X_train_path, "X_train"), (X_test_path, "X_test"), (y_test_path, "y_test")]:
        if not p.exists():
            raise FileNotFoundError(
                f"{name} not found at {p}. Run Phase 3 first."
            )

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    logger.info(f"X_train shape: {X_train.shape} (background data)")
    logger.info(f"X_test shape:  {X_test.shape} (SHAP computation target)")
    logger.info(f"y_test shape:  {y_test.shape}")

    return X_train, X_test, y_test


def load_feature_names(config: dict) -> list:
    """Load feature names from Phase 3 artifact."""
    feature_path = resolve_path(
        config["preprocessing"]["processed_dir"]
    ) / "feature_names.json"

    # if feature_path.exists():
    #     with open(feature_path, "r") as f:
    #         names = json.load(f)
    #     logger.info(f"Loaded {len(names)} feature names")
    #     return names
    if feature_path.exists():
        with open(feature_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            names = data
        elif isinstance(data, dict):
            for key in ["features", "feature_names", "columns", "names"]:
                if key in data:
                    names = data[key]
                    break
            else:
                names = next(
                    (v for v in data.values() if isinstance(v, list)),
                    list(data.keys())
                )
        else:
            names = None
        if names:
            logger.info(f"Loaded {len(names)} feature names")
        return names
    else:
        logger.warning("feature_names.json not found. Using column indices.")
        return None


def load_optimal_threshold(config: dict) -> float:
    """Load the optimal threshold from Phase 5 metrics."""
    metrics_path = resolve_path(config["model"]["metrics_path"])
    if not metrics_path.exists():
        logger.warning("metrics.json not found. Using default threshold 0.5")
        return 0.5

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Find the winning model's optimal threshold
    winner_key = None
    for key in ["xgboost", "lightgbm"]:
        if key in metrics:
            if winner_key is None:
                winner_key = key
            elif metrics[key]["auprc"] > metrics[winner_key]["auprc"]:
                winner_key = key

    if winner_key and "optimal_threshold" in metrics[winner_key]:
        threshold = metrics[winner_key]["optimal_threshold"]["threshold"]
        logger.info(f"Optimal threshold from {winner_key}: {threshold:.6f}")
        return threshold

    logger.warning("Could not extract optimal threshold. Using 0.5")
    return 0.5


# ===================================================================
# SHAP EXPLAINER
# ===================================================================

def fit_shap_explainer(model, X_background: pd.DataFrame, n_background: int = 100):
    """
    Fit a SHAP TreeExplainer using a subsample of training data as
    the background dataset.

    Args:
        model: Trained tree-based model (XGBoost or LightGBM)
        X_background: Training data to sample from
        n_background: Number of background samples (default from config)

    Returns:
        shap.TreeExplainer object
    """
    import shap

    # Sample background data for efficiency
    if len(X_background) > n_background:
        bg_sample = X_background.sample(
            n=n_background, random_state=42
        )
        logger.info(
            f"Sampled {n_background} background instances "
            f"from {len(X_background)} training rows"
        )
    else:
        bg_sample = X_background
        logger.info(f"Using all {len(bg_sample)} training rows as background")

    logger.info("Fitting SHAP TreeExplainer...")
    start = time.time()

    explainer = shap.TreeExplainer(
        model,
        data=bg_sample,
        feature_perturbation="interventional",
    )

    elapsed = time.time() - start
    logger.info(f"SHAP TreeExplainer fitted in {elapsed:.2f}s")

    return explainer


def compute_shap_values(explainer, X_test: pd.DataFrame):
    """
    Compute SHAP values for the entire test set.

    Returns:
        numpy array of SHAP values, shape (n_samples, n_features)
    """
    import shap

    logger.info(f"Computing SHAP values for {len(X_test)} test samples...")
    start = time.time()

    shap_values = explainer.shap_values(X_test)

    # Handle different SHAP output formats
    # For binary classifiers, shap_values may be a list of 2 arrays
    # (one per class). We want the positive class (fraud = 1).
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            logger.info("SHAP returned list of 2 arrays. Using class 1 (fraud).")
            shap_values = shap_values[1]
        else:
            logger.info(f"SHAP returned list of {len(shap_values)} arrays. Using last.")
            shap_values = shap_values[-1]

    elapsed = time.time() - start
    logger.info(
        f"SHAP values computed in {elapsed:.2f}s "
        f"-- shape: {shap_values.shape}"
    )

    return shap_values


def compute_global_feature_importance(shap_values: np.ndarray, feature_names: list):
    """
    Compute global feature importance as mean absolute SHAP values.

    Returns:
        List of dicts: [{"feature": name, "importance": float}, ...]
        sorted descending by importance.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    if feature_names and len(feature_names) == len(mean_abs):
        names = feature_names
    else:
        names = [f"Feature_{i}" for i in range(len(mean_abs))]

    importance = [
        {"feature": name, "importance": round(float(val), 6)}
        for name, val in zip(names, mean_abs)
    ]
    importance.sort(key=lambda x: x["importance"], reverse=True)

    return importance


def generate_example_explanations(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list,
    model,
    threshold: float,
    n_examples: int = 5,
):
    """
    Generate plain-English explanations for example transactions.
    Picks fraud cases and legitimate cases for demonstration.

    Returns:
        List of example explanation dicts.
    """
    explanations = []
    feature_labels = feature_names if feature_names else [
        f"Feature_{i}" for i in range(X_test.shape[1])
    ]

    # Get fraud and legitimate indices
    fraud_idx = y_test[y_test == 1].index.tolist()
    legit_idx = y_test[y_test == 0].index.tolist()

    # Map to positional indices (in case index is not 0-based sequential)
    fraud_positions = [
        i for i, idx in enumerate(X_test.index) if idx in fraud_idx
    ]
    legit_positions = [
        i for i, idx in enumerate(X_test.index) if idx in legit_idx
    ]

    # Select examples: more fraud than legit for demonstration
    n_fraud = min(n_examples - 1, len(fraud_positions))
    n_legit = min(1, len(legit_positions))

    np.random.seed(42)
    selected_fraud = np.random.choice(
        fraud_positions, size=n_fraud, replace=False
    ) if len(fraud_positions) >= n_fraud else fraud_positions
    selected_legit = np.random.choice(
        legit_positions, size=n_legit, replace=False
    ) if len(legit_positions) >= n_legit else legit_positions[:n_legit]

    selected = list(selected_fraud) + list(selected_legit)

    for pos in selected:
        sv = shap_values[pos]
        row = X_test.iloc[pos]
        actual_label = int(y_test.iloc[pos])

        # Get model prediction probability
        prob = float(model.predict_proba(row.values.reshape(1, -1))[0, 1])
        predicted_fraud = prob >= threshold

        # Top contributing features (sorted by absolute SHAP value)
        sorted_indices = np.argsort(np.abs(sv))[::-1]
        top_n = min(5, len(sorted_indices))

        top_features = []
        reasons = []
        for rank, idx in enumerate(sorted_indices[:top_n]):
            feat_name = feature_labels[idx]
            shap_val = float(sv[idx])
            feat_val = float(row.iloc[idx])
            direction = "increases" if shap_val > 0 else "decreases"

            top_features.append({
                "rank": rank + 1,
                "feature": feat_name,
                "shap_value": round(shap_val, 6),
                "feature_value": round(feat_val, 6),
                "direction": direction,
            })

            # Plain-English reason
            pct_contribution = abs(shap_val) / (sum(np.abs(sv)) + 1e-10) * 100
            reasons.append(
                f"{feat_name} = {feat_val:.4f} "
                f"({direction} fraud risk by {pct_contribution:.1f}%)"
            )

        # Build plain-English summary
        if predicted_fraud:
            summary = (
                f"FLAGGED AS FRAUD (score: {prob:.4f}, threshold: {threshold:.4f}). "
                f"Key drivers: {'; '.join(reasons[:3])}"
            )
        else:
            summary = (
                f"CLEARED (score: {prob:.4f}, threshold: {threshold:.4f}). "
                f"Key factors: {'; '.join(reasons[:3])}"
            )

        explanations.append({
            "position_in_test_set": int(pos),
            "actual_label": actual_label,
            "actual_label_text": "FRAUD" if actual_label == 1 else "LEGITIMATE",
            "predicted_probability": round(prob, 6),
            "predicted_fraud": predicted_fraud,
            "threshold_used": round(threshold, 6),
            "base_value": round(float(np.mean(shap_values)), 6),
            "top_features": top_features,
            "plain_english_explanation": summary,
        })

    return explanations


# ===================================================================
# LIME EXPLAINER
# ===================================================================

def fit_lime_explainer(X_train: pd.DataFrame, feature_names: list, class_names=None):
    """
    Fit a LIME TabularExplainer for local interpretability.

    Unlike SHAP (which computes exact Shapley values for tree models),
    LIME builds a local linear approximation around each prediction.
    We fit it once and save it so the Dashboard can call
    explainer.explain_instance() on demand.

    Returns:
        lime.lime_tabular.LimeTabularExplainer object
    """
    import lime
    import lime.lime_tabular

    if class_names is None:
        class_names = ["Legitimate", "Fraud"]

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    logger.info("Fitting LIME TabularExplainer...")
    start = time.time()

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    elapsed = time.time() - start
    logger.info(f"LIME TabularExplainer fitted in {elapsed:.2f}s")

    return explainer


def generate_lime_example(lime_explainer, model, X_test: pd.DataFrame, position: int):
    """
    Generate a single LIME explanation for a test instance.
    Used for the report -- the Dashboard will generate these on-the-fly.

    Returns:
        Dict with feature contributions from LIME.
    """
    instance = X_test.iloc[position].values

    explanation = lime_explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=10,
        num_samples=1000,
    )

    lime_features = []
    for feat_name, weight in explanation.as_list():
        lime_features.append({
            "feature_rule": feat_name,
            "weight": round(float(weight), 6),
            "direction": "increases risk" if weight > 0 else "decreases risk",
        })

    return {
        "position_in_test_set": position,
        "lime_predicted_class": int(explanation.predict_proba.argmax())
            if hasattr(explanation, "predict_proba") else None,
        "lime_features": lime_features,
    }


# ===================================================================
# REPORT GENERATION
# ===================================================================

def generate_xai_report(
    global_importance: list,
    example_explanations: list,
    lime_examples: list,
    shap_values_shape: tuple,
    shap_fit_time: float,
    shap_compute_time: float,
    lime_fit_time: float,
    n_background: int,
    max_display: int,
    config: dict,
):
    """Generate JSON and human-readable reports."""

    # --- JSON Report ---
    report_data = {
        "phase": "Phase 7: Explainable AI (XAI) Generation",
        "status": "COMPLETED",
        "timestamp": pd.Timestamp.now().isoformat(),
        "configuration": {
            "background_samples": n_background,
            "max_display_features": max_display,
            "shap_values_shape": list(shap_values_shape),
            "shap_method": "TreeExplainer (interventional)",
            "lime_method": "LimeTabularExplainer (discretize_continuous=True)",
        },
        "timing": {
            "shap_explainer_fit_seconds": round(shap_fit_time, 2),
            "shap_values_compute_seconds": round(shap_compute_time, 2),
            "lime_explainer_fit_seconds": round(lime_fit_time, 2),
            "total_seconds": round(shap_fit_time + shap_compute_time + lime_fit_time, 2),
        },
        "global_feature_importance": global_importance[:max_display],
        "example_shap_explanations": example_explanations,
        "example_lime_explanations": lime_examples,
    }

    json_path = resolve_path("reports/xai_report.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    logger.info(f"XAI report (JSON) saved to {json_path}")

    # --- Human-Readable Report ---
    txt_path = resolve_path("reports/xai_report.txt")
    lines = []
    lines.append("=" * 70)
    lines.append("EXPLAINABLE AI (XAI) REPORT")
    lines.append("Phase 7: The Translator")
    lines.append("=" * 70)
    lines.append("")
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    lines.append(
        "This report documents the explainability layer fitted on top of "
        "the trained fraud detection model. Two industry-standard XAI "
        "frameworks were applied:"
    )
    lines.append("")
    lines.append(
        "  1. SHAP (SHapley Additive exPlanations) -- Uses cooperative "
        "game theory to assign exact contribution values to each feature. "
        "Provides both global (model-wide) and local (per-transaction) "
        "interpretability."
    )
    lines.append(
        "  2. LIME (Local Interpretable Model-agnostic Explanations) -- "
        "Builds local linear approximations around individual predictions "
        "for fast, human-readable explanations."
    )
    lines.append("")
    lines.append(
        f"SHAP values were computed for {shap_values_shape[0]} test "
        f"transactions across {shap_values_shape[1]} features."
    )
    lines.append("")

    # Timing
    lines.append("COMPUTATION TIMING")
    lines.append("-" * 40)
    lines.append(f"  SHAP explainer fit:     {shap_fit_time:.2f}s")
    lines.append(f"  SHAP values computed:   {shap_compute_time:.2f}s")
    lines.append(f"  LIME explainer fit:     {lime_fit_time:.2f}s")
    lines.append(
        f"  Total:                  "
        f"{shap_fit_time + shap_compute_time + lime_fit_time:.2f}s"
    )
    lines.append("")

    # Global importance
    lines.append("GLOBAL FEATURE IMPORTANCE (Top 15 by mean |SHAP|)")
    lines.append("-" * 40)
    lines.append(f"  {'Rank':<6}{'Feature':<25}{'Mean |SHAP|':<15}")
    lines.append(f"  {'----':<6}{'-------':<25}{'-----------':<15}")
    for i, feat in enumerate(global_importance[:15]):
        lines.append(
            f"  {i+1:<6}{feat['feature']:<25}{feat['importance']:<15.6f}"
        )
    lines.append("")

    # Example explanations
    lines.append("EXAMPLE TRANSACTION EXPLANATIONS (SHAP)")
    lines.append("-" * 40)
    for ex in example_explanations:
        lines.append("")
        lines.append(
            f"  Transaction #{ex['position_in_test_set']} "
            f"[Actual: {ex['actual_label_text']}]"
        )
        lines.append(f"  Fraud probability: {ex['predicted_probability']:.4f}")
        lines.append(
            f"  Model decision:    "
            f"{'FLAGGED' if ex['predicted_fraud'] else 'CLEARED'}"
        )
        lines.append(f"  Explanation: {ex['plain_english_explanation']}")
        lines.append(f"  Top contributing features:")
        for tf in ex["top_features"][:3]:
            lines.append(
                f"    {tf['rank']}. {tf['feature']} = {tf['feature_value']:.4f} "
                f"(SHAP: {tf['shap_value']:+.4f}, {tf['direction']})"
            )

    lines.append("")

    # LIME examples
    if lime_examples:
        lines.append("EXAMPLE TRANSACTION EXPLANATIONS (LIME)")
        lines.append("-" * 40)
        for lex in lime_examples:
            lines.append("")
            lines.append(f"  Transaction #{lex['position_in_test_set']}")
            lines.append(f"  Top LIME rules:")
            for lf in lex["lime_features"][:5]:
                lines.append(
                    f"    - {lf['feature_rule']} "
                    f"(weight: {lf['weight']:+.4f}, {lf['direction']})"
                )

    lines.append("")
    lines.append("=" * 70)
    lines.append("COMPLIANCE NOTE")
    lines.append(
        "These explainability artifacts satisfy FCA Consumer Duty "
        "requirements by providing transparent, auditable justifications "
        "for every algorithmic decision. Each flagged transaction can be "
        "traced to specific feature contributions."
    )
    lines.append("=" * 70)
    lines.append("")
    lines.append("Artifacts saved:")
    lines.append("  - models/shap_explainer.pkl")
    lines.append("  - models/shap_values.pkl")
    lines.append("  - models/lime_explainer.pkl")
    lines.append("  - reports/xai_report.json")
    lines.append("  - reports/xai_report.txt")
    lines.append("")
    lines.append("End of XAI Report.")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"XAI report (TXT) saved to {txt_path}")

    return report_data


# ===================================================================
# MAIN PIPELINE
# ===================================================================

def main():
    """Run the complete Phase 7 XAI pipeline."""

    phase_name = "Phase 7: Explainable AI Generation"
    log_phase_start(phase_name)

    try:
        config = load_config()
        xai_cfg = config["explainability"]
        n_background = xai_cfg.get("background_samples", 100)
        max_display = xai_cfg.get("max_display_features", 15)

        # ---------------------------------------------------
        # Step 1: Load model and data
        # ---------------------------------------------------
        logger.info("=" * 50)
        logger.info("STEP 1/6: Loading model and data")
        logger.info("=" * 50)

        model = load_model(config)
        X_train, X_test, y_test = load_data(config)
        feature_names = load_feature_names(config)
        threshold = load_optimal_threshold(config)

        # Verify feature count matches
        expected_features = X_train.shape[1]
        logger.info(f"Feature count: {expected_features}")
        if feature_names and len(feature_names) != expected_features:
            logger.warning(
                f"Feature names count ({len(feature_names)}) does not match "
                f"data columns ({expected_features}). Using column names."
            )
            feature_names = list(X_train.columns)

        # ---------------------------------------------------
        # Step 2: Fit SHAP TreeExplainer
        # ---------------------------------------------------
        logger.info("")
        logger.info("=" * 50)
        logger.info("STEP 2/6: Fitting SHAP TreeExplainer")
        logger.info("=" * 50)

        shap_fit_start = time.time()
        shap_explainer = fit_shap_explainer(model, X_train, n_background)
        shap_fit_time = time.time() - shap_fit_start

        # Save SHAP explainer
        shap_explainer_path = resolve_path(xai_cfg["shap_explainer_path"])
        shap_explainer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(shap_explainer_path, "wb") as f:
            pickle.dump(shap_explainer, f)
        logger.info(f"SHAP explainer saved to {shap_explainer_path}")

        # ---------------------------------------------------
        # Step 3: Compute SHAP values for test set
        # ---------------------------------------------------
        logger.info("")
        logger.info("=" * 50)
        logger.info("STEP 3/6: Computing SHAP values for test set")
        logger.info("=" * 50)

        shap_compute_start = time.time()
        shap_values = compute_shap_values(shap_explainer, X_test)
        shap_compute_time = time.time() - shap_compute_start

        # Save SHAP values
        shap_values_path = resolve_path(xai_cfg["shap_values_path"])
        with open(shap_values_path, "wb") as f:
            pickle.dump(shap_values, f)
        logger.info(f"SHAP values saved to {shap_values_path}")
        logger.info(f"SHAP values shape: {shap_values.shape}")

        # ---------------------------------------------------
        # Step 4: Compute global feature importance
        # ---------------------------------------------------
        logger.info("")
        logger.info("=" * 50)
        logger.info("STEP 4/6: Computing global feature importance")
        logger.info("=" * 50)

        global_importance = compute_global_feature_importance(
            shap_values, feature_names
        )

        logger.info("Top 10 features by mean |SHAP value|:")
        for i, feat in enumerate(global_importance[:10]):
            logger.info(
                f"  {i+1:>2}. {feat['feature']:<25} "
                f"importance: {feat['importance']:.6f}"
            )

        # Generate example SHAP explanations
        example_explanations = generate_example_explanations(
            shap_values, X_test, y_test, feature_names,
            model, threshold, n_examples=5
        )

        for ex in example_explanations:
            logger.info("")
            logger.info(
                f"  Example: Transaction #{ex['position_in_test_set']} "
                f"[{ex['actual_label_text']}]"
            )
            logger.info(f"    {ex['plain_english_explanation']}")

        # ---------------------------------------------------
        # Step 5: Fit LIME explainer
        # ---------------------------------------------------
        logger.info("")
        logger.info("=" * 50)
        logger.info("STEP 5/6: Fitting LIME TabularExplainer")
        logger.info("=" * 50)

        lime_fit_start = time.time()
        lime_explainer = fit_lime_explainer(X_train, feature_names)
        lime_fit_time = time.time() - lime_fit_start

        # Save LIME explainer
        # lime_explainer_path = resolve_path(xai_cfg["lime_explainer_path"])
        # with open(lime_explainer_path, "wb") as f:
        #     pickle.dump(lime_explainer, f)
        # logger.info(f"LIME explainer saved to {lime_explainer_path}")
        # Save LIME explainer using dill (handles lambdas that pickle can't)
        import dill
        lime_explainer_path = resolve_path(xai_cfg["lime_explainer_path"])
        with open(lime_explainer_path, "wb") as f:
            dill.dump(lime_explainer, f)
        logger.info(f"LIME explainer saved to {lime_explainer_path}")
        # Generate a couple of LIME examples for the report
        # Use the same fraud positions as SHAP examples
        lime_examples = []
        for ex in example_explanations[:2]:
            try:
                lime_ex = generate_lime_example(
                    lime_explainer, model, X_test,
                    ex["position_in_test_set"]
                )
                lime_examples.append(lime_ex)
                logger.info(
                    f"  LIME explanation generated for "
                    f"transaction #{ex['position_in_test_set']}"
                )
            except Exception as e:
                logger.warning(
                    f"  LIME explanation failed for "
                    f"transaction #{ex['position_in_test_set']}: {e}"
                )

        # ---------------------------------------------------
        # Step 6: Generate reports
        # ---------------------------------------------------
        logger.info("")
        logger.info("=" * 50)
        logger.info("STEP 6/6: Generating XAI reports")
        logger.info("=" * 50)

        report = generate_xai_report(
            global_importance=global_importance,
            example_explanations=example_explanations,
            lime_examples=lime_examples,
            shap_values_shape=shap_values.shape,
            shap_fit_time=shap_fit_time,
            shap_compute_time=shap_compute_time,
            lime_fit_time=lime_fit_time,
            n_background=n_background,
            max_display=max_display,
            config=config,
        )

        # ---------------------------------------------------
        # Summary
        # ---------------------------------------------------
        logger.info("")
        logger.info("=" * 50)
        logger.info("PHASE 7 COMPLETE -- SUMMARY")
        logger.info("=" * 50)
        logger.info(f"SHAP explainer:   models/shap_explainer.pkl")
        logger.info(f"SHAP values:      models/shap_values.pkl")
        logger.info(f"LIME explainer:   models/lime_explainer.pkl")
        logger.info(f"XAI report JSON:  reports/xai_report.json")
        logger.info(f"XAI report TXT:   reports/xai_report.txt")
        logger.info(f"Top feature:      {global_importance[0]['feature']} "
                     f"(importance: {global_importance[0]['importance']:.6f})")
        logger.info(f"SHAP values shape: {shap_values.shape}")
        total_time = shap_fit_time + shap_compute_time + lime_fit_time
        logger.info(f"Total computation time: {total_time:.2f}s")

        log_phase_end(phase_name, "SUCCESS")

    except Exception as e:
        logger.error(f"Phase 7 FAILED: {e}", exc_info=True)
        log_phase_end(phase_name, "FAILED", str(e))
        raise


if __name__ == "__main__":
    main()
