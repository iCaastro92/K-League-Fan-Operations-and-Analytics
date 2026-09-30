import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# 1. Target engineering
# ---------------------------------------------------------------------------
def create_churn_risk(df: pd.DataFrame,
                      no_show_threshold: float = 0.45,
                      merch_quantile: float = 0.35,
                      base_noise: float = 0.15,
                      random_state: int = 42) -> pd.DataFrame:
    """
    Build a probabilistic binary `churn_risk` label.

    Logic:
      - Fans with no_show_rate > threshold AND low merch spend get a HIGH
        probability of churn (0.85).
      - Fans matching only one condition get a MEDIUM probability (0.45).
      - Everyone else gets a LOW probability (0.10).
      - Bernoulli sampling on top of these probabilities introduces
        randomness so the rule is not deterministic.
    """
    df = df.copy()
    rng = np.random.default_rng(random_state)

    low_spend_cutoff = df["total_merch_spend_krw"].quantile(merch_quantile)

    high_no_show = df["no_show_rate"] > no_show_threshold
    low_spend    = df["total_merch_spend_krw"] < low_spend_cutoff

    probs = np.select(
        [high_no_show & low_spend,
         high_no_show | low_spend],
        [0.85,
         0.45],
        default=0.10
    )

    df["churn_risk"] = rng.binomial(1, probs)
    return df


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals and drop identifier columns."""
    df = df.copy()
    df = pd.get_dummies(df, columns=["membership_type"], drop_first=False)

    drop_cols = [c for c in ["fan_id", "churn_risk"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Coerce any residual object dtypes (defensive)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X


# ---------------------------------------------------------------------------
# 3. Model training
# ---------------------------------------------------------------------------
def train_random_forest(X: pd.DataFrame,
                        y: pd.Series,
                        test_size: float = 0.2,
                        random_state: int = 42):
    """Split, fit a RandomForestClassifier, return fitted model + splits."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test) -> str:
    """Print classification report and return predictions."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    return y_pred


def plot_feature_importance(model,
                            feature_names,
                            top_n: int = 15,
                            figsize=(10, 7)) -> None:
    """Horizontal bar plot of the top-N most important features."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=figsize)
    sns.barplot(
        x=top_features.values,
        y=top_features.index,
        hue=top_features.index,  # <-- FIX: Asignamos 'y' a 'hue'
        palette="viridis",
        legend=False             # <-- FIX: Apagamos la leyenda
    )
    plt.title(f"Top {top_n} Feature Importances - Churn Risk Model")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------
def run_churn_pipeline(crm_df: pd.DataFrame):
    """End-to-end: label -> preprocess -> train -> evaluate -> plot."""
    # Step 1: probabilistic churn label
    df_labeled = create_churn_risk(crm_df)

    # Step 2: feature matrix
    X = preprocess_features(df_labeled)
    y = df_labeled["churn_risk"]

    # Step 3: train
    model, X_train, X_test, y_train, y_test = train_random_forest(X, y)

    # Step 4: evaluate
    evaluate_model(model, X_test, y_test)

    # Step 5: visualize
    plot_feature_importance(model, X.columns, top_n=15)

    return model, X, y


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load the synthetic data generated in Phase 1 & 2
    crm_df = pd.read_csv("k_league_crm_data.csv")
    
    # Run the machine learning pipeline
    model, X, y = run_churn_pipeline(crm_df)

# Generate predictions and probabilities for the entire dataset
    crm_df['churn_prediction'] = model.predict(X)
    crm_df['churn_probability'] = model.predict_proba(X)[:, 1]

    # Export the enriched dataset for Phase 4 (Power BI)
    output_filename = "k_league_predictions_2026.csv"
    crm_df.to_csv(output_filename, index=False)
    print(f"Phase 3 Complete: Predictions successfully saved to {output_filename}")