import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings

warnings.filterwarnings('ignore')

def run_model_and_generate_output(df: pd.DataFrame) -> dict:
    # ===== 1. Feature Engineering =====
    feature_columns = [
        'LanguageMatch', 'GenderMatch', 'CompanyMatch', 'CareerFieldMatch',
        'RoleMatch', 'SocialStyleMatch', 'GuidanceStyleMatch',
        'CommunicationStyleMatch', 'LearningStyleMatch', 'ExperienceLevelMatch',
        'LanguageImportant', 'GenderImportant', 'CompanyImportant'
    ]

    df[feature_columns] = df[feature_columns].fillna(0)

    df['LanguageMatch_Important'] = df['LanguageMatch'] * df['LanguageImportant']
    df['GenderMatch_Important'] = df['GenderMatch'] * df['GenderImportant']
    df['CompanyMatch_Important'] = df['CompanyMatch'] * df['CompanyImportant']

    extended_features = feature_columns + [
        'LanguageMatch_Important', 'GenderMatch_Important', 'CompanyMatch_Important'
    ]
    X = df[extended_features].copy()
    y = df['Success'].copy()

    if len(df) > 10 and len(y.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    # ===== 2. Scaling + Model Training =====
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_weight = 'balanced' if len(y.unique()) > 1 else None
    model = LogisticRegression(max_iter=1000, class_weight=class_weight, solver='liblinear')

    try:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if len(y_test.unique()) > 1 else None

        importance = pd.DataFrame({'Feature': extended_features, 'Importance': model.coef_[0]})

        if len(X_test) > 1 and len(y_test.unique()) > 1:
            perm = permutation_importance(model, X_test_scaled, y_test, scoring='accuracy',
                                          n_repeats=10, random_state=42)
            importance['Importance'] = perm.importances_mean
    except Exception:
        importance = pd.DataFrame({'Feature': extended_features,
                                   'Importance': np.random.rand(len(extended_features)) * 0.001})

    # ===== 3. Calculate Weights =====
    feature_to_param = {
        'LanguageMatch': 'Language', 'LanguageImportant': 'LanguageImportant',
        'LanguageMatch_Important': 'LanguageImportant', 'GenderMatch': 'Gender',
        'GenderImportant': 'GenderImportant', 'GenderMatch_Important': 'GenderImportant',
        'CareerFieldMatch': 'CareerField', 'CompanyMatch': 'Company',
        'CompanyImportant': 'CompanyImportant', 'CompanyMatch_Important': 'CompanyImportant',
        'SocialStyleMatch': 'SocialStyle', 'GuidanceStyleMatch': 'GuidanceStyle',
        'CommunicationStyleMatch': 'CommunicationStyle',
        'LearningStyleMatch': 'LearningStyle', 'RoleMatch': 'Role',
        'ExperienceLevelMatch': 'JobExperienceLevel'
    }

    param_weights = {}
    for _, row in importance.iterrows():
        feature = row['Feature']
        importance_val = max(0, abs(row['Importance']))
        param = feature_to_param.get(feature)
        if param:
            param_weights[param] = param_weights.get(param, 0) + importance_val

    # Normalize
    epsilon = 1e-9
    total = sum(param_weights.values()) + epsilon
    normalized_weights = {k: v / total for k, v in param_weights.items()}

    # ===== 4. Generate CSV =====
    results_df = pd.DataFrame([{'ParameterName': k, 'NewWeight': round(v, 3)}
                               for k, v in normalized_weights.items()]).sort_values('NewWeight', ascending=False)
    results_df.to_csv("new_weights_recommendations.csv", index=False)

    # ===== 5. Prepare Comparison Plot =====
    current_weights = {
        'Language': 0.2, 'LanguageImportant': 0.3, 'Gender': 0.15,
        'GenderImportant': 0.23, 'CareerField': 0.25, 'Company': 0.1,
        'CompanyImportant': 0.15, 'SocialStyle': 0.08, 'GuidanceStyle': 0.08,
        'CommunicationStyle': 0.07, 'LearningStyle': 0.07, 'Role': 0.2,
        'JobExperienceLevel': 0.06
    }

    comparison_df = pd.DataFrame([{
        'Parameter': key,
        'Current': current_weights.get(key, 0),
        'New': normalized_weights.get(key, 0)
    } for key in sorted(set(list(current_weights.keys()) + list(normalized_weights.keys())))])

    # ===== 6. Create Graphs =====
    image_uris = []

    # First plot: new weights
    plt.figure(figsize=(8, 6))
    sns.barplot(data=results_df, y='ParameterName', x='NewWeight')
    plt.title("🔧 Recommended New Weights")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    image_uris.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    plt.close()

    # Second plot: comparison
    melted = comparison_df.melt(id_vars='Parameter', var_name='Type', value_name='Weight')
    plt.figure(figsize=(10, 8))
    sns.barplot(data=melted, y='Parameter', x='Weight', hue='Type')
    plt.title("⚖️ Current vs Recommended Weights")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    image_uris.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    plt.close()

    # ===== 7. Read CSV for client
    with open("new_weights_recommendations.csv", "rb") as f:
        csv_base64 = base64.b64encode(f.read()).decode("utf-8")

    # ===== 8. Summary string
    summary = f"Model trained on {len(df)} rows.\n"
    if 'accuracy' in locals(): summary += f"Accuracy: {accuracy:.3f}\n"
    if 'auc' in locals() and auc is not None: summary += f"AUC Score: {auc:.3f}\n"

    summary += "\nTop 3 parameters:\n"
    for _, row in results_df.head(3).iterrows():
        summary += f"- {row['ParameterName']}: {row['NewWeight']}\n"

    return {
        "graphList": image_uris,  # base64 list of graphs
        "summary": summary,
        "weights": results_df.to_dict(orient="records"),
        "csv_base64": csv_base64
    }
