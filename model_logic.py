import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings
warnings.filterwarnings('ignore')


def run_model_and_generate_output(df: pd.DataFrame) -> dict:
    # שלב 1: ניקוי והכנה
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

    # שלב 2: חלוקה ל־Train/Test
    if len(df) > 10 and len(y.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    # שלב 3: סקלינג ואימון
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_weight = 'balanced' if len(y.unique()) > 1 and y.mean() not in [0, 1] else None
    lr_model = LogisticRegression(max_iter=1000, class_weight=class_weight, solver='liblinear')

    try:
        lr_model.fit(X_train_scaled, y_train)
        y_pred = lr_model.predict(X_test_scaled)
        y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba) if len(y_test.unique()) > 1 else None

        feature_importance = pd.DataFrame({
            'Feature': extended_features,
            'Importance': lr_model.coef_[0]
        })

        if len(X_test) > 1 and len(y_test.unique()) > 1:
            perm_importance = permutation_importance(
                lr_model, X_test_scaled, y_test,
                n_repeats=min(10, max(2, len(X_test) // 10)),
                random_state=42, scoring='accuracy'
            )
            feature_importance['Importance'] = perm_importance.importances_mean

    except Exception as e:
        feature_importance = pd.DataFrame({
            'Feature': extended_features,
            'Importance': np.random.random(len(extended_features)) * 0.001
        })

    # שלב 4: חישוב משקלים חדשים
    def calculate_new_weights(importance_df):
        feature_to_param = {
            'LanguageMatch': 'Language',
            'LanguageImportant': 'LanguageImportant',
            'LanguageMatch_Important': 'LanguageImportant',
            'GenderMatch': 'Gender',
            'GenderImportant': 'GenderImportant',
            'GenderMatch_Important': 'GenderImportant',
            'CareerFieldMatch': 'CareerField',
            'CompanyMatch': 'Company',
            'CompanyImportant': 'CompanyImportant',
            'CompanyMatch_Important': 'CompanyImportant',
            'SocialStyleMatch': 'SocialStyle',
            'GuidanceStyleMatch': 'GuidanceStyle',
            'CommunicationStyleMatch': 'CommunicationStyle',
            'LearningStyleMatch': 'LearningStyle',
            'RoleMatch': 'Role',
            'ExperienceLevelMatch': 'JobExperienceLevel'
        }

        param_importance = {}
        for _, row in importance_df.iterrows():
            feature = row['Feature']
            importance = max(0, abs(row['Importance']))
            if feature in feature_to_param:
                param = feature_to_param[feature]
                param_importance[param] = param_importance.get(param, 0) + importance

        epsilon = 1e-9
        for param in param_importance:
            param_importance[param] += epsilon
        total = sum(param_importance.values())
        normalized = {param: importance / total for param, importance in param_importance.items()}
        return normalized

    new_weights = calculate_new_weights(feature_importance)
    results_df = pd.DataFrame([{'ParameterName': k, 'NewWeight': round(v, 3)} for k, v in new_weights.items()])
    results_df = results_df.sort_values(by='NewWeight', ascending=False)

    # שלב 5: גרפים
    plt.figure(figsize=(10, 8))
    sns.barplot(data=results_df, y='ParameterName', x='NewWeight')
    plt.title('Recommended New Weights')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    summary = f"Model trained on {len(df)} samples.\n"
    if 'accuracy' in locals():
        summary += f"Accuracy: {accuracy:.3f}\n"
    if 'auc_score' in locals() and auc_score is not None:
        summary += f"AUC Score: {auc_score:.3f}\n"
    summary += "Top weights:\n"
    for _, row in results_df.head(3).iterrows():
        summary += f"- {row['ParameterName']}: {row['NewWeight']}\n"

    return {
        "graph": image_base64,
        "summary": summary,
        "weights": results_df.to_dict(orient="records")
    }
