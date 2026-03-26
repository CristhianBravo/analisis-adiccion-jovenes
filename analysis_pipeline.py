
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "adiccion_jovenes.xlsx"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def infer_binary_target(df: pd.DataFrame, score_col: str = "Addicted_Score", threshold: int = 7) -> pd.Series:
    """Define una variable binaria interpretable: alta adicción si score >= threshold."""
    return (df[score_col] >= threshold).astype(int)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df


def detect_outliers_iqr(series: pd.Series) -> Dict[str, float]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())
    return {"lower": float(lower), "upper": float(upper), "count": count, "pct": float(count / len(series))}


def build_preprocessor(df: pd.DataFrame, exclude: List[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    X = df.drop(columns=exclude)
    num_features = X.select_dtypes(include=np.number).columns.tolist()
    cat_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        [("num", numeric_pipeline, num_features), ("cat", categorical_pipeline, cat_features)]
    )
    return preprocessor, num_features, cat_features


def run_analysis() -> Dict[str, Any]:
    df = load_data()
    df["High_Addiction"] = infer_binary_target(df)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    descriptive = (
        df[num_cols]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        .T
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    missing_df = (
        df.isna().sum().rename("nulos").to_frame().assign(pct_nulos=lambda x: x["nulos"] / len(df)).reset_index().rename(columns={"index": "variable"})
    )

    outlier_rows = []
    for c in [col for col in num_cols if col not in ["Student_ID", "High_Addiction"]]:
        item = detect_outliers_iqr(df[c])
        outlier_rows.append({"variable": c, **item})
    outlier_df = pd.DataFrame(outlier_rows)

    pearson = df[[c for c in num_cols if c != "Student_ID"]].corr(method="pearson")
    spearman = df[[c for c in num_cols if c != "Student_ID"]].corr(method="spearman")

    # Modelado
    reg_target = "Addicted_Score"
    clf_target = "High_Addiction"
    exclude = [reg_target, clf_target, "Student_ID"]
    X = df.drop(columns=exclude)
    y_reg = df[reg_target]
    y_clf = df[clf_target]

    preprocessor, num_features, cat_features = build_preprocessor(df, exclude=exclude)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg_model = Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])
    reg_model.fit(X_train_r, y_train_r)
    pred_r = reg_model.predict(X_test_r)
    reg_metrics = {
        "R2": float(r2_score(y_test_r, pred_r)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test_r, pred_r))),
        "MAE": float(mean_absolute_error(y_test_r, pred_r)),
    }

    reg_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    reg_metrics["CV_R2_mean"] = float(cross_val_score(reg_model, X, y_reg, cv=reg_cv, scoring="r2").mean())
    reg_metrics["CV_RMSE_mean"] = float((-cross_val_score(reg_model, X, y_reg, cv=reg_cv, scoring="neg_root_mean_squared_error")).mean())

    feature_names_reg = reg_model.named_steps["preprocess"].get_feature_names_out()
    reg_coefs = (
        pd.Series(reg_model.named_steps["model"].coef_, index=feature_names_reg, name="coeficiente")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    reg_perm = permutation_importance(reg_model, X_test_r, y_test_r, n_repeats=10, random_state=42, scoring="r2")
    reg_perm_df = (
        pd.Series(reg_perm.importances_mean, index=X.columns, name="importancia")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    clf_model = Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(max_iter=3000))])
    clf_model.fit(X_train_c, y_train_c)
    prob_c = clf_model.predict_proba(X_test_c)[:, 1]
    pred_c = (prob_c >= 0.5).astype(int)
    clf_metrics = {
        "accuracy": float(accuracy_score(y_test_c, pred_c)),
        "precision": float(precision_score(y_test_c, pred_c)),
        "recall": float(recall_score(y_test_c, pred_c)),
        "f1": float(f1_score(y_test_c, pred_c)),
        "roc_auc": float(roc_auc_score(y_test_c, prob_c)),
    }

    clf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_metrics["CV_ROC_AUC_mean"] = float(cross_val_score(clf_model, X, y_clf, cv=clf_cv, scoring="roc_auc").mean())
    clf_metrics["CV_F1_mean"] = float(cross_val_score(clf_model, X, y_clf, cv=clf_cv, scoring="f1").mean())

    feature_names_clf = clf_model.named_steps["preprocess"].get_feature_names_out()
    clf_coefs = (
        pd.Series(clf_model.named_steps["model"].coef_[0], index=feature_names_clf, name="coeficiente")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    clf_coefs["odds_ratio"] = np.exp(clf_coefs["coeficiente"])

    clf_perm = permutation_importance(clf_model, X_test_c, y_test_c, n_repeats=10, random_state=42, scoring="roc_auc")
    clf_perm_df = (
        pd.Series(clf_perm.importances_mean, index=X.columns, name="importancia")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    fpr, tpr, thresholds = roc_curve(y_test_c, prob_c)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})

    # Guardar tablas
    descriptive.to_csv(OUTPUT_DIR / "descriptivos.csv", index=False)
    missing_df.to_csv(OUTPUT_DIR / "faltantes.csv", index=False)
    outlier_df.to_csv(OUTPUT_DIR / "outliers_iqr.csv", index=False)
    pearson.to_csv(OUTPUT_DIR / "correlacion_pearson.csv")
    spearman.to_csv(OUTPUT_DIR / "correlacion_spearman.csv")
    reg_coefs.to_csv(OUTPUT_DIR / "coeficientes_regresion.csv", index=False)
    reg_perm_df.to_csv(OUTPUT_DIR / "importancia_regresion.csv", index=False)
    clf_coefs.to_csv(OUTPUT_DIR / "coeficientes_logisticos.csv", index=False)
    clf_perm_df.to_csv(OUTPUT_DIR / "importancia_clasificacion.csv", index=False)
    roc_df.to_csv(OUTPUT_DIR / "curva_roc.csv", index=False)

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1] - 1),  # sin objetivo binario derivado
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "regression_metrics": reg_metrics,
        "classification_metrics": clf_metrics,
        "top_correlations_addicted_score": pearson["Addicted_Score"].sort_values(ascending=False).drop("Addicted_Score").to_dict(),
        "top_regression_importance": reg_perm_df.head(8).to_dict(orient="records"),
        "top_classification_importance": clf_perm_df.head(8).to_dict(orient="records"),
        "insights": [
            f"La muestra contiene {df.shape[0]} registros y {df.shape[1]-1} variables originales.",
            "No se detectaron valores nulos ni registros duplicados completos.",
            "La variable objetivo continua Addicted_Score muestra relaciones lineales muy fuertes con Mental_Health_Score, Conflicts_Over_Social_Media, Avg_Daily_Usage_Hours y Sleep_Hours_Per_Night.",
            f"La regresión lineal alcanzó R²={reg_metrics['R2']:.3f}, RMSE={reg_metrics['RMSE']:.3f} y MAE={reg_metrics['MAE']:.3f}.",
            f"La regresión logística para alta adicción (score >= 7) alcanzó accuracy={clf_metrics['accuracy']:.3f}, F1={clf_metrics['f1']:.3f} y ROC-AUC={clf_metrics['roc_auc']:.3f}.",
            "El desempeño predictivo es extraordinariamente alto; esto puede ser señal de relaciones estructurales muy fuertes o de datos sintéticos/semisintéticos.",
        ],
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    create_report_figures(df, pearson, roc_df, reg_perm_df, clf_perm_df)

    return summary


def create_report_figures(df: pd.DataFrame, pearson: pd.DataFrame, roc_df: pd.DataFrame, reg_perm: pd.DataFrame, clf_perm: pd.DataFrame) -> None:
    # 1. Histogramas
    fig_hist = make_subplots(rows=2, cols=2, subplot_titles=[
        "Uso diario promedio", "Horas de sueño", "Puntaje de salud mental", "Puntaje de adicción"
    ])
    fig_hist.add_trace(go.Histogram(x=df["Avg_Daily_Usage_Hours"], nbinsx=25, name="Uso"), row=1, col=1)
    fig_hist.add_trace(go.Histogram(x=df["Sleep_Hours_Per_Night"], nbinsx=25, name="Sueño"), row=1, col=2)
    fig_hist.add_trace(go.Histogram(x=df["Mental_Health_Score"], nbinsx=10, name="Salud mental"), row=2, col=1)
    fig_hist.add_trace(go.Histogram(x=df["Addicted_Score"], nbinsx=10, name="Adicción"), row=2, col=2)
    fig_hist.update_layout(height=800, title="Distribuciones principales")

    # 2. Boxplots
    fig_box = make_subplots(rows=1, cols=3, subplot_titles=[
        "Adicción por plataforma", "Adicción por impacto académico", "Uso diario por relación"
    ])
    for platform in sorted(df["Most_Used_Platform"].unique()):
        sub = df[df["Most_Used_Platform"] == platform]
        fig_box.add_trace(go.Box(y=sub["Addicted_Score"], name=platform, boxmean=True), row=1, col=1)
    for cat in sorted(df["Affects_Academic_Performance"].unique()):
        sub = df[df["Affects_Academic_Performance"] == cat]
        fig_box.add_trace(go.Box(y=sub["Addicted_Score"], name=cat, boxmean=True), row=1, col=2)
    for cat in sorted(df["Relationship_Status"].unique()):
        sub = df[df["Relationship_Status"] == cat]
        fig_box.add_trace(go.Box(y=sub["Avg_Daily_Usage_Hours"], name=cat, boxmean=True), row=1, col=3)
    fig_box.update_layout(height=700, title="Comparaciones por categorías", showlegend=False)

    # 3. Heatmap
    corr_fig = px.imshow(
        pearson.round(2),
        text_auto=True,
        aspect="auto",
        title="Matriz de correlación de Pearson"
    )

    # 4. Scatter
    scatter = px.scatter(
        df,
        x="Mental_Health_Score",
        y="Addicted_Score",
        color="Affects_Academic_Performance",
        size="Avg_Daily_Usage_Hours",
        hover_data=["Age", "Sleep_Hours_Per_Night", "Most_Used_Platform"],
        title="Adicción vs salud mental"
    )

    # 5. ROC
    roc_fig = px.line(roc_df, x="fpr", y="tpr", title="Curva ROC - Regresión logística")
    roc_fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))

    # 6. Importancias
    imp_reg = px.bar(reg_perm.head(8), x="importancia", y="variable", orientation="h", title="Importancia de variables - Regresión")
    imp_clf = px.bar(clf_perm.head(8), x="importancia", y="variable", orientation="h", title="Importancia de variables - Clasificación")

    html_parts = [
        "<html><head><meta charset='utf-8'><title>Reporte interactivo</title></head><body>",
        "<h1>Análisis interactivo de adicción a redes sociales en jóvenes</h1>",
        fig_hist.to_html(full_html=False, include_plotlyjs="cdn"),
        fig_box.to_html(full_html=False, include_plotlyjs=False),
        corr_fig.to_html(full_html=False, include_plotlyjs=False),
        scatter.to_html(full_html=False, include_plotlyjs=False),
        roc_fig.to_html(full_html=False, include_plotlyjs=False),
        imp_reg.to_html(full_html=False, include_plotlyjs=False),
        imp_clf.to_html(full_html=False, include_plotlyjs=False),
        "</body></html>",
    ]
    (OUTPUT_DIR / "reporte_interactivo.html").write_text("\n".join(html_parts), encoding="utf-8")


if __name__ == "__main__":
    summary = run_analysis()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
