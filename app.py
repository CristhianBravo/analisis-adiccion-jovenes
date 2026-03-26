import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, dash_table

from analysis_pipeline import run_analysis, load_data


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


def safe_run_analysis():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SUMMARY_PATH.exists():
        run_analysis()


def safe_load_summary() -> dict:
    safe_run_analysis()
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def safe_load_df() -> pd.DataFrame:
    df_local = load_data().copy()

    required_cols = ["Addicted_Score", "Age"]
    for col in required_cols:
        if col not in df_local.columns:
            raise ValueError(f"La columna requerida '{col}' no existe en la base.")

    numeric_candidates = [
        "Age",
        "Addicted_Score",
        "Mental_Health_Score",
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Conflicts_Over_Social_Media",
        "Academic_Performance_Impact",
    ]
    for col in numeric_candidates:
        if col in df_local.columns:
            df_local[col] = pd.to_numeric(df_local[col], errors="coerce")

    df_local["High_Addiction"] = (df_local["Addicted_Score"] >= 7).astype(int)
    return df_local


summary = safe_load_summary()
df = safe_load_df()

numeric_cols = [
    c for c in df.select_dtypes(include="number").columns
    if c not in ["Student_ID", "High_Addiction"]
]

age_series = df["Age"].dropna()
if age_series.empty:
    raise ValueError("La columna 'Age' no contiene valores válidos.")

age_min = int(age_series.min())
age_max = int(age_series.max())

# Claves string para evitar el error:
# TypeError: keys must be str, int, float, bool or None, not numpy.int64
age_marks = {str(i): str(i) for i in range(age_min, age_max + 1)}

platform_options = [{"label": "Todas", "value": "ALL"}]
if "Most_Used_Platform" in df.columns:
    platform_values = sorted(df["Most_Used_Platform"].dropna().astype(str).unique().tolist())
    platform_options += [{"label": x, "value": x} for x in platform_values]

academic_options = [{"label": "Todos", "value": "ALL"}]
if "Affects_Academic_Performance" in df.columns:
    academic_values = sorted(df["Affects_Academic_Performance"].dropna().astype(str).unique().tolist())
    academic_options += [{"label": x, "value": x} for x in academic_values]


app = Dash(__name__)
server = app.server
app.title = "Dashboard - Adicción Jóvenes"


def metric_card(title: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, style={"fontSize": "14px", "color": "#666"}),
            html.Div(value, style={"fontSize": "28px", "fontWeight": "bold"}),
        ],
        style={
            "border": "1px solid #ddd",
            "borderRadius": "12px",
            "padding": "16px",
            "backgroundColor": "#fafafa",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
        },
    )


def empty_figure(title: str, message: str = "No hay datos para mostrar con los filtros seleccionados.") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "x": 0.5,
                "y": 0.5,
                "font": {"size": 16},
            }
        ],
        template="plotly_white",
    )
    return fig


app.layout = html.Div(
    [
        html.H1("Dashboard interactivo: adicción a redes sociales en jóvenes"),
        html.P("Explora calidad de datos, relaciones estadísticas y desempeño predictivo de los modelos."),

        html.Div(
            [
                metric_card("Registros", str(summary.get("rows", len(df)))),
                metric_card("Variables", str(summary.get("columns", df.shape[1]))),
                metric_card(
                    "R² regresión",
                    f"{summary.get('regression_metrics', {}).get('R2', 0):.3f}",
                ),
                metric_card(
                    "ROC-AUC clasificación",
                    f"{summary.get('classification_metrics', {}).get('roc_auc', 0):.3f}",
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))",
                "gap": "16px",
                "marginBottom": "24px",
            },
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.Label("Plataforma"),
                        dcc.Dropdown(
                            id="platform-filter",
                            options=platform_options,
                            value="ALL",
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "260px"},
                ),

                html.Div(
                    [
                        html.Label("Impacto académico"),
                        dcc.Dropdown(
                            id="academic-filter",
                            options=academic_options,
                            value="ALL",
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "260px"},
                ),

                html.Div(
                    [
                        html.Label("Edad"),
                        dcc.RangeSlider(
                            id="age-slider",
                            min=age_min,
                            max=age_max,
                            step=1,
                            marks=age_marks,
                            value=[age_min, age_max],
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    style={"padding": "10px 8px 0 8px"},
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
                "gap": "16px",
                "marginBottom": "24px",
                "alignItems": "end",
            },
        ),

        html.Div(
            [
                dcc.Graph(id="histogram"),
                dcc.Graph(id="scatter"),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))",
                "gap": "20px",
                "marginBottom": "16px",
            },
        ),

        html.Div(
            [
                dcc.Graph(id="heatmap"),
                dcc.Graph(id="boxplot"),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))",
                "gap": "20px",
                "marginBottom": "24px",
            },
        ),

        html.Div(
            [
                html.H2("Insights automáticos"),
                html.Ul(
                    [html.Li(x) for x in summary.get("insights", [])],
                    style={"lineHeight": "1.8"},
                ),
            ],
            style={
                "padding": "18px",
                "border": "1px solid #ddd",
                "borderRadius": "12px",
                "backgroundColor": "#fcfcfc",
                "marginBottom": "24px",
            },
        ),

        html.Div(
            [
                html.H2("Tabla filtrada"),
                dash_table.DataTable(
                    id="data-table",
                    page_size=12,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "padding": "6px",
                        "fontFamily": "Arial",
                        "fontSize": "13px",
                    },
                    style_header={"fontWeight": "bold"},
                ),
            ]
        ),

        dcc.Markdown(
            """
**Interpretación sugerida**

- Un mayor puntaje de adicción se asocia con peor salud mental, más conflictos por redes y más horas de uso.
- Las métricas predictivas son muy altas; conviene validar si la base es observacional real o sintética.
- El dashboard prioriza interpretación y exploración dinámica para facilitar decisiones sobre bienestar estudiantil.
            """
        ),
    ],
    style={
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "24px",
        "fontFamily": "Arial, sans-serif",
    },
)


@app.callback(
    Output("histogram", "figure"),
    Output("scatter", "figure"),
    Output("heatmap", "figure"),
    Output("boxplot", "figure"),
    Output("data-table", "data"),
    Output("data-table", "columns"),
    Input("platform-filter", "value"),
    Input("academic-filter", "value"),
    Input("age-slider", "value"),
)
def update_dashboard(platform_value, academic_value, age_range):
    dff = df.copy()

    if age_range and len(age_range) == 2:
        age_low = int(age_range[0])
        age_high = int(age_range[1])
        dff = dff[(dff["Age"] >= age_low) & (dff["Age"] <= age_high)]

    if platform_value != "ALL" and "Most_Used_Platform" in dff.columns:
        dff = dff[dff["Most_Used_Platform"].astype(str) == str(platform_value)]

    if academic_value != "ALL" and "Affects_Academic_Performance" in dff.columns:
        dff = dff[dff["Affects_Academic_Performance"].astype(str) == str(academic_value)]

    if dff.empty:
        empty_hist = empty_figure("Distribución de Addicted_Score")
        empty_scatter = empty_figure("Adicción vs salud mental")
        empty_heat = empty_figure("Correlación de Pearson")
        empty_box = empty_figure("Adicción por plataforma")
        return empty_hist, empty_scatter, empty_heat, empty_box, [], []

    hist = px.histogram(
        dff,
        x="Addicted_Score",
        color="Affects_Academic_Performance" if "Affects_Academic_Performance" in dff.columns else None,
        marginal="box",
        nbins=10,
        title="Distribución de Addicted_Score",
        template="plotly_white",
    )

    if {"Mental_Health_Score", "Addicted_Score"}.issubset(dff.columns):
        scatter = px.scatter(
            dff,
            x="Mental_Health_Score",
            y="Addicted_Score",
            color="Most_Used_Platform" if "Most_Used_Platform" in dff.columns else None,
            size="Avg_Daily_Usage_Hours" if "Avg_Daily_Usage_Hours" in dff.columns else None,
            hover_data=[
                c for c in ["Age", "Sleep_Hours_Per_Night", "Country", "Relationship_Status"]
                if c in dff.columns
            ],
            title="Adicción vs salud mental",
            template="plotly_white",
        )
    else:
        scatter = empty_figure("Adicción vs salud mental", "Faltan columnas necesarias para este gráfico.")

    corr_cols = [c for c in numeric_cols if c in dff.columns]
    corr_cols = [c for c in corr_cols if pd.api.types.is_numeric_dtype(dff[c])]

    if len(corr_cols) >= 2:
        corr = dff[corr_cols].corr(numeric_only=True)
        heat = px.imshow(
            corr.round(2),
            text_auto=True,
            aspect="auto",
            title="Correlación de Pearson",
            template="plotly_white",
        )
    else:
        heat = empty_figure("Correlación de Pearson", "No hay suficientes variables numéricas para calcular correlaciones.")

    if {"Most_Used_Platform", "Addicted_Score"}.issubset(dff.columns):
        box = px.box(
            dff,
            x="Most_Used_Platform",
            y="Addicted_Score",
            color="Affects_Academic_Performance" if "Affects_Academic_Performance" in dff.columns else None,
            points="all",
            title="Adicción por plataforma",
            template="plotly_white",
        )
    else:
        box = empty_figure("Adicción por plataforma", "Faltan columnas necesarias para este gráfico.")

    table_data = dff.to_dict("records")
    table_cols = [{"name": str(c), "id": str(c)} for c in dff.columns]

    return hist, scatter, heat, box, table_data, table_cols


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
