from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis_utils import (
    MODEL_B_CATEGORICAL,
    MODEL_B_NUMERIC,
    NUMERIC_COLUMNS,
    VARIABLE_DESCRIPTIONS,
    breusch_pagan_table,
    clean_dataset,
    coefficient_table,
    descriptive_statistics,
    hypothesis_tests,
    load_student_data,
    model_metrics_table,
    outlier_summary,
    prediction_defaults,
    train_models,
    vif_table,
)

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
VIS_DIR = ASSETS_DIR / "visuals"

st.set_page_config(
    page_title="DATA 200 | Student Performance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2.2rem; max-width: 1450px;}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #0B1F3A 0%, #123B66 100%);}
    [data-testid="stSidebar"] * {color: #F5F9FF;}
    .hero {
        padding: 2.1rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #0B1F3A 0%, #134E8A 58%, #2563EB 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 18px 45px rgba(20, 52, 94, 0.18);
    }
    .hero h1 {font-size: 2.25rem; margin: 0 0 .5rem 0;}
    .hero p {font-size: 1.05rem; opacity: .92; max-width: 920px; margin: 0;}
    .metric-card {
        background: white; border: 1px solid #E5EAF2; border-radius: 16px;
        padding: 1rem 1.1rem; box-shadow: 0 8px 24px rgba(25, 48, 80, .07);
        min-height: 112px;
    }
    .metric-label {color:#667085; font-size:.82rem; text-transform:uppercase; letter-spacing:.04em;}
    .metric-value {color:#102A43; font-size:1.65rem; font-weight:750; margin-top:.3rem;}
    .section-card {
        background: white; border: 1px solid #E5EAF2; border-radius: 18px;
        padding: 1.2rem 1.3rem; box-shadow: 0 8px 24px rgba(25, 48, 80, .05);
        margin-bottom: .8rem; color: #102A43;
    }
    .insight {
        border-left: 5px solid #2563EB; background:#EEF5FF; border-radius:10px;
        padding:.85rem 1rem; margin:.6rem 0; color:#102A43;
    }
    .warning-note {
        border-left: 5px solid #F59E0B; background:#FFF8E6; border-radius:10px;
        padding:.85rem 1rem; margin:.6rem 0; color:#102A43;
    }
    .small-muted {color:#667085; font-size:.9rem;}
    div[data-testid="stDataFrame"] {border: 1px solid #E5EAF2; border-radius: 12px; overflow:hidden;}
    .team-card {background:white;border:1px solid #E5EAF2;border-radius:16px;padding:1rem;min-height:180px;color:#102A43;}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data + model loading
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(uploaded_bytes: bytes | None):
    uploaded_obj = io.BytesIO(uploaded_bytes) if uploaded_bytes else None
    raw, source = load_student_data(uploaded_obj)
    clean, cleaning = clean_dataset(raw)
    return raw, clean, cleaning, source


@st.cache_resource(show_spinner=False)
def get_models(df: pd.DataFrame):
    return train_models(df)


st.session_state.setdefault("active_dataset_bytes", None)
st.session_state.setdefault("dataset_deploy_error", None)
st.session_state.setdefault("dataset_deploy_error_bytes", None)

with st.sidebar:
    st.markdown("## 🎓 DATA 200")
    st.caption("Applied Statistical Analysis")
    st.markdown("---")
    uploaded = st.file_uploader(
        "Optional dataset upload",
        type=["csv"],
        help="Upload the UCI Portuguese student-performance CSV, then deploy it to use it across the dashboard.",
        key="dataset_upload",
    )
    uploaded_bytes = uploaded.getvalue() if uploaded else None

    deploy_clicked = st.button(
        "Deploy data",
        type="primary",
        width="stretch",
        disabled=uploaded_bytes is None,
        help="Validate and apply the selected CSV to every dashboard section.",
    )
    if deploy_clicked and uploaded_bytes is not None:
        try:
            _, _, _, uploaded_source = get_data(uploaded_bytes)
            if uploaded_source != "Uploaded CSV":
                raise ValueError(
                    "This file does not match the required student-performance CSV schema. "
                    "Upload the UCI Portuguese student dataset with all required columns."
                )
        except Exception as exc:
            st.session_state["dataset_deploy_error"] = str(exc)
            st.session_state["dataset_deploy_error_bytes"] = uploaded_bytes
        else:
            st.session_state["active_dataset_bytes"] = uploaded_bytes
            st.session_state["dataset_deploy_error"] = None
            st.session_state["dataset_deploy_error_bytes"] = None

    if uploaded_bytes is not None and uploaded_bytes != st.session_state["active_dataset_bytes"]:
        st.info("CSV selected. Click **Deploy data** to apply it across the dashboard.")
    elif st.session_state["active_dataset_bytes"] is not None:
        st.success("Uploaded dataset is deployed across the dashboard.")

    if (
        uploaded_bytes is not None
        and uploaded_bytes == st.session_state["dataset_deploy_error_bytes"]
        and st.session_state["dataset_deploy_error"]
    ):
        st.error("Could not deploy this CSV. The currently deployed dataset is unchanged.")
        st.caption(st.session_state["dataset_deploy_error"])

    if st.session_state["active_dataset_bytes"] is not None and st.button(
        "Restore default dataset", width="stretch"
    ):
        st.session_state["active_dataset_bytes"] = None
        st.session_state["dataset_deploy_error"] = None
        st.session_state["dataset_deploy_error_bytes"] = None

try:
    raw_df, df, cleaning_summary, data_source = get_data(
        st.session_state["active_dataset_bytes"]
    )
except Exception as exc:
    st.error("The application could not load the actual student dataset.")
    st.code(str(exc))
    st.info("Upload the real `student-por.csv` file in the sidebar. No dummy data will be substituted.")
    st.stop()

models = get_models(df)
metrics_df = model_metrics_table(models)
model_a_name = "Model A – Explanatory (without G1/G2)"
model_b_name = "Model B – Predictive (with G1/G2)"
model_a = models[model_a_name]
model_b = models[model_b_name]

# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.success(f"Dataset loaded: {len(df)} rows")
    st.caption(f"Source: {data_source}")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🗂️ Dataset Overview",
            "📊 Exploratory Data Analysis",
            "🧹 Data Cleaning",
            "📐 Statistical Analysis",
            "🤖 Models",
            "📈 Model Evaluation",
            "🔮 Prediction",
            "💡 Insights & Conclusion",
            "👥 Team Members",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Team United • University Project Demo")

# -----------------------------------------------------------------------------
# Reusable components
# -----------------------------------------------------------------------------
def metric_card(label: str, value: str):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def image_path(filename: str) -> Path:
    nested_path = VIS_DIR / filename
    root_path = BASE_DIR / filename
    return nested_path if nested_path.exists() else root_path


VISUALS = {
    "1. Final Grade Distribution": {
        "file": "01_final_grade_distribution.png",
        "finding": "Most final grades are concentrated around the middle of the 0–20 scale, especially around 10–14. The distribution shows the overall performance pattern, but it does not explain why students received those grades.",
    },
    "2. Study Time vs Final Grade": {
        "file": "02_study_time_vs_grade.png",
        "finding": "Average final grade generally rises across the lower study-time categories. This is evidence of an association, not proof that study time alone causes higher grades.",
    },
    "3. Absences vs Final Grade": {
        "file": "03_absences_vs_grade.png",
        "finding": "The relationship between absences and final grade is relatively weak and noisy. Some high-absence groups contain very few students, so isolated points should be interpreted carefully.",
    },
    "4. G2 vs G3": {
        "file": "04_G2_vs_G3.png",
        "finding": "Second-period grade (G2) has a very strong positive relationship with final grade (G3), making previous academic performance a powerful predictor of final performance.",
    },
    "5. Correlation Heatmap": {
        "file": "05_correlation_heatmap.png",
        "finding": "G1 and G2 have the strongest correlations with G3. Previous failures are negatively associated with G3. The heatmap also helps identify relationships among predictors before regression modeling.",
    },
    "6. Family Support vs Grade": {
        "file": "06_family_support_grade.png",
        "finding": "Students with and without reported family educational support show similar grade distributions. The difference should be evaluated statistically rather than judged from the graph alone.",
    },
    "7. School Comparison": {
        "file": "07_school_comparison.png",
        "finding": "Average final grades differ between the two schools in this dataset. Because this is observational data, the difference should not automatically be interpreted as a school effect.",
    },
    "8. Higher Education Intention": {
        "file": "08_higher_education_intent.png",
        "finding": "Students who reported an intention to pursue higher education have a higher average final grade in this dataset. This is an association and may reflect other related factors.",
    },
    "9. Previous Failures vs Grade": {
        "file": "09_failures_vs_grade.png",
        "finding": "Students with no previous class failures have a higher average final grade than students with one or more failures. Failure history is one of the stronger non-grade predictors in the dataset.",
    },
    "10. Mother's Education vs Grade": {
        "file": "10_mother_education_vs_grade.png",
        "finding": "Mean final grade tends to be higher at higher levels of mother's education, although group sizes and other family variables may also matter.",
    },
    "11. Internet Access and Grade Bands": {
        "file": "11_internet_grade_bands.png",
        "finding": "The grade-band composition differs between students with and without home internet access. The graph is descriptive and does not establish that internet access causes the difference.",
    },
    "12. Weekend Alcohol vs Grade": {
        "file": "12_weekend_alcohol_vs_grade.png",
        "finding": "Average final grade trends downward as reported weekend alcohol consumption increases. The pattern is observational and may be influenced by other behavioral or demographic variables.",
    },
}


def interactive_visual(title: str):
    """Return an interactive Plotly version of the selected verified visual."""
    if title.startswith("1."):
        counts = df["G3"].value_counts().sort_index().rename_axis("G3").reset_index(name="Students")
        return px.bar(counts, x="G3", y="Students", title="Interactive: Final Grade Distribution")

    if title.startswith("2."):
        labels = {1: "<2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: ">10 hours"}
        temp = df.copy()
        temp["Study time"] = temp["studytime"].map(labels)
        return px.box(temp, x="Study time", y="G3", points="outliers", title="Interactive: Study Time vs G3")

    if title.startswith("3."):
        temp = df.groupby("absences")["G3"].agg(["mean", "count"]).reset_index()
        return px.scatter(temp, x="absences", y="mean", size="count", hover_data=["count"],
                          labels={"mean": "Mean G3", "absences": "Absences"},
                          title="Interactive: Absences vs Mean Final Grade")

    if title.startswith("4."):
        return px.scatter(df, x="G2", y="G3", trendline="ols", opacity=.55,
                          title="Interactive: G2 vs G3", labels={"G2":"Second-period grade", "G3":"Final grade"})

    if title.startswith("5."):
        corr = df[NUMERIC_COLUMNS].corr()
        return px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                         title="Interactive: Correlation Heatmap")

    if title.startswith("6."):
        return px.box(df, x="famsup", y="G3", points="outliers",
                      labels={"famsup":"Family support", "G3":"Final grade"},
                      title="Interactive: Family Support vs G3")

    if title.startswith("7."):
        temp = df.groupby("school")["G3"].mean().reset_index()
        return px.bar(temp, x="school", y="G3", text_auto=".2f", range_y=[0,20],
                      title="Interactive: Mean G3 by School")

    if title.startswith("8."):
        temp = df.groupby("higher")["G3"].mean().reset_index()
        return px.bar(temp, x="higher", y="G3", text_auto=".2f", range_y=[0,20],
                      labels={"higher":"Plans higher education"},
                      title="Interactive: Higher-Education Intention vs G3")

    if title.startswith("9."):
        temp = df.groupby("failures")["G3"].agg(["mean", "count"]).reset_index()
        return px.bar(temp, x="failures", y="mean", text_auto=".2f", range_y=[0,20],
                      hover_data=["count"], labels={"mean":"Mean G3"},
                      title="Interactive: Previous Failures vs G3")

    if title.startswith("10."):
        temp = df.groupby("Medu")["G3"].agg(["mean", "count"]).reset_index()
        return px.line(temp, x="Medu", y="mean", markers=True,
                       hover_data=["count"], labels={"mean":"Mean G3", "Medu":"Mother's education (0–4)"},
                       title="Interactive: Mother's Education vs G3")

    if title.startswith("11."):
        temp = df[["internet", "G3"]].copy()
        temp["Grade band"] = pd.cut(temp["G3"], bins=[-1,9,14,20], labels=["Below 10", "10–14", "15+"])
        group = temp.groupby(["internet", "Grade band"], observed=False).size().reset_index(name="count")
        totals = group.groupby("internet")["count"].transform("sum")
        group["Percent"] = group["count"] / totals * 100
        return px.bar(group, x="internet", y="Percent", color="Grade band", barmode="stack",
                      labels={"internet":"Home internet access"},
                      title="Interactive: Grade Bands by Internet Access")

    temp = df.groupby("Walc")["G3"].agg(["mean", "count"]).reset_index()
    return px.line(temp, x="Walc", y="mean", markers=True, hover_data=["count"],
                   labels={"mean":"Mean G3", "Walc":"Weekend alcohol level"},
                   title="Interactive: Weekend Alcohol vs G3")


# -----------------------------------------------------------------------------
# HOME
# -----------------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <h1>Predicting Student Academic Performance</h1>
            <p>DATA 200 Applied Statistical Analysis • Team United<br>
            An end-to-end statistical project combining real-world student data, exploratory analysis,
            hypothesis testing, multiple linear regression, model validation, and interactive prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Students", f"{len(df):,}")
    with c2: metric_card("Variables", str(df.shape[1]))
    with c3: metric_card("Target", "G3 (0–20)")
    with c4: metric_card("Missing values", f"{int(df.isna().sum().sum())}")

    st.markdown("### Project objective")
    objective_card = """
        <div class="section-card">
        <b>Main research question</b><br><br>
        What academic, behavioral, family, and school-related factors are significantly associated
        with students' final academic performance, and how accurately can multiple linear regression
        predict a student's final grade?<br><br>
        <b>What this application demonstrates</b><br>
        • transparent data-quality checks<br>
        • 12 verified EDA visualizations<br>
        • hypothesis tests and statistical inference<br>
        • explanatory vs predictive regression<br>
        • train/test evaluation and residual diagnostics<br>
        • real-time student grade prediction
        </div>
        """
    hero = ASSETS_DIR / "project_hero.png"
    if hero.exists():
        left, right = st.columns([1.25, 1])
        with left:
            st.markdown(objective_card, unsafe_allow_html=True)
        with right:
            st.image(str(hero), width="stretch")
    else:
        st.markdown(objective_card, unsafe_allow_html=True)

    st.markdown(
        '<div class="warning-note"><b>Interpretation rule:</b> This is observational data. '
        'Associations and predictions should not be described as proof of causation.</div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# DATASET OVERVIEW
# -----------------------------------------------------------------------------
elif page == "🗂️ Dataset Overview":
    st.title("Dataset Overview")
    st.write(
        "We use the **UCI Student Performance** Portuguese-course dataset. The target variable is `G3`, "
        "the student's final grade on a 0–20 scale."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Rows", f"{len(df):,}")
    with c2: metric_card("Columns", str(df.shape[1]))
    with c3: metric_card("Mean G3", f"{df['G3'].mean():.2f}")
    with c4: metric_card("Median G3", f"{df['G3'].median():.0f}")

    st.subheader("Explore the data")
    col1, col2, col3 = st.columns(3)
    school_filter = col1.multiselect("School", sorted(df["school"].unique()), default=sorted(df["school"].unique()))
    sex_filter = col2.multiselect("Sex", sorted(df["sex"].unique()), default=sorted(df["sex"].unique()))
    grade_range = col3.slider("Final grade range", int(df.G3.min()), int(df.G3.max()), (int(df.G3.min()), int(df.G3.max())))

    filtered = df[
        df["school"].isin(school_filter)
        & df["sex"].isin(sex_filter)
        & df["G3"].between(grade_range[0], grade_range[1])
    ]
    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.dataframe(filtered, width="stretch", height=360)

    tab1, tab2 = st.tabs(["Summary statistics", "Variable dictionary"])
    with tab1:
        st.dataframe(descriptive_statistics(df).round(3), width="stretch")
    with tab2:
        dictionary = pd.DataFrame({
            "Variable": list(VARIABLE_DESCRIPTIONS.keys()),
            "Type": ["Numeric" if c in NUMERIC_COLUMNS else "Categorical" for c in VARIABLE_DESCRIPTIONS],
            "Description": list(VARIABLE_DESCRIPTIONS.values()),
            "Role": ["Target" if c == "G3" else ("Predictor" if c in MODEL_B_NUMERIC + MODEL_B_CATEGORICAL else "Context") for c in VARIABLE_DESCRIPTIONS],
        })
        st.dataframe(dictionary, width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# EDA
# -----------------------------------------------------------------------------
elif page == "📊 Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.write("Explore 12 charts generated from the currently deployed dataset. Choose a chart below; each visualization updates when you deploy another CSV.")

    selected = st.selectbox("Choose a visualization", list(VISUALS.keys()))
    info = VISUALS[selected]
    st.plotly_chart(
        interactive_visual(selected),
        width="stretch",
        height=520,
        alt=f"{selected}, generated from the active student dataset",
    )
    original_image = image_path(info["file"])
    if original_image.exists():
        with st.expander("View supplied static image"):
            st.image(str(original_image), width="stretch", alt=selected)

    st.markdown(f'<div class="insight"><b>Key finding:</b> {info["finding"]}</div>', unsafe_allow_html=True)

    with st.expander(
        "View all 12 interactive visualizations",
        key="eda_gallery",
        on_change="rerun",
    ) as gallery:
        if gallery.open:
            keys = list(VISUALS.keys())
            for i in range(0, len(keys), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(keys):
                        key = keys[i+j]
                        with col:
                            st.markdown(f"**{key}**")
                            st.plotly_chart(
                                interactive_visual(key),
                                width="stretch",
                                key=f"eda_gallery_{i + j}",
                                alt=f"{key}, generated from the active student dataset",
                            )
                            st.caption(VISUALS[key]["finding"])

# -----------------------------------------------------------------------------
# DATA CLEANING
# -----------------------------------------------------------------------------
elif page == "🧹 Data Cleaning":
    st.title("Data Cleaning Process")
    st.write(
        "The cleaning process was designed to protect valid observations. We check missing values, data types, exact duplicates, ranges, and potential outliers. "
        "Outliers are **flagged for review rather than automatically deleted**."
    )

    st.subheader("Before vs after")
    st.dataframe(cleaning_summary, width="stretch", hide_index=True)

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Missing values", str(int(df.isna().sum().sum())))
    with c2: metric_card("Exact duplicates", str(int(df.duplicated().sum())))
    with c3: metric_card("Rows retained", f"{len(df):,}")

    st.subheader("Missing values by variable")
    missing = df.isna().sum().sort_values(ascending=False).reset_index()
    missing.columns = ["Variable", "Missing"]
    st.plotly_chart(px.bar(missing, x="Variable", y="Missing", title="Missing Values by Variable"), width="stretch")

    st.subheader("Potential outliers (IQR rule)")
    outliers = outlier_summary(df)
    st.dataframe(outliers.round(3), width="stretch", hide_index=True)

    variable = st.selectbox("Inspect a numeric variable", NUMERIC_COLUMNS, index=NUMERIC_COLUMNS.index("absences"))
    st.plotly_chart(px.box(df, y=variable, points="outliers", title=f"Outlier Review: {variable}"), width="stretch")
    st.markdown(
        '<div class="warning-note"><b>Important:</b> A statistical outlier is not automatically a data error. '
        'For example, a student with many absences may be unusual but still valid. We retain plausible values unless there is evidence that the record is incorrect.</div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# STATISTICAL ANALYSIS
# -----------------------------------------------------------------------------
elif page == "📐 Statistical Analysis":
    st.title("Statistical Analysis")
    st.write("We use descriptive statistics, correlation, hypothesis testing, ANOVA/t-tests, regression inference, VIF, and residual diagnostics.")

    st.subheader("Hypothesis tests")
    tests = hypothesis_tests(df)
    display_tests = tests.copy()
    display_tests["Statistic"] = display_tests["Statistic"].map(lambda x: f"{x:.4f}")
    display_tests["P-value"] = display_tests["P-value"].map(lambda x: f"{x:.6f}")
    st.dataframe(display_tests, width="stretch", hide_index=True)
    st.caption("Decision rule: p < 0.05 → Reject H₀; otherwise → Fail to reject H₀.")

    st.subheader("Regression inference")
    chosen_model = st.radio("Choose regression specification", [model_a_name, model_b_name], horizontal=True)
    chosen = models[chosen_model]
    a, b, c = st.columns(3)
    with a: metric_card("OLS R²", f"{chosen.ols_model.rsquared:.3f}")
    with b: metric_card("Adjusted R²", f"{chosen.ols_model.rsquared_adj:.3f}")
    with c: metric_card("Observations", f"{int(chosen.ols_model.nobs):,}")

    coef = coefficient_table(chosen)
    only_sig = st.checkbox("Show only statistically significant terms (p < 0.05)", value=False)
    if only_sig:
        coef = coef[coef["Significant (p<0.05)"]]
    st.dataframe(coef.round(4), width="stretch", hide_index=True, height=400)

    tab1, tab2 = st.tabs(["Multicollinearity (VIF)", "Homoscedasticity test"])
    with tab1:
        st.dataframe(vif_table(df).round(3), width="stretch", hide_index=True)
        st.caption("VIF is used as a diagnostic. High values suggest predictors may contain overlapping information.")
    with tab2:
        bp = breusch_pagan_table(model_b)
        st.dataframe(bp.round(6), width="stretch", hide_index=True)
        st.caption("The Breusch–Pagan test checks whether residual variance is constant across fitted values.")

# -----------------------------------------------------------------------------
# MODELS
# -----------------------------------------------------------------------------
elif page == "🤖 Models":
    st.title("Machine Learning / Statistical Models")
    st.write("The project intentionally stays focused on interpretable statistical modeling rather than using unnecessarily complex machine-learning algorithms.")

    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="section-card">
            <h3>Model A — Explanatory</h3>
            <b>Multiple Linear Regression without G1 and G2</b><br><br>
            <b>Why:</b> This model asks which behavioral, family, school, and background factors are associated with final performance without allowing earlier grades to dominate the model.<br><br>
            <b>Examples:</b> study time, failures, absences, parental education, family support, school support, higher-education intention, internet access, school, sex, and address.<br><br>
            <b>Use:</b> statistical explanation and coefficient/p-value interpretation.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="section-card">
            <h3>Model B — Predictive</h3>
            <b>Multiple Linear Regression with G1 and G2</b><br><br>
            <b>Why:</b> G1 and G2 represent earlier academic performance and have strong relationships with G3. Including them gives a more realistic prediction model when prior grades are available.<br><br>
            <b>Use:</b> real-time prediction and out-of-sample performance evaluation.<br><br>
            <b>Difference:</b> Model B focuses more on prediction, while Model A is more useful for explaining non-grade factors.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Regression form")
    st.latex(r"G3 = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon")
    st.write("Each coefficient estimates the expected change in final grade associated with a one-unit change in a predictor, holding the other included variables constant.")

    st.subheader("Why not Logistic Regression as the main model?")
    st.info("Our target, G3, is numerical (0–20). Multiple Linear Regression directly models the final score. Logistic Regression would be more appropriate if we changed the target into categories such as pass/fail, which would answer a different research question.")

# -----------------------------------------------------------------------------
# MODEL EVALUATION
# -----------------------------------------------------------------------------
elif page == "📈 Model Evaluation":
    st.title("Model Evaluation")
    st.write("Both models use the same reproducible 80/20 train/test split (`random_state=42`). Metrics below are calculated from the actual data each time the application loads.")

    display_metrics = metrics_df.copy()
    numeric_cols = ["Test R²", "OLS R²", "OLS Adjusted R²", "MAE", "RMSE"]
    display_metrics[numeric_cols] = display_metrics[numeric_cols].round(4)
    st.dataframe(display_metrics, width="stretch", hide_index=True)

    m = metrics_df.melt(id_vars="Model", value_vars=["Test R²", "MAE", "RMSE"], var_name="Metric", value_name="Value")
    st.plotly_chart(px.bar(m, x="Metric", y="Value", color="Model", barmode="group", title="Model Performance Comparison"), width="stretch")

    st.subheader("Actual vs predicted — Model B")
    pred_df = pd.DataFrame({"Actual G3": model_b.y_test.values, "Predicted G3": model_b.predictions})
    fig = px.scatter(pred_df, x="Actual G3", y="Predicted G3", opacity=.7, title="Actual vs Predicted Final Grades")
    lo = min(pred_df.min())
    hi = max(pred_df.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction", line=dict(dash="dash")))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Residual diagnostics — Model B")
    residuals = model_b.y_test.values - model_b.predictions
    residual_df = pd.DataFrame({"Predicted": model_b.predictions, "Residual": residuals})
    fig2 = px.scatter(residual_df, x="Predicted", y="Residual", opacity=.7, title="Residuals vs Predicted Values")
    fig2.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig2, width="stretch")

    st.markdown(
        """
        **Metric meanings**
        - **Test R²:** proportion of variation in unseen test grades explained by the model.
        - **OLS Adjusted R²:** full-sample statistical fit adjusted for the number of predictors.
        - **MAE:** average absolute prediction error in grade points.
        - **RMSE:** error measure that penalizes larger prediction errors more strongly.
        """
    )

# -----------------------------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------------------------
elif page == "🔮 Prediction":
    st.title("Real-Time Final Grade Prediction")
    st.write("This page uses **Model B**, trained on the actual UCI dataset. No dummy prediction formula is used.")

    defaults = prediction_defaults(df, model_b)
    mode = st.radio("Input mode", ["Quick", "Advanced"], horizontal=True, help="Quick mode lets you change the most important inputs; other fields use dataset medians/modes.")
    values = defaults.copy()

    key_numeric = ["G1", "G2", "studytime", "failures", "absences", "Medu", "Fedu"]
    key_categorical = ["higher", "internet", "famsup"]

    if mode == "Quick":
        st.caption("Fields not shown below are automatically set to the dataset median (numeric) or mode (categorical).")
        cols = st.columns(2)
        for i, col in enumerate(key_numeric):
            with cols[i % 2]:
                min_v, max_v = float(df[col].min()), float(df[col].max())
                step = 1.0
                values[col] = st.slider(VARIABLE_DESCRIPTIONS.get(col, col), min_v, max_v, float(defaults[col]), step=step, key=f"q_{col}")
        for i, col in enumerate(key_categorical):
            with cols[i % 2]:
                options = sorted(df[col].dropna().unique().tolist())
                default_index = options.index(defaults[col]) if defaults[col] in options else 0
                values[col] = st.selectbox(VARIABLE_DESCRIPTIONS.get(col, col), options, index=default_index, key=f"q_{col}")
    else:
        st.caption("Advanced mode exposes every predictor used by Model B.")
        with st.form("advanced_prediction_form"):
            left, right = st.columns(2)
            for i, col in enumerate(model_b.numeric_features):
                with (left if i % 2 == 0 else right):
                    min_v, max_v = float(df[col].min()), float(df[col].max())
                    values[col] = st.number_input(
                        VARIABLE_DESCRIPTIONS.get(col, col),
                        min_value=min_v,
                        max_value=max_v,
                        value=float(defaults[col]),
                        step=1.0,
                        key=f"a_{col}",
                    )
            for i, col in enumerate(model_b.categorical_features):
                with (left if i % 2 == 0 else right):
                    options = sorted(df[col].dropna().unique().tolist())
                    default_index = options.index(defaults[col]) if defaults[col] in options else 0
                    values[col] = st.selectbox(VARIABLE_DESCRIPTIONS.get(col, col), options, index=default_index, key=f"a_{col}")
            submitted = st.form_submit_button("Prepare prediction")

    prediction_row = pd.DataFrame([{c: values[c] for c in model_b.numeric_features + model_b.categorical_features}])

    if st.button("Predict Final Grade", type="primary", width="stretch"):
        try:
            raw_prediction = float(model_b.pipeline.predict(prediction_row)[0])
            bounded_prediction = float(np.clip(raw_prediction, 0, 20))
            st.markdown(
                f'<div class="hero"><h1>Predicted Final Grade: {bounded_prediction:.2f} / 20</h1>'
                f'<p>Raw regression estimate: {raw_prediction:.2f}. The displayed score is bounded to the documented 0–20 grade scale.</p></div>',
                unsafe_allow_html=True,
            )
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bounded_prediction,
                number={"suffix":" / 20"},
                gauge={"axis":{"range":[0,20]}},
                title={"text":"Model B Prediction"},
            ))
            st.plotly_chart(gauge, width="stretch")
            st.info("This prediction is an estimate from an observational statistical model. It is not a guaranteed future grade and should not be used as a high-stakes decision by itself.")
        except Exception as exc:
            st.error("Prediction failed. Please check the selected input values.")
            st.code(str(exc))

# -----------------------------------------------------------------------------
# INSIGHTS
# -----------------------------------------------------------------------------
elif page == "💡 Insights & Conclusion":
    st.title("Key Insights and Conclusion")

    corr = df[NUMERIC_COLUMNS].corr()["G3"].drop("G3")
    strongest_pos = corr.idxmax()
    strongest_neg = corr.idxmin()
    test_table = hypothesis_tests(df)
    significant_count = int(test_table["Significant at α=0.05"].sum())

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Strongest positive numeric relationship", f"{strongest_pos} ({corr[strongest_pos]:.3f})")
    with c2: metric_card("Strongest negative numeric relationship", f"{strongest_neg} ({corr[strongest_neg]:.3f})")
    with c3: metric_card("Significant planned tests", f"{significant_count} / {len(test_table)}")

    st.markdown("### Main conclusions")
    st.markdown(
        f"""
        1. **Previous academic performance matters strongly.** In this dataset, `{strongest_pos}` has the strongest positive numeric correlation with final grade.
        2. **Previous failures are important.** Failure history is negatively related to final performance and is useful in the explanatory model.
        3. **Study behavior and background factors add context.** Study time, parental education, higher-education intention, school, internet access, and other variables show patterns worth testing in a multivariable framework.
        4. **Prediction and explanation are different goals.** Model A excludes G1/G2 to study non-grade factors; Model B includes them for stronger predictive performance.
        5. **Causation cannot be claimed.** The dataset is observational, so significant associations do not prove that changing one variable will cause a change in grades.
        """
    )

    st.markdown("### Limitations")
    st.write(
        "The dataset represents students from two Portuguese secondary schools and may not generalize to all students or countries. "
        "Some variables are self-reported or categorical, unmeasured factors may exist, and linear regression depends on model assumptions. "
        "Predictions therefore contain uncertainty."
    )

    st.markdown("### Final takeaway")
    st.success("The project demonstrates a complete statistical workflow: real data → cleaning → EDA → hypothesis testing → interpretable regression → validation → interactive prediction.")

# -----------------------------------------------------------------------------
# TEAM
# -----------------------------------------------------------------------------
else:
    st.title("Team Members")
    st.write("Team United — DATA 200 Applied Statistical Analysis")

    members = [
        ("Samrat Jung Shahi", "Project framing, dataset understanding, and data-quality presentation"),
        ("Saksham Smith Sunar", "Exploratory data analysis and visualization presentation"),
        ("Rishav Yadav", "Statistical analysis, modeling, and evaluation presentation"),
        ("Aryan Shrestha", "Streamlit application, integration, and final presentation support"),
    ]
    cols = st.columns(4)
    for col, (name, contribution) in zip(cols, members):
        with col:
            initials = "".join([p[0] for p in name.split()[:2]])
            st.markdown(
                f"""
                <div class="team-card">
                    <div style="font-size:2rem;font-weight:800;color:#2563EB;">{initials}</div>
                    <h3>{name}</h3>
                    <p>{contribution}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption("Contribution labels follow the project's presentation/work division and can be edited in `app.py` if your final team roles differ.")

    st.markdown("### Project technology")
    tech = pd.DataFrame({
        "Area": ["Data", "Statistics", "Modeling", "Visualization", "Web Application"],
        "Tools": ["Pandas, NumPy", "SciPy, Statsmodels", "Scikit-learn", "Plotly, Matplotlib, Seaborn", "Streamlit"],
    })
    st.dataframe(tech, width="stretch", hide_index=True)

st.markdown("---")
st.caption("DATA 200 • Student Performance Project • Results are computed from the loaded dataset; no dummy statistics or predictions are used.")
