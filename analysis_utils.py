from __future__ import annotations

import io
import zipfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
GITHUB_FALLBACK_URL = "https://raw.githubusercontent.com/shiva154200/Datasets/main/student-por.csv"

REQUIRED_COLUMNS = {
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "G3"
}

NUMERIC_COLUMNS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"
]

MODEL_A_NUMERIC = [
    "studytime", "failures", "absences", "Medu", "Fedu", "traveltime",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health"
]
MODEL_A_CATEGORICAL = [
    "famsup", "schoolsup", "activities", "higher", "internet", "paid",
    "school", "sex", "address"
]
MODEL_B_NUMERIC = MODEL_A_NUMERIC + ["G1", "G2"]
MODEL_B_CATEGORICAL = MODEL_A_CATEGORICAL.copy()

VARIABLE_DESCRIPTIONS = {
    "school": "Student's school (GP or MS)",
    "sex": "Student sex",
    "age": "Student age in years",
    "address": "Home address type: urban or rural",
    "famsize": "Family size category",
    "Pstatus": "Parents' cohabitation status",
    "Medu": "Mother's education level (0–4)",
    "Fedu": "Father's education level (0–4)",
    "Mjob": "Mother's occupation",
    "Fjob": "Father's occupation",
    "reason": "Reason for choosing the school",
    "guardian": "Student's guardian",
    "traveltime": "Home-to-school travel time category (1–4)",
    "studytime": "Weekly study-time category (1–4)",
    "failures": "Number of previous class failures",
    "schoolsup": "Extra educational support from school",
    "famsup": "Family educational support",
    "paid": "Extra paid classes",
    "activities": "Extracurricular activities",
    "nursery": "Attended nursery school",
    "higher": "Wants to pursue higher education",
    "internet": "Internet access at home",
    "romantic": "In a romantic relationship",
    "famrel": "Quality of family relationships (1–5)",
    "freetime": "Free time after school (1–5)",
    "goout": "Going out with friends (1–5)",
    "Dalc": "Workday alcohol consumption (1–5)",
    "Walc": "Weekend alcohol consumption (1–5)",
    "health": "Current health status (1–5)",
    "absences": "Number of school absences",
    "G1": "First-period grade (0–20)",
    "G2": "Second-period grade (0–20)",
    "G3": "Final grade (0–20) — target variable",
}


def _read_csv_flexibly(content: bytes) -> pd.DataFrame:
    # UCI archive uses semicolons; common mirrors may use commas.
    text = content.decode("utf-8-sig")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep)


def load_student_data(uploaded_file=None, timeout: int = 20) -> Tuple[pd.DataFrame, str]:
    """Load actual UCI student performance data.

    Order: user upload -> official UCI ZIP -> GitHub mirror fallback.
    No synthetic or dummy data is ever created.
    """
    errors: List[str] = []

    project_dir = Path(__file__).resolve().parent
    local_csvs = [
        project_dir / "data" / "student-por.csv",
        project_dir / "student-por.csv",
    ]
    if uploaded_file is None:
        for local_csv in local_csvs:
            if local_csv.exists():
                try:
                    df = _read_csv_flexibly(local_csv.read_bytes())
                    _validate_dataset(df)
                    return df, "Bundled local student-por.csv"
                except Exception as exc:
                    errors.append(f"Local CSV ({local_csv.name}): {exc}")

    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue()
            df = _read_csv_flexibly(content)
            _validate_dataset(df)
            return df, "Uploaded CSV"
        except Exception as exc:
            errors.append(f"Uploaded CSV: {exc}")

    try:
        r = requests.get(UCI_ZIP_URL, timeout=timeout)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            candidates = [n for n in zf.namelist() if n.endswith("student-por.csv")]
            if not candidates:
                raise FileNotFoundError("student-por.csv not found in UCI archive")
            content = zf.read(candidates[0])
        df = _read_csv_flexibly(content)
        _validate_dataset(df)
        return df, "UCI Machine Learning Repository"
    except Exception as exc:
        errors.append(f"UCI: {exc}")

    try:
        r = requests.get(GITHUB_FALLBACK_URL, timeout=timeout)
        r.raise_for_status()
        df = _read_csv_flexibly(r.content)
        _validate_dataset(df)
        return df, "GitHub mirror of UCI student-por.csv"
    except Exception as exc:
        errors.append(f"GitHub fallback: {exc}")

    raise RuntimeError(
        "Could not load the actual dataset. Upload student-por.csv in the sidebar. "
        + " | ".join(errors)
    )


def _validate_dataset(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(df) < 100:
        raise ValueError("Dataset contains too few rows to be the selected UCI student dataset.")


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    before_rows = len(df)
    before_missing = int(df.isna().sum().sum())
    before_duplicates = int(df.duplicated().sum())

    cleaned = df.copy()
    # Convert known numeric fields; invalid values become NaN and are reported.
    for col in NUMERIC_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Exact duplicates are safe to remove. We do not automatically delete outliers.
    cleaned = cleaned.drop_duplicates().copy()

    after_rows = len(cleaned)
    after_missing = int(cleaned.isna().sum().sum())
    after_duplicates = int(cleaned.duplicated().sum())

    summary = pd.DataFrame({
        "Measure": ["Rows", "Missing values", "Exact duplicate rows"],
        "Before": [before_rows, before_missing, before_duplicates],
        "After": [after_rows, after_missing, after_duplicates],
    })
    return cleaned, summary


def outlier_summary(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    columns = columns or ["age", "studytime", "failures", "absences", "G1", "G2", "G3"]
    records = []
    for col in columns:
        series = df[col].dropna().astype(float)
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        count = int(((series < low) | (series > high)).sum())
        records.append({
            "Variable": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower bound": low,
            "Upper bound": high,
            "Flagged observations": count,
        })
    return pd.DataFrame(records)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    d = df[NUMERIC_COLUMNS].describe().T
    d.insert(2, "median", df[NUMERIC_COLUMNS].median())
    return d[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]]


def hypothesis_tests(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows = []

    r, p = stats.pearsonr(df["studytime"], df["G3"])
    rows.append(["Study time vs final grade", "Pearson correlation", r, p])

    r, p = stats.pearsonr(df["absences"], df["G3"])
    rows.append(["Absences vs final grade", "Pearson correlation", r, p])

    groups = [g["G3"].dropna().values for _, g in df.groupby("studytime")]
    f, p = stats.f_oneway(*groups)
    rows.append(["Final grade across study-time groups", "One-way ANOVA", f, p])

    yes = df.loc[df["famsup"] == "yes", "G3"].dropna()
    no = df.loc[df["famsup"] == "no", "G3"].dropna()
    t, p = stats.ttest_ind(yes, no, equal_var=False)
    rows.append(["Family support groups and final grade", "Welch t-test", t, p])

    out = pd.DataFrame(rows, columns=["Question", "Method", "Statistic", "P-value"])
    out["Decision"] = np.where(out["P-value"] < alpha, "Reject H₀", "Fail to reject H₀")
    out["Significant at α=0.05"] = out["P-value"] < alpha
    return out


def _formula(target: str, numeric: List[str], categorical: List[str]) -> str:
    terms = numeric + [f"C({c})" for c in categorical]
    return target + " ~ " + " + ".join(terms)


def _build_pipeline(numeric: List[str], categorical: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ])


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    numeric_features: List[str]
    categorical_features: List[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictions: np.ndarray
    test_r2: float
    mae: float
    rmse: float
    ols_model: object


def train_models(df: pd.DataFrame, random_state: int = 42, test_size: float = 0.20) -> Dict[str, ModelResult]:
    y = df["G3"].astype(float)
    all_features = sorted(set(MODEL_B_NUMERIC + MODEL_B_CATEGORICAL))
    valid = df[all_features + ["G3"]].dropna().copy()
    y = valid["G3"].astype(float)

    train_idx, test_idx = train_test_split(
        valid.index, test_size=test_size, random_state=random_state
    )

    specifications = {
        "Model A – Explanatory (without G1/G2)": (MODEL_A_NUMERIC, MODEL_A_CATEGORICAL),
        "Model B – Predictive (with G1/G2)": (MODEL_B_NUMERIC, MODEL_B_CATEGORICAL),
    }

    results: Dict[str, ModelResult] = {}
    for name, (num, cat) in specifications.items():
        X = valid[num + cat].copy()
        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]

        pipeline = _build_pipeline(num, cat)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        ols = smf.ols(_formula("G3", num, cat), data=valid).fit()

        results[name] = ModelResult(
            name=name,
            pipeline=pipeline,
            numeric_features=num,
            categorical_features=cat,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            predictions=pred,
            test_r2=float(r2_score(y_test, pred)),
            mae=float(mean_absolute_error(y_test, pred)),
            rmse=float(np.sqrt(mean_squared_error(y_test, pred))),
            ols_model=ols,
        )
    return results


def model_metrics_table(models: Dict[str, ModelResult]) -> pd.DataFrame:
    records = []
    for name, result in models.items():
        records.append({
            "Model": name,
            "Test R²": result.test_r2,
            "OLS R²": float(result.ols_model.rsquared),
            "OLS Adjusted R²": float(result.ols_model.rsquared_adj),
            "MAE": result.mae,
            "RMSE": result.rmse,
        })
    return pd.DataFrame(records)


def coefficient_table(model_result: ModelResult) -> pd.DataFrame:
    model = model_result.ols_model
    ci = model.conf_int()
    table = pd.DataFrame({
        "Term": model.params.index,
        "Coefficient": model.params.values,
        "Std. Error": model.bse.values,
        "t-statistic": model.tvalues.values,
        "P-value": model.pvalues.values,
        "CI Lower": ci.iloc[:, 0].values,
        "CI Upper": ci.iloc[:, 1].values,
    })
    table["Significant (p<0.05)"] = table["P-value"] < 0.05
    return table


def vif_table(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    columns = columns or MODEL_B_NUMERIC
    X = df[columns].dropna().astype(float).copy()
    # Add intercept manually to avoid artificially high VIF due to no constant.
    X.insert(0, "const", 1.0)
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        rows.append({"Variable": col, "VIF": variance_inflation_factor(X.values, i)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def breusch_pagan_table(model_result: ModelResult) -> pd.DataFrame:
    model = model_result.ols_model
    labels = ["LM statistic", "LM p-value", "F statistic", "F p-value"]
    values = het_breuschpagan(model.resid, model.model.exog)
    return pd.DataFrame({"Measure": labels, "Value": values})


def prediction_defaults(df: pd.DataFrame, result: ModelResult) -> Dict[str, object]:
    defaults: Dict[str, object] = {}
    for col in result.numeric_features:
        defaults[col] = float(df[col].median())
    for col in result.categorical_features:
        mode = df[col].mode(dropna=True)
        defaults[col] = mode.iloc[0] if not mode.empty else ""
    return defaults
