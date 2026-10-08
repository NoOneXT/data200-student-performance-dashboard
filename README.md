# DATA 200: Student Performance Dashboard

A Streamlit application for exploring the UCI Student Performance dataset and comparing explanatory and predictive regression models.

## Project contents

```text
.
├── app.py                                      # Streamlit dashboard entry point
├── analysis_utils.py                           # Data loading, statistics, and model helpers
├── requirements.txt                            # Python dependencies
├── student-por.csv                             # Portuguese-course UCI dataset
├── 01_final_grade_distribution.png             # Supplied static chart
├── DATA200_Analysis_Colab.ipynb                # Analysis notebook
└── DATA200_TeamUnited_Final_Presentation_4_People_Equal.pptx
```

The local `.venv/`, Python caches, editor settings, and Streamlit secrets are excluded by `.gitignore`.

## Features

- Dataset overview, filters, data-quality summaries, and exploratory analysis.
- Interactive charts generated from the currently deployed dataset.
- Hypothesis tests, regression coefficients, confidence intervals, VIF, and residual diagnostics.
- Model A (without earlier grades) for explanatory analysis and Model B (with G1/G2) for prediction.
- CSV upload with validation and an explicit **Deploy data** action; an invalid upload does not replace the active data.

## Run locally

Requires Python 3.10 or newer. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Dataset

The bundled CSV is the Portuguese-language subset of the UCI Student Performance dataset. The app uses the bundled file when present and can also load a compatible CSV through the sidebar. The data is observational; associations and predictions do not establish causation.

## Deployment

Deploy through Streamlit Community Cloud by selecting this repository and setting `app.py` as the main file. Install dependencies from `requirements.txt`. No API keys are required.

## Team United

- Samrat Jung Shahi
- Saksham Smith Sunar
- Rishav Yadav
- Aryan Shrestha
