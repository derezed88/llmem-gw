"""
Stats Engine — statistical analysis for llmem-gw.

Accepts data from multiple sources (SQL query, CSV file, JSON, Google Drive),
normalizes to DataFrame, and runs statistical operations.
"""

import json
import io
import pandas as pd
import numpy as np
from typing import Optional


# ── Data Loading ─────────────────────────────────────────────

async def load_data(
    sql: str = "",
    csv_path: str = "",
    json_data: str = "",
    drive_file_id: str = "",
) -> pd.DataFrame:
    """Load data from one of the supported sources into a DataFrame."""

    if sql:
        from database import execute_sql
        raw = await execute_sql(sql)
        return _parse_db_result(raw)

    if csv_path:
        return pd.read_csv(csv_path)

    if json_data:
        parsed = json.loads(json_data)
        if isinstance(parsed, list):
            return pd.DataFrame(parsed)
        elif isinstance(parsed, dict):
            return pd.DataFrame(parsed)
        raise ValueError("JSON must be an array of objects or a dict of arrays")

    if drive_file_id:
        from google_drive import download_file_content
        content = await download_file_content(drive_file_id)
        # Try CSV first, then JSON
        try:
            return pd.read_csv(io.StringIO(content))
        except Exception:
            return pd.DataFrame(json.loads(content))

    raise ValueError("No data source provided. Use sql, csv_path, json_data, or drive_file_id.")


def _parse_db_result(raw: str) -> pd.DataFrame:
    """Parse the tabular text output from db_query into a DataFrame."""
    lines = raw.strip().split('\n')
    if len(lines) < 2:
        raise ValueError(f"Query returned no data: {raw}")

    # Header is first line, separator is second line (dashes)
    header_line = lines[0]
    # Find separator line (contains only dashes, spaces, +)
    sep_idx = 1
    for i, line in enumerate(lines[1:], 1):
        if set(line.strip()).issubset({'-', '+', ' '}):
            sep_idx = i
            break

    headers = [h.strip() for h in header_line.split('|')]
    headers = [h for h in headers if h]  # remove empty

    rows = []
    for line in lines[sep_idx + 1:]:
        if not line.strip() or line.strip().startswith('('):
            continue
        vals = [v.strip() for v in line.split('|')]
        vals = [v for v in vals if v != '']
        if len(vals) == len(headers):
            rows.append(vals)

    df = pd.DataFrame(rows, columns=headers)

    # Auto-convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    return df


# ── Statistical Operations ───────────────────────────────────

def descriptive_stats(df: pd.DataFrame, columns: list = None) -> dict:
    """Descriptive statistics for numeric columns."""
    if columns:
        df = df[columns]
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return {"error": "No numeric columns found"}

    result = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        result[col] = {
            "count": int(s.count()),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "25%": round(float(s.quantile(0.25)), 4),
            "50%": round(float(s.quantile(0.50)), 4),
            "75%": round(float(s.quantile(0.75)), 4),
            "max": round(float(s.max()), 4),
            "skew": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
        }
    return result


def correlation_matrix(df: pd.DataFrame, columns: list = None, method: str = "pearson") -> dict:
    """Correlation matrix for numeric columns."""
    if columns:
        df = df[columns]
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns for correlation"}

    corr = numeric.corr(method=method)
    # Round and convert to nested dict
    result = {}
    for col in corr.columns:
        result[col] = {k: round(v, 4) for k, v in corr[col].items()}
    return {"method": method, "matrix": result}


def ols_regression(df: pd.DataFrame, y_col: str, x_cols: list) -> dict:
    """OLS linear regression with diagnostics."""
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    subset = df[[y_col] + x_cols].dropna()
    if subset.shape[0] < len(x_cols) + 2:
        return {"error": f"Not enough observations ({subset.shape[0]}) for {len(x_cols)} predictors"}

    Y = subset[y_col].astype(float)
    X = subset[x_cols].astype(float)
    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()

    # Coefficients
    coefficients = {}
    for name, coef, se, t, p in zip(
        model.params.index, model.params, model.bse, model.tvalues, model.pvalues
    ):
        coefficients[name] = {
            "coefficient": round(float(coef), 6),
            "std_error": round(float(se), 6),
            "t_statistic": round(float(t), 4),
            "p_value": round(float(p), 6),
            "significant": bool(p < 0.05),
        }

    # VIF for multicollinearity (only if >1 predictor)
    vif = {}
    if len(x_cols) > 1:
        for i, col in enumerate(x_cols):
            try:
                vif[col] = round(float(variance_inflation_factor(X.values, i + 1)), 4)
            except Exception:
                vif[col] = None  # zero-variance or singular column

    # Diagnostics
    diagnostics = {
        "r_squared": round(float(model.rsquared), 6),
        "adj_r_squared": round(float(model.rsquared_adj), 6),
        "f_statistic": round(float(model.fvalue), 4),
        "f_p_value": round(float(model.f_pvalue), 6),
        "aic": round(float(model.aic), 2),
        "bic": round(float(model.bic), 2),
        "durbin_watson": round(float(durbin_watson(model.resid)), 4),
        "observations": int(model.nobs),
    }

    return {
        "type": "ols_regression",
        "dependent": y_col,
        "predictors": x_cols,
        "coefficients": coefficients,
        "vif": vif if vif else None,
        "diagnostics": diagnostics,
    }


def logistic_regression(df: pd.DataFrame, y_col: str, x_cols: list) -> dict:
    """Logistic regression for binary outcomes."""
    import statsmodels.api as sm

    subset = df[[y_col] + x_cols].dropna()
    Y = subset[y_col].astype(float)
    unique = Y.unique()
    if len(unique) != 2:
        return {"error": f"Logistic regression requires binary outcome, got {len(unique)} unique values"}

    X = subset[x_cols].astype(float)
    X = sm.add_constant(X)

    model = sm.Logit(Y, X).fit(disp=0)

    coefficients = {}
    for name, coef, se, z, p in zip(
        model.params.index, model.params, model.bse, model.tvalues, model.pvalues
    ):
        odds_ratio = float(np.exp(coef))
        coefficients[name] = {
            "coefficient": round(float(coef), 6),
            "odds_ratio": round(odds_ratio, 6),
            "std_error": round(float(se), 6),
            "z_statistic": round(float(z), 4),
            "p_value": round(float(p), 6),
            "significant": bool(p < 0.05),
        }

    diagnostics = {
        "pseudo_r_squared": round(float(model.prsquared), 6),
        "log_likelihood": round(float(model.llf), 4),
        "aic": round(float(model.aic), 2),
        "bic": round(float(model.bic), 2),
        "observations": int(model.nobs),
    }

    return {
        "type": "logistic_regression",
        "dependent": y_col,
        "predictors": x_cols,
        "coefficients": coefficients,
        "diagnostics": diagnostics,
    }


def time_series_decompose(df: pd.DataFrame, value_col: str, period: int = None, date_col: str = None) -> dict:
    """Decompose a time series into trend, seasonal, and residual components."""
    from statsmodels.tsa.seasonal import seasonal_decompose

    if date_col:
        df = df.sort_values(date_col)
        series = df[value_col].dropna().reset_index(drop=True)
    else:
        series = df[value_col].dropna().reset_index(drop=True)

    if period is None:
        period = min(7, len(series) // 3)
    if period < 2:
        return {"error": "Not enough data points for decomposition"}

    try:
        result = seasonal_decompose(series, model='additive', period=period)
    except Exception as e:
        return {"error": str(e)}

    trend = result.trend.dropna()
    seasonal = result.seasonal.dropna()
    resid = result.resid.dropna()

    return {
        "type": "time_series_decomposition",
        "column": value_col,
        "period": period,
        "observations": len(series),
        "trend": {
            "mean": round(float(trend.mean()), 4),
            "direction": "increasing" if trend.iloc[-1] > trend.iloc[0] else "decreasing",
        },
        "seasonality": {
            "amplitude": round(float(seasonal.max() - seasonal.min()), 4),
            "present": bool(seasonal.std() > series.std() * 0.05),
        },
        "residual": {
            "mean": round(float(resid.mean()), 4),
            "std": round(float(resid.std()), 4),
        },
    }


def frequency_analysis(df: pd.DataFrame, column: str, top_n: int = 20) -> dict:
    """Frequency counts for a categorical or discrete column."""
    counts = df[column].value_counts().head(top_n)
    total = len(df[column].dropna())
    return {
        "column": column,
        "total": total,
        "unique": int(df[column].nunique()),
        "top_values": {
            str(k): {"count": int(v), "pct": round(v / total * 100, 2)}
            for k, v in counts.items()
        },
    }


# ── Dispatcher ───────────────────────────────────────────────

async def run_stats(
    operation: str,
    sql: str = "",
    csv_path: str = "",
    json_data: str = "",
    drive_file_id: str = "",
    y_col: str = "",
    x_cols: str = "",
    columns: str = "",
    method: str = "pearson",
    period: int = 0,
    date_col: str = "",
    top_n: int = 20,
) -> str:
    """Main dispatcher for stats operations."""
    try:
        df = await load_data(sql=sql, csv_path=csv_path, json_data=json_data, drive_file_id=drive_file_id)
    except Exception as e:
        return json.dumps({"error": f"Data loading failed: {str(e)}"})

    col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else []
    x_list = [c.strip() for c in x_cols.split(",") if c.strip()] if x_cols else []

    try:
        if operation == "describe":
            result = descriptive_stats(df, col_list or None)
        elif operation == "correlation":
            result = correlation_matrix(df, col_list or None, method=method)
        elif operation == "ols":
            if not y_col or not x_list:
                return json.dumps({"error": "OLS requires y_col and x_cols"})
            result = ols_regression(df, y_col, x_list)
        elif operation == "logistic":
            if not y_col or not x_list:
                return json.dumps({"error": "Logistic regression requires y_col and x_cols"})
            result = logistic_regression(df, y_col, x_list)
        elif operation == "decompose":
            if not y_col:
                return json.dumps({"error": "Time series decomposition requires y_col (the value column)"})
            result = time_series_decompose(df, y_col, period=period or None, date_col=date_col or None)
        elif operation == "frequency":
            if not y_col:
                return json.dumps({"error": "Frequency analysis requires y_col (the column to count)"})
            result = frequency_analysis(df, y_col, top_n=top_n)
        elif operation == "columns":
            result = {
                "rows": len(df),
                "columns": {
                    col: {"dtype": str(df[col].dtype), "non_null": int(df[col].count()), "null": int(df[col].isnull().sum())}
                    for col in df.columns
                },
            }
        else:
            return json.dumps({"error": f"Unknown operation: {operation}. Use: describe, correlation, ols, logistic, decompose, frequency, columns"})

        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"{operation} failed: {str(e)}"})
