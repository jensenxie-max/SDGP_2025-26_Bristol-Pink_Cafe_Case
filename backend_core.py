from __future__ import annotations

import glob
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

APP_TITLE = "Bristol-Pink Bakery Sales Prediction Dashboard"
TRAINING_MIN_WEEKS = 4
TRAINING_MAX_WEEKS = 8
FORECAST_DAYS = 28
VALIDATION_DAYS = 7
DEFAULT_MODELS = ["Linear Regression", "Random Forest", "7-Day Moving Average"]
PAGE_OPTIONS = [
    "🏠 Home",
    "📊 Historical Analysis",
    "📈 Forecast Studio",
    "⚙️ Model Evaluation",
    "🗂️ Data Centre",
]
ACCENT_COLORS = ["#0B63CE", "#F97316", "#10B981", "#7C3AED", "#EF4444", "#14B8A6"]


def infer_category(filename: str) -> str:
    lower = filename.lower()
    coffee_keywords = ["coffee", "americano", "latte", "cappuccino", "espresso", "mocha"]
    return "Coffee" if any(keyword in lower for keyword in coffee_keywords) else "Food"



def clean_product_name(raw_name: str) -> str:
    if raw_name is None:
        return "Unknown Product"
    name = str(raw_name)
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    name = re.sub(r"^pink[_\- ]*", "", name, flags=re.I)
    name = re.sub(r"sales", "", name, flags=re.I)
    name = re.sub(r"march.*$", "", name, flags=re.I)
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name.title() if name else "Unknown Product"



def parse_sales_file(file_obj) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    filename = getattr(file_obj, "name", "uploaded.csv")
    category = infer_category(filename)

    file_obj.seek(0)
    df_raw = pd.read_csv(file_obj)

    first_cell_is_nan = pd.isna(df_raw.iloc[0, 0]) if not df_raw.empty else False
    has_unnamed_cols = any(str(col).startswith("Unnamed") for col in df_raw.columns)

    if first_cell_is_nan or has_unnamed_cols:
        file_obj.seek(0)
        df_temp = pd.read_csv(file_obj, skiprows=1)
        if df_temp.empty:
            return frames

        first_col = df_temp.columns[0]
        df_temp = df_temp.rename(columns={first_col: "Date"})
        product_columns = [col for col in df_temp.columns if col != "Date"]

        for col in product_columns:
            temp_df = df_temp[["Date", col]].copy()
            temp_df.columns = ["Date", "Sales_Volume"]
            temp_df["Product_Name"] = clean_product_name(col)
            temp_df["Category"] = category
            temp_df["Source_File"] = filename
            frames.append(temp_df)
        return frames

    if "Date" in df_raw.columns:
        sales_column = None
        for candidate in ["Number Sold", "Sales", "Sales Volume", "Quantity", "Qty"]:
            if candidate in df_raw.columns:
                sales_column = candidate
                break

        if sales_column is not None:
            temp_df = df_raw[["Date", sales_column]].copy()
            temp_df.columns = ["Date", "Sales_Volume"]
            temp_df["Product_Name"] = clean_product_name(filename)
            temp_df["Category"] = category
            temp_df["Source_File"] = filename
            frames.append(temp_df)

    return frames



def load_uploaded_or_local_data(uploaded_files: Optional[Iterable], warning_callback=None) -> pd.DataFrame:
    all_frames: List[pd.DataFrame] = []

    if uploaded_files:
        for file_obj in uploaded_files:
            try:
                all_frames.extend(parse_sales_file(file_obj))
            except Exception as exc:
                if warning_callback:
                    warning_callback(f"Skipped {getattr(file_obj, 'name', 'a file')}: {exc}")
    else:
        class LocalFileWrapper:
            def __init__(self, filepath: str):
                self.filepath = filepath
                self.name = os.path.basename(filepath)
                self._fh = None

            def __enter__(self):
                self._fh = open(self.filepath, "rb")
                return self

            def __exit__(self, exc_type, exc, tb):
                if self._fh:
                    self._fh.close()

            def seek(self, *args, **kwargs):
                return self._fh.seek(*args, **kwargs)

            def read(self, *args, **kwargs):
                return self._fh.read(*args, **kwargs)

            def readline(self, *args, **kwargs):
                return self._fh.readline(*args, **kwargs)

            def __iter__(self):
                return iter(self._fh)

        local_csvs = glob.glob("*.csv") + glob.glob("/mnt/data/*.csv")
        seen = set()
        for path in local_csvs:
            if path in seen:
                continue
            seen.add(path)
            try:
                with LocalFileWrapper(path) as fh:
                    all_frames.extend(parse_sales_file(fh))
            except Exception:
                continue

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Sales_Volume"] = pd.to_numeric(df["Sales_Volume"], errors="coerce")
    df = df.dropna(subset=["Date", "Sales_Volume", "Product_Name", "Category"]).copy()

    df = (
        df.groupby(["Date", "Product_Name", "Category"], as_index=False)["Sales_Volume"]
        .sum()
        .sort_values(["Category", "Product_Name", "Date"])
    )
    return df



def build_daily_series(df: pd.DataFrame, product_name: str) -> pd.Series:
    series = (
        df.loc[df["Product_Name"] == product_name, ["Date", "Sales_Volume"]]
        .groupby("Date")["Sales_Volume"]
        .sum()
        .sort_index()
    )

    if series.empty:
        return series

    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_index, fill_value=0.0)
    series.index.name = "Date"
    return series.astype(float)



def get_top_products(df: pd.DataFrame, category: str, n: int = 3) -> List[str]:
    subset = df[df["Category"] == category]
    if subset.empty:
        return []
    return (
        subset.groupby("Product_Name")["Sales_Volume"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .index.tolist()
    )



def make_time_features(dates: pd.DatetimeIndex, start_date: pd.Timestamp) -> pd.DataFrame:
    day_idx = (dates - start_date).days.values
    feature_df = pd.DataFrame({"day_idx": day_idx, "day_of_week": dates.dayofweek}, index=dates)
    dow_dummies = pd.get_dummies(feature_df["day_of_week"], prefix="dow", drop_first=False)
    return pd.concat([feature_df[["day_idx"]], dow_dummies], axis=1)



def forecast_linear_regression(series: pd.Series, horizon: int) -> np.ndarray:
    dates = pd.DatetimeIndex(series.index)
    X_train = make_time_features(dates, dates.min())
    y_train = series.values

    model = LinearRegression()
    model.fit(X_train, y_train)

    future_dates = pd.date_range(dates.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    X_future = make_time_features(future_dates, dates.min())
    X_future = X_future.reindex(columns=X_train.columns, fill_value=0)
    preds = model.predict(X_future)
    return np.maximum(preds, 0)



def create_rf_training_frame(series: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"y": series.astype(float)})
    frame["lag_1"] = frame["y"].shift(1)
    frame["lag_7"] = frame["y"].shift(7)
    frame["rolling_mean_7"] = frame["y"].rolling(7).mean().shift(1)
    frame["day_of_week"] = frame.index.dayofweek
    return frame.dropna()



def forecast_random_forest(series: pd.Series, horizon: int) -> np.ndarray:
    model_data = create_rf_training_frame(series)
    if len(model_data) < 10:
        return forecast_linear_regression(series, horizon)

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    X = model_data[["lag_1", "lag_7", "rolling_mean_7", "day_of_week"]]
    y = model_data["y"]
    model.fit(X, y)

    history = list(series.values.astype(float))
    last_date = pd.Timestamp(series.index.max())
    preds = []

    for step in range(1, horizon + 1):
        future_date = last_date + pd.Timedelta(days=step)
        lag_1 = history[-1]
        lag_7 = history[-7] if len(history) >= 7 else history[-1]
        rolling_mean_7 = float(np.mean(history[-7:])) if len(history) >= 7 else float(np.mean(history))
        features = pd.DataFrame(
            [[lag_1, lag_7, rolling_mean_7, future_date.dayofweek]],
            columns=["lag_1", "lag_7", "rolling_mean_7", "day_of_week"],
        )
        pred = float(model.predict(features)[0])
        pred = max(pred, 0.0)
        preds.append(pred)
        history.append(pred)

    return np.array(preds)



def forecast_moving_average(series: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    history = list(series.values.astype(float))
    preds = []

    for _ in range(horizon):
        lookback = history[-window:] if len(history) >= window else history
        pred = max(float(np.mean(lookback)), 0.0)
        preds.append(pred)
        history.append(pred)

    return np.array(preds)



def forecast_with_model(series: pd.Series, model_name: str, horizon: int) -> np.ndarray:
    if model_name == "Linear Regression":
        return forecast_linear_regression(series, horizon)
    if model_name == "Random Forest":
        return forecast_random_forest(series, horizon)
    if model_name == "7-Day Moving Average":
        return forecast_moving_average(series, horizon)
    raise ValueError(f"Unsupported model: {model_name}")



def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)



def evaluate_models(product_series: pd.Series, training_days: int) -> pd.DataFrame:
    product_series = product_series.dropna().sort_index()
    if len(product_series) < max(training_days, 14):
        return pd.DataFrame()

    window = product_series.tail(training_days)
    val_days = min(VALIDATION_DAYS, max(3, len(window) // 4))
    train_series = window.iloc[:-val_days]
    val_series = window.iloc[-val_days:]

    results = []
    for model_name in DEFAULT_MODELS:
        try:
            preds = forecast_with_model(train_series, model_name, len(val_series))
            y_true = val_series.values.astype(float)
            mae = mean_absolute_error(y_true, preds)
            rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
            mape = safe_mape(y_true, preds)
            results.append(
                {
                    "Model": model_name,
                    "Validation Days": len(val_series),
                    "MAE": round(float(mae), 2),
                    "RMSE": round(float(rmse), 2),
                    "MAPE (%)": round(float(mape), 2) if pd.notna(mape) else np.nan,
                }
            )
        except Exception:
            continue

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    return result_df



def choose_model(product_series: pd.Series, training_days: int, preference: str) -> Tuple[str, pd.DataFrame]:
    evaluation_df = evaluate_models(product_series, training_days)
    if preference != "Auto (Best RMSE)":
        return preference, evaluation_df
    if evaluation_df.empty:
        return "Linear Regression", evaluation_df
    return str(evaluation_df.iloc[0]["Model"]), evaluation_df
