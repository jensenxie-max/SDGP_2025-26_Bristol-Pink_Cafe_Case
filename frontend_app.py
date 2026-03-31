from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend_core import (
    ACCENT_COLORS,
    APP_TITLE,
    DEFAULT_MODELS,
    FORECAST_DAYS,
    PAGE_OPTIONS,
    TRAINING_MAX_WEEKS,
    TRAINING_MIN_WEEKS,
    build_daily_series,
    choose_model,
    evaluate_models,
    get_top_products,
    load_uploaded_or_local_data,
    forecast_with_model,
)


# ---------- Theme / Frontend ----------

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #eef4fb;
            --surface: #ffffff;
            --surface-soft: #f8fbff;
            --text: #0f172a;
            --muted: #5b6475;
            --primary: #0b63ce;
            --secondary: #f97316;
            --border: #d8e4f2;
            --shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
        }

        .stApp {
            background: linear-gradient(180deg, #eef4fb 0%, #e6eef8 100%);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1f33 0%, #163a64 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        [data-testid="metric-container"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 10px 8px;
            box-shadow: var(--shadow);
        }

        [data-testid="metric-container"] * { color: var(--text) !important; }
        [data-testid="metric-container"] [data-testid="stMetricLabel"] { color: var(--muted) !important; }
        [data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--text) !important; }

        .hero-banner {
            background: linear-gradient(135deg, #0f1f33 0%, #0b63ce 55%, #f97316 120%);
            border-radius: 24px;
            padding: 28px 32px;
            margin-bottom: 18px;
            box-shadow: 0 18px 40px rgba(11, 99, 206, 0.24);
        }

        .hero-banner h1 { font-size: 2.2rem; margin-bottom: 0.35rem; }
        .hero-banner p { font-size: 1rem; margin-bottom: 0; }

        .section-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 18px 18px 12px 18px;
            box-shadow: var(--shadow);
            margin-bottom: 14px;
        }

        .section-card h3 { margin: 0 0 6px 0; color: var(--text); }
        .section-card p { color: var(--muted); margin-bottom: 0.4rem; }

        .feature-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 20px;
            min-height: 220px;
            box-shadow: var(--shadow);
            margin-bottom: 8px;
        }

        .feature-icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
        .feature-card h4 { margin: 0 0 8px 0; color: var(--text); }
        .feature-card p { color: var(--muted); min-height: 70px; }

        .pill {
            display: inline-block;
            background: #e8f1fb;
            color: #0b63ce;
            border: 1px solid #c7ddf6;
            border-radius: 999px;
            padding: 6px 11px;
            margin: 0 8px 8px 0;
            font-size: 0.85rem;
            font-weight: 700;
        }

        .subtle-note {
            background: #fff7ed;
            border-left: 5px solid #f97316;
            color: #7c2d12;
            padding: 12px 14px;
            border-radius: 12px;
            margin-bottom: 8px;
        }

        .stButton > button,
        .stDownloadButton > button {
            width: 100%;
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 700;
            background: linear-gradient(90deg, #0b63ce 0%, #f97316 100%);
            box-shadow: 0 10px 18px rgba(11, 99, 206, 0.18);
        }

        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #eef4fb;
            border-radius: 12px 12px 0 0;
            padding: 10px 16px;
            font-weight: 700;
        }
        .stTabs [aria-selected="true"] {
            background-color: #dceafc !important;
            color: #0b63ce !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 18px;
            overflow: hidden;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4,
        [data-testid="stMarkdownContainer"] h5,
        [data-testid="stMarkdownContainer"] h6,
        .stRadio label,
        .stSelectbox label,
        .stMultiSelect label,
        .stDateInput label,
        .stSlider label,
        .stTextInput label,
        .stNumberInput label {
            color: var(--text) !important;
        }

        .section-card,
        .section-card *,
        .feature-card,
        .feature-card *,
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] *,
        .stAlert,
        .stAlert *,
        .stInfo,
        .stInfo *,
        .stWarning,
        .stWarning *,
        .stSuccess,
        .stSuccess * {
            color: var(--text) !important;
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] .stDateInput label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stFileUploader label,
        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stNumberInput label {
            color: #f7fbff !important;
        }

        .hero-banner,
        .hero-banner *,
        [data-testid="stMarkdownContainer"] .hero-banner,
        [data-testid="stMarkdownContainer"] .hero-banner * {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def navigate_to(page_name: str) -> None:
    st.session_state["nav_page"] = page_name
    rerun_app()


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero-banner">
            <h1>{APP_TITLE}</h1>
            <p>
                A cleaner multi-section dashboard with stronger visual contrast, clearer navigation and dedicated workspaces
                for historical analysis, forecasting, model comparison and raw data review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_metric_cards(df: pd.DataFrame) -> None:
    products = df["Product_Name"].nunique()
    total_records = len(df)
    total_sales = int(df["Sales_Volume"].sum()) if not df.empty else 0
    latest_date = df["Date"].max().strftime("%Y-%m-%d") if not df.empty else "-"
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Products", products)
    col2.metric("Historical Records", total_records)
    col3.metric("Total Sales", f"{total_sales:,}")
    col4.metric("Latest Date", latest_date)


def build_line_chart(df: pd.DataFrame, title: str, x_col: str, y_col: str, color_col: str) -> go.Figure:
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        markers=True,
        color_discrete_sequence=ACCENT_COLORS,
        template="plotly_white",
        title=title,
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text=color_col.replace("_", " "),
        xaxis_title=x_col.replace("_", " "),
        yaxis_title=y_col.replace("_", " "),
        title_font=dict(size=20),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig


def build_bar_chart(df: pd.DataFrame, title: str, x_col: str, y_col: str, color_col: Optional[str] = None) -> go.Figure:
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        template="plotly_white",
        color_discrete_sequence=ACCENT_COLORS,
        title=title,
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
        title_font=dict(size=20),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig


def build_forecast_chart(actual_series: pd.Series, forecast_df: pd.DataFrame, product_name: str) -> go.Figure:
    actual_df = actual_series.reset_index()
    actual_df.columns = ["Date", "Actual Sales"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual_df["Date"],
            y=actual_df["Actual Sales"],
            mode="lines+markers",
            name="Actual Sales",
            line=dict(color="#0B63CE", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Predicted Sales"],
            mode="lines+markers",
            name="Predicted Sales",
            line=dict(color="#F97316", width=3, dash="dot"),
        )
    )
    fig.add_vrect(
        x0=forecast_df["Date"].min(),
        x1=forecast_df["Date"].max(),
        fillcolor="#F97316",
        opacity=0.08,
        line_width=0,
        annotation_text="Forecast Window",
        annotation_position="top left",
    )
    fig.update_layout(
        title=f"4-Week Forecast for {product_name}",
        xaxis_title="Date",
        yaxis_title="Sales Volume",
        legend_title_text="Series",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig


def build_summary_cards(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    category_summary = (
        df.groupby("Category", as_index=False)
        .agg(Total_Sales=("Sales_Volume", "sum"), Products=("Product_Name", "nunique"))
        .sort_values("Total_Sales", ascending=False)
    )
    product_summary = (
        df.groupby(["Category", "Product_Name"], as_index=False)["Sales_Volume"]
        .sum()
        .sort_values(["Category", "Sales_Volume"], ascending=[True, False])
    )
    return category_summary, product_summary


def render_page_intro(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{title}</h3>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_page(df: pd.DataFrame, filtered_df: pd.DataFrame, top3_foods: List[str], top3_coffees: List[str]) -> None:
    render_page_intro(
        "Dashboard overview",
        "Use the buttons below to jump into a dedicated workspace. This makes the app feel less crowded than placing all tasks in one long page.",
    )

    card_cols = st.columns(4)
    feature_cards = [
        ("📊", "Historical Analysis", "Focus on top sellers, compare category patterns and inspect recent sales fluctuations with clearer visual separation.", "📊 Historical Analysis"),
        ("📈", "Forecast Studio", "Choose the training window, compare models and open a product-by-product forecasting workspace instead of a cramped single tab.", "📈 Forecast Studio"),
        ("⚙️", "Model Evaluation", "Review MAE, RMSE and MAPE in a dedicated comparison page, with summary charts that highlight the strongest model.", "⚙️ Model Evaluation"),
        ("🗂️", "Data Centre", "Browse the cleaned dataset, export processed tables and quickly check what files or records are driving the dashboard.", "🗂️ Data Centre"),
    ]

    for col, (icon, title, desc, page_name) in zip(card_cols, feature_cards):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Open {title}", key=f"open_{page_name}"):
                navigate_to(page_name)

    st.markdown("### Quick insights")
    info_col1, info_col2 = st.columns([1.2, 1])
    with info_col1:
        summary = (
            filtered_df.groupby(["Category", "Product_Name"], as_index=False)["Sales_Volume"]
            .sum()
            .sort_values(["Category", "Sales_Volume"], ascending=[True, False])
        )
        if not summary.empty:
            quick_fig = build_bar_chart(summary, "Sales leaderboard in selected range", "Product_Name", "Sales_Volume", "Category")
            st.plotly_chart(quick_fig, use_container_width=True)
    with info_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Top foods**")
        for name in top3_foods or ["No food data available"]:
            st.markdown(f"<span class='pill'>{name}</span>", unsafe_allow_html=True)
        st.markdown("**Top coffees**")
        for name in top3_coffees or ["No coffee data available"]:
            st.markdown(f"<span class='pill'>{name}</span>", unsafe_allow_html=True)
        st.markdown("<div class='subtle-note'>The page layout now separates summary, analysis, forecasting and data exploration so the interface feels clearer and less repetitive.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    timeline_df = (
        filtered_df.groupby(["Date", "Category"], as_index=False)["Sales_Volume"]
        .sum()
        .sort_values("Date")
    )
    if not timeline_df.empty:
        st.markdown("### Category trend snapshot")
        trend_fig = build_line_chart(timeline_df, "Category sales trend in selected period", "Date", "Sales_Volume", "Category")
        st.plotly_chart(trend_fig, use_container_width=True)


def render_historical_page(filtered_df: pd.DataFrame, top3_foods: List[str], top3_coffees: List[str]) -> None:
    render_page_intro(
        "Historical analysis workspace",
        "Explore sales movement over time with separate sections for foods and coffees. You can keep the default top sellers or manually choose products.",
    )

    all_products = sorted(filtered_df["Product_Name"].unique().tolist())
    default_selection = [name for name in top3_foods + top3_coffees if name in all_products]
    selected_history_products = st.multiselect("Products to compare", options=all_products, default=default_selection)

    if not selected_history_products:
        st.info("Select at least one product to display historical charts.")
        return

    history_view = st.radio("Display mode", ["Line Chart", "Table", "Both"], horizontal=True)
    selected_df = filtered_df[filtered_df["Product_Name"].isin(selected_history_products)].copy()

    col1, col2 = st.columns(2)
    with col1:
        food_df = selected_df[selected_df["Category"] == "Food"]
        st.markdown("### Food products")
        if food_df.empty:
            st.info("No food products are selected in the current range.")
        else:
            if history_view in ["Line Chart", "Both"]:
                food_fig = build_line_chart(food_df, "Food sales fluctuation", "Date", "Sales_Volume", "Product_Name")
                st.plotly_chart(food_fig, use_container_width=True)
            if history_view in ["Table", "Both"]:
                st.dataframe(food_df.sort_values(["Product_Name", "Date"]), use_container_width=True)

    with col2:
        coffee_df = selected_df[selected_df["Category"] == "Coffee"]
        st.markdown("### Coffee products")
        if coffee_df.empty:
            st.info("No coffee products are selected in the current range.")
        else:
            if history_view in ["Line Chart", "Both"]:
                coffee_fig = build_line_chart(coffee_df, "Coffee sales fluctuation", "Date", "Sales_Volume", "Product_Name")
                st.plotly_chart(coffee_fig, use_container_width=True)
            if history_view in ["Table", "Both"]:
                st.dataframe(coffee_df.sort_values(["Product_Name", "Date"]), use_container_width=True)

    rank_df = (
        filtered_df.groupby(["Category", "Product_Name"], as_index=False)["Sales_Volume"]
        .sum()
        .sort_values(["Category", "Sales_Volume"], ascending=[True, False])
    )
    st.markdown("### Ranked sales summary")
    st.dataframe(rank_df, use_container_width=True)


def render_forecast_page(df: pd.DataFrame, training_weeks: int, training_days: int, model_preference: str, default_products: List[str]) -> None:
    render_page_intro(
        "Forecast studio",
        "This page is focused only on prediction work. It separates model choice, product selection, forecast output and zoom controls from the rest of the dashboard.",
    )

    all_products = sorted(df["Product_Name"].unique().tolist())
    selected_products = st.multiselect("Products to forecast", options=all_products, default=[name for name in default_products if name in all_products])

    if not selected_products:
        st.info("Choose at least one product to generate a forecast.")
        return

    st.markdown(
        f"<div class='subtle-note'>Forecast horizon: <strong>{FORECAST_DAYS} days</strong> | Training window: <strong>{training_weeks} weeks</strong> | Model mode: <strong>{model_preference}</strong></div>",
        unsafe_allow_html=True,
    )

    forecast_outputs: Dict[str, pd.DataFrame] = {}
    product_tabs = st.tabs([name for name in selected_products])

    for product_name, product_tab in zip(selected_products, product_tabs):
        with product_tab:
            series = build_daily_series(df, product_name)
            if len(series) < training_days:
                st.warning(f"{product_name} does not have enough daily history for a {training_weeks}-week training window.")
                continue

            training_series = series.tail(training_days)
            chosen_model, evaluation_df = choose_model(training_series, training_days, model_preference)
            preds = forecast_with_model(training_series, chosen_model, FORECAST_DAYS)

            forecast_dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")
            forecast_df = pd.DataFrame({
                "Date": forecast_dates,
                "Predicted Sales": np.round(preds, 2),
                "Product Name": product_name,
                "Model Used": chosen_model,
            })
            forecast_outputs[product_name] = forecast_df

            top_info1, top_info2, top_info3 = st.columns(3)
            top_info1.metric("Model Used", chosen_model)
            top_info2.metric("Avg Forecast", round(float(forecast_df["Predicted Sales"].mean()), 2))
            top_info3.metric("Max Forecast", round(float(forecast_df["Predicted Sales"].max()), 2))

            if not evaluation_df.empty:
                best_row = evaluation_df[evaluation_df["Model"] == chosen_model].head(1)
                if not best_row.empty:
                    metric_cols = st.columns(3)
                    metric_cols[0].metric("MAE", best_row.iloc[0]["MAE"])
                    metric_cols[1].metric("RMSE", best_row.iloc[0]["RMSE"])
                    metric_cols[2].metric("MAPE (%)", best_row.iloc[0]["MAPE (%)"])

            recent_actual = series.tail(training_days)
            forecast_fig = build_forecast_chart(recent_actual, forecast_df[["Date", "Predicted Sales"]], product_name)
            st.plotly_chart(forecast_fig, use_container_width=True)

            zoom_start, zoom_end = st.select_slider(
                "Choose forecast sub-range",
                options=list(forecast_dates.date),
                value=(forecast_dates[0].date(), forecast_dates[-1].date()),
                key=f"zoom_{product_name}",
            )
            zoomed_forecast = forecast_df[(forecast_df["Date"].dt.date >= zoom_start) & (forecast_df["Date"].dt.date <= zoom_end)]

            view_mode = st.radio("Forecast output view", ["Graph", "Table", "Both"], horizontal=True, key=f"view_{product_name}")
            if view_mode in ["Graph", "Both"]:
                zoom_fig = build_line_chart(zoomed_forecast, f"Zoomed forecast for {product_name}", "Date", "Predicted Sales", "Product Name")
                st.plotly_chart(zoom_fig, use_container_width=True)
            if view_mode in ["Table", "Both"]:
                st.dataframe(zoomed_forecast, use_container_width=True)

    if forecast_outputs:
        export_df = pd.concat(forecast_outputs.values(), ignore_index=True)
        st.download_button(
            "Download all forecast results as CSV",
            export_df.to_csv(index=False).encode("utf-8"),
            file_name="bristol_pink_forecasts.csv",
            mime="text/csv",
        )


def render_evaluation_page(df: pd.DataFrame, training_days: int, default_products: List[str]) -> None:
    render_page_intro(
        "Model evaluation centre",
        "Compare algorithms in a dedicated page instead of hiding evaluation at the bottom of a crowded dashboard. This makes the accuracy story easier to explain in your report.",
    )

    all_products = sorted(df["Product_Name"].unique().tolist())
    selected_products = st.multiselect(
        "Products included in evaluation",
        options=all_products,
        default=[name for name in default_products if name in all_products],
        key="eval_products",
    )

    if not selected_products:
        st.info("Choose at least one product to compare models.")
        return

    evaluation_rows = []
    for product_name in selected_products:
        series = build_daily_series(df, product_name)
        if len(series) < training_days:
            continue
        evaluation_df = evaluate_models(series.tail(training_days), training_days)
        if evaluation_df.empty:
            continue
        evaluation_df.insert(0, "Product", product_name)
        evaluation_rows.append(evaluation_df)

    if not evaluation_rows:
        st.info("More historical data is needed before algorithm evaluation can be shown.")
        return

    all_eval = pd.concat(evaluation_rows, ignore_index=True)
    st.dataframe(all_eval, use_container_width=True)

    summary = all_eval.groupby("Model", as_index=False)[["MAE", "RMSE", "MAPE (%)"]].mean(numeric_only=True).sort_values("RMSE")
    summary[["MAE", "RMSE", "MAPE (%)"]] = summary[["MAE", "RMSE", "MAPE (%)"]].round(2)

    col1, col2 = st.columns([1.15, 1])
    with col1:
        rmse_fig = build_bar_chart(summary, "Average RMSE by forecast model", "Model", "RMSE")
        st.plotly_chart(rmse_fig, use_container_width=True)
    with col2:
        mae_fig = build_bar_chart(summary, "Average MAE by forecast model", "Model", "MAE")
        st.plotly_chart(mae_fig, use_container_width=True)

    st.markdown("### Average validation error by model")
    st.dataframe(summary, use_container_width=True)
    best_model = summary.iloc[0]["Model"]
    st.success(f"Best overall validation model in the current training window: {best_model}")


def render_data_centre(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    render_page_intro(
        "Data centre",
        "Inspect cleaned records, summary tables and export files from one place. This page also helps during testing because you can directly verify the underlying dataset.",
    )

    category_summary, product_summary = build_summary_cards(df)
    col1, col2 = st.columns([0.8, 1.2])
    with col1:
        st.markdown("### Category summary")
        st.dataframe(category_summary, use_container_width=True)
        st.markdown("### Export cleaned data")
        st.download_button(
            "Download cleaned merged dataset",
            df.to_csv(index=False).encode("utf-8"),
            file_name="bristol_pink_cleaned_data.csv",
            mime="text/csv",
        )
    with col2:
        st.markdown("### Product summary")
        st.dataframe(product_summary, use_container_width=True)

    st.markdown("### Filtered records in active history range")
    st.dataframe(filtered_df.sort_values(["Category", "Product_Name", "Date"]), use_container_width=True, height=420)


def main() -> None:
    st.set_page_config(page_title="Bristol-Pink Dashboard", layout="wide")
    inject_custom_css()

    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = PAGE_OPTIONS[0]

    render_hero()

    with st.sidebar:
        st.markdown("## Navigation")
        current_page = st.radio(
            "Choose workspace",
            PAGE_OPTIONS,
            index=PAGE_OPTIONS.index(st.session_state["nav_page"]),
            label_visibility="collapsed",
        )
        st.session_state["nav_page"] = current_page

        st.markdown("---")
        uploaded_files = st.file_uploader("Upload Bristol-Pink sales CSV files", type=["csv"], accept_multiple_files=True)

    df = load_uploaded_or_local_data(uploaded_files, warning_callback=st.warning)

    if df.empty:
        render_page_intro(
            "No data loaded yet",
            "Upload one or more CSV files from the sidebar to activate the dashboard. If sample CSV files are stored next to the script, they will be loaded automatically.",
        )
        return

    show_metric_cards(df)

    with st.sidebar:
        st.markdown("---")
        st.markdown("## Analysis controls")
        training_weeks = st.slider("Training period (weeks)", min_value=TRAINING_MIN_WEEKS, max_value=TRAINING_MAX_WEEKS, value=6)
        training_days = training_weeks * 7

        model_preference = st.selectbox("Forecast model", ["Auto (Best RMSE)"] + DEFAULT_MODELS, index=0)

        st.markdown("### Historical range")
        default_history_end = df["Date"].max().date()
        default_history_start = max(df["Date"].min().date(), (df["Date"].max() - pd.Timedelta(days=27)).date())
        history_start = st.date_input("History start date", value=default_history_start)
        history_end = st.date_input("History end date", value=default_history_end)

    if history_start > history_end:
        st.error("Historical start date cannot be after end date.")
        return

    history_mask = (df["Date"].dt.date >= history_start) & (df["Date"].dt.date <= history_end)
    filtered_df = df.loc[history_mask].copy()
    if filtered_df.empty:
        st.warning("No historical data exists in the selected date range.")
        return

    top3_foods = get_top_products(filtered_df, "Food", 3)
    top3_coffees = get_top_products(filtered_df, "Coffee", 3)
    default_products = top3_foods + top3_coffees

    if current_page == "🏠 Home":
        render_home_page(df, filtered_df, top3_foods, top3_coffees)
    elif current_page == "📊 Historical Analysis":
        render_historical_page(filtered_df, top3_foods, top3_coffees)
    elif current_page == "📈 Forecast Studio":
        render_forecast_page(df, training_weeks, training_days, model_preference, default_products)
    elif current_page == "⚙️ Model Evaluation":
        render_evaluation_page(df, training_days, default_products)
    elif current_page == "🗂️ Data Centre":
        render_data_centre(df, filtered_df)


if __name__ == "__main__":
    main()
