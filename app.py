import pandas as pd
import numpy as np
import gradio as gr
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import scipy.stats as stats

# ------------------------------ Load Data ------------------------------
df = pd.read_csv("chocolate_sales.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

def train_and_forecast():
    # Split into train/test
    train = df.iloc[:-52]
    test = df.iloc[-52:]
    order = (2, 0, 2)

    # Fit ARIMA model
    model = ARIMA(train["sales"], order=order)
    model_fit = model.fit()

    # Forecast next 52 weeks
    fc_res = model_fit.get_forecast(steps=52, alpha=0.10)
    forecast = fc_res.predicted_mean
    conf_int = fc_res.conf_int()
    dates_fc = pd.date_range(start="2025-01-05", periods=52, freq="W-SUN")
    forecast.index = dates_fc
    conf_int.index = dates_fc

    # Round for display
    forecast_rounded = forecast.round(2)
    conf_int_rounded = conf_int.round(2)

    # Evaluate on test set
    ev_res = model_fit.get_forecast(steps=52, alpha=0.10)
    test_fc = ev_res.predicted_mean
    test_ci = ev_res.conf_int()
    test_fc.index = test.index
    test_fc_rounded = test_fc.round(2)
    test_ci_rounded = test_ci.round(2)

    r2 = r2_score(test["sales"], test_fc_rounded)
    mse = mean_squared_error(test["sales"], test_fc_rounded)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test["sales"], test_fc_rounded)
    mape = np.mean(np.abs((test["sales"] - test_fc_rounded) / test["sales"])) * 100

    return (
        model_fit,
        forecast_rounded,
        conf_int_rounded,
        test_fc_rounded,
        test_ci_rounded,
        r2,
        rmse,
        mae,
        mape,
        train.index,
        model_fit.resid
    )

# Train once at startup
(
    model_fit,
    forecast_rounded,
    conf_int_rounded,
    test_forecast_rounded,
    test_conf_int_rounded,
    r2,
    rmse,
    mae,
    mape,
    train_index,
    residuals
) = train_and_forecast()


# ------------------------------ Plotting Functions ------------------------------

def plot_forecast():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_rounded.index,
        y=forecast_rounded,
        mode="lines",
        name="Forecast",
        line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=list(forecast_rounded.index) + list(forecast_rounded.index[::-1]),
        y=list(conf_int_rounded.iloc[:, 0]) + list(conf_int_rounded.iloc[:, 1][::-1]),
        fill="toself",
        fillcolor="rgba(0,0,255,0.2)",
        hoverinfo="skip",
        name="90% CI"
    ))
    fig.update_layout(
        title="2025 Weekly Chocolate Sales Forecast",
        xaxis_title="Week",
        yaxis_title="Sales",
        hovermode="x unified"
    )
    return fig

def plot_evaluation():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index[-52:], y=df["sales"][-52:],
        mode="lines", name="Actual", line=dict(color="black")
    ))
    fig.add_trace(go.Scatter(
        x=test_forecast_rounded.index, y=test_forecast_rounded,
        mode="lines", name="Forecast", line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=list(test_forecast_rounded.index) + list(test_forecast_rounded.index[::-1]),
        y=list(test_conf_int_rounded.iloc[:, 0]) + list(test_conf_int_rounded.iloc[:, 1][::-1]),
        fill="toself",
        fillcolor="rgba(0,0,255,0.2)",
        hoverinfo="skip",
        name="90% CI"
    ))
    fig.update_layout(
        title="2024 Actual vs Forecast",
        xaxis_title="Week",
        yaxis_title="Sales",
        hovermode="x unified"
    )
    return fig

def plot_residuals():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_index, y=residuals,
        mode="lines", name="Residuals"
    ))
    fig.update_layout(
        title="Residuals Over Time",
        xaxis_title="Date",
        yaxis_title="Residual"
    )
    return fig

def plot_resid_hist():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=20, edgecolor="k", alpha=0.7)
    ax.set_title("Residual Histogram")
    return fig

def plot_resid_qq():
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot")
    return fig

def plot_resid_acf():
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(residuals, ax=ax, lags=40)
    ax.set_title("Residual ACF")
    return fig

# ------------------------------ Utility Functions ------------------------------

def lookup_sales(date):
    date = pd.to_datetime(date)
    if date not in df.index:
        return f"No data for {date.date()}"
    return f"Sales on {date.date()}: ${df.loc[date, 'sales']:.2f}"

def get_csv():
    out = pd.DataFrame({
        "date": forecast_rounded.index,
        "forecast": forecast_rounded.values,
        "ci_lower": conf_int_rounded.iloc[:, 0].values,
        "ci_upper": conf_int_rounded.iloc[:, 1].values,
    }).set_index("date")
    return out.to_csv().encode("utf-8")


# ------------------------------ Gradio Interface ------------------------------

with gr.Blocks() as demo:
    gr.Markdown("## 🍫 Chocolate Sales Forecast (ARIMA)")
    gr.Markdown("![Logo](https://i.imgur.com/oDM4ECC.jpeg)")

    with gr.Tab("2025 Forecast & Summary"):
        gr.Plot(plot_forecast)
        date_sel = gr.Date(
            label="Select a forecast week",
            value=forecast_rounded.index.min().date(),
            minimum=forecast_rounded.index.min().date(),
            maximum=forecast_rounded.index.max().date()
        )
        out_sales = gr.Textbox(label="Forecasted Sales", interactive=False)
        out_ci = gr.Textbox(label="90% Confidence Interval", interactive=False)

        def update(date):
            date = pd.to_datetime(date)
            if date not in forecast_rounded.index:
                return "Invalid", ""
            v = forecast_rounded[date]
            ci = conf_int_rounded.loc[date]
            return f"${v:.2f}", f"[{ci[0]:.2f}, {ci[1]:.2f}]"

        date_sel.change(update, date_sel, [out_sales, out_ci])

        gr.Markdown("### Summary Metrics")
        total = forecast_rounded.sum()
        avg = forecast_rounded.mean()
        mn = forecast_rounded.min()
        mx = forecast_rounded.max()
        week_min = forecast_rounded.idxmin().date()
        week_max = forecast_rounded.idxmax().date()
        with gr.Row():
            gr.Markdown(f"**Total:** {total:,.2f}")
            gr.Markdown(f"**Average:** {avg:.2f}")
            gr.Markdown(f"**Min:** {mn:.2f} (week of {week_min})")
            gr.Markdown(f"**Max:** {mx:.2f} (week of {week_max})")

        gr.DownloadButton("Download CSV", get_csv, file_name="forecast_2025.csv")

    with gr.Tab("2024 Model Evaluation"):
        gr.Markdown("### Performance Metrics")
        with gr.Row():
            gr.Markdown(f"R²: {r2:.4f}")
            gr.Markdown(f"RMSE: {rmse:.2f}")
            gr.Markdown(f"MAE: {mae:.2f}")
            gr.Markdown(f"MAPE: {mape:.2f}%")
        gr.Plot(plot_evaluation)

    with gr.Tab("Residual Diagnostics"):
        gr.Plot(plot_residuals)
        gr.Plot(plot_resid_hist)
        gr.Plot(plot_resid_qq)
        gr.Plot(plot_resid_acf)

    with gr.Tab("Historical Sales Lookup"):
        hist_date = gr.Date(
            label="Select a date",
            value=df.index.max().date(),
            minimum=df.index.min().date(),
            maximum=df.index.max().date()
        )
        hist_out = gr.Textbox(label="Sales", interactive=False)
        hist_date.change(lookup_sales, hist_date, hist_out)

if __name__ == "__main__":
    demo.launch()

