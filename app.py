import pandas as pd
import numpy as np
import gradio as gr
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import io
import base64

# Load data once globally
df = pd.read_csv("chocolate_sales.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

def train_and_forecast():
    train = df.iloc[:-52]
    test = df.iloc[-52:]
    order = (2, 0, 2)

    model = ARIMA(train["sales"], order=order)
    model_fit = model.fit()

    forecast_result = model_fit.get_forecast(steps=52, alpha=0.10)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    forecast_dates = pd.date_range(start="2025-01-05", periods=52, freq='W-SUN')
    forecast.index = forecast_dates
    conf_int.index = forecast_dates
    forecast_rounded = forecast.round(2)
    conf_int_rounded = conf_int.round(2)

    # Evaluation on test set
    test_forecast_result = model_fit.get_forecast(steps=52, alpha=0.10)
    test_forecast = test_forecast_result.predicted_mean
    test_conf_int = test_forecast_result.conf_int()
    test_forecast.index = test.index
    test_forecast_rounded = test_forecast.round(2)
    test_conf_int_rounded = test_conf_int.round(2)

    r2 = r2_score(test["sales"], test_forecast_rounded)
    mse = mean_squared_error(test["sales"], test_forecast_rounded)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test["sales"], test_forecast_rounded)
    mape = np.mean(np.abs((test["sales"] - test_forecast_rounded) / test["sales"])) * 100

    return (model_fit, forecast_rounded, conf_int_rounded,
            test_forecast_rounded, test_conf_int_rounded,
            r2, rmse, mae, mape,
            train.index, model_fit.resid)

# Train model once at startup
(model_fit, forecast_rounded, conf_int_rounded,
 test_forecast_rounded, test_conf_int_rounded,
 r2, rmse, mae, mape,
 train_index, residuals) = train_and_forecast()

# Plot forecast with confidence interval
def plot_forecast():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_rounded.index,
        y=forecast_rounded,
        mode="lines",
        name="Forecasted Sales",
        line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=list(forecast_rounded.index) + list(forecast_rounded.index[::-1]),
        y=list(conf_int_rounded.iloc[:, 0]) + list(conf_int_rounded.iloc[:, 1][::-1]),
        fill="toself",
        fillcolor="rgba(0,0,255,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="90% Confidence Interval"
    ))
    fig.update_layout(
        title="Projected Chocolate Sales (2025)",
        xaxis_title="Week",
        yaxis_title="Sales ($)",
        hovermode="x unified"
    )
    return fig

# Plot evaluation actual vs forecast
def plot_evaluation():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index[-52:], y=df["sales"][-52:],
        mode="lines", name="Actual Sales", line=dict(color="black")
    ))
    fig.add_trace(go.Scatter(
        x=test_forecast_rounded.index, y=test_forecast_rounded,
        mode="lines", name="Forecasted Sales", line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=list(test_forecast_rounded.index) + list(test_forecast_rounded.index[::-1]),
        y=list(test_conf_int_rounded.iloc[:, 0]) + list(test_conf_int_rounded.iloc[:, 1][::-1]),
        fill="toself",
        fillcolor="rgba(0,0,255,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="90% Confidence Interval"
    ))
    fig.update_layout(
        title="Forecast vs Actual Sales (2024)",
        xaxis_title="Week",
        yaxis_title="Sales ($)",
        hovermode="x unified"
    )
    return fig

# Residuals plot
def plot_residuals():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_index, y=residuals, mode="lines", name="Residuals"))
    fig.update_layout(title="Residuals Over Time", xaxis_title="Date", yaxis_title="Residual")
    return fig

# Residual histogram plot (matplotlib)
def plot_resid_hist():
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(residuals, bins=20, edgecolor="k", alpha=0.7)
    ax.set_title("Histogram of Residuals")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    return fig

# Residual QQ plot (matplotlib)
def plot_resid_qq():
    fig, ax = plt.subplots(figsize=(6,6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot of Residuals")
    return fig

# Residual ACF plot (matplotlib)
def plot_resid_acf():
    fig, ax = plt.subplots(figsize=(10,4))
    plot_acf(residuals, ax=ax, lags=40)
    ax.set_title("Autocorrelation (ACF) of Residuals")
    return fig

# Lookup historical sales for a date
def lookup_sales(date):
    date = pd.to_datetime(date)
    if date not in df.index:
        return f"No data for {date.date()}"
    sales = df.loc[date, "sales"]
    return f"Sales on {date.date()}: ${sales:.2f}"

# Download CSV of forecast
def get_csv():
    download_df = pd.DataFrame({
        "date": forecast_rounded.index,
        "forecasted_sales": forecast_rounded.values,
        "ci_lower_90": conf_int_rounded.iloc[:, 0].values,
        "ci_upper_90": conf_int_rounded.iloc[:, 1].values
    }).set_index("date")
    return download_df.to_csv().encode('utf-8')

# Define Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🍫 Chocolate Sales Forecast (Optimized ARIMA)")
    gr.Markdown("![Logo](https://i.imgur.com/oDM4ECC.jpeg)")

    with gr.Tab("2025 Forecast & Summary"):
        forecast_plot = gr.Plot(value=plot_forecast())
        date_input = gr.Date(label="Select a week in 2025", value=forecast_rounded.index.min().date(),
                             minimum=forecast_rounded.index.min().date(),
                             maximum=forecast_rounded.index.max().date())

        forecast_output = gr.Textbox(label="Forecasted Sales", interactive=False)
        ci_output = gr.Textbox(label="90% Confidence Interval", interactive=False)

        def update_forecast(date):
            date = pd.to_datetime(date)
            if date not in forecast_rounded.index:
                return "Invalid date", ""
            val = forecast_rounded[date]
            ci = conf_int_rounded.loc[date]
            ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
            return f"${val:.2f}", ci_str

        date_input.change(update_forecast, inputs=date_input, outputs=[forecast_output, ci_output])

        gr.Markdown("### 2025 Forecast Summary")
        total_sales = forecast_rounded.sum()
        avg_sales = forecast_rounded.mean()
        min_sales = forecast_rounded.min()
        max_sales = forecast_rounded.max()
        min_week = forecast_rounded.idxmin().date()
        max_week = forecast_rounded.idxmax().date()

        with gr.Row():
            gr.Text(f"**Total Forecast Sales:** {total_sales:,.2f}")
            gr.Text(f"**Average Weekly Sales:** {avg_sales:.2f}")
            gr.Text(f"**Min Weekly Sales:** {min_sales:.2f} (Week of {min_week})")
            gr.Text(f"**Max Weekly Sales:** {max_sales:.2f} (Week of {max_week})")

        gr.DownloadButton("Download 2025 Forecast as CSV", get_csv, file_name="chocolate_sales_forecast_2025.csv")

    with gr.Tab("2024 Model Evaluation"):
        gr.Markdown("### Model Performance on 2024 Actual Data")
        with gr.Row():
            gr.Text(f"R²: {r2:.4f}")
            gr.Text(f"RMSE: {rmse:.2f}")
            gr.Text(f"MAE: {mae:.2f}")
            gr.Text(f"MAPE: {mape:.2f}%")

        eval_plot = gr.Plot(value=plot_evaluation())

    with gr.Tab("Residual Diagnostics"):
        resid_plot = gr.Plot(value=plot_residuals())
        resid_hist = gr.Plot(value=plot_resid_hist())
        resid_qq = gr.Plot(value=plot_resid_qq())
        resid_acf = gr.Plot(value=plot_resid_acf())

    with gr.Tab("Historical Sales Lookup"):
        lookup_date = gr.Date(label="Select a date to view historical sales",
                             value=df.index.max().date(),
                             minimum=df.index.min().date(),
                             maximum=df.index.max().date())
        lookup_output = gr.Textbox(label="Sales on selected date", interactive=False)

        lookup_date.change(lookup_sales, inputs=lookup_date, outputs=lookup_output)

if __name__ == "__main__":
    demo.launch()
