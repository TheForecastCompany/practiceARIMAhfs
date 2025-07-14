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

# ------------------------------ Train & Forecast ------------------------------
def train_and_forecast():
    train = df.iloc[:-52]
    test = df.iloc[-52:]
    order = (2, 0, 2)

    model = ARIMA(train["sales"], order=order)
    model_fit = model.fit()

    # Forecast next 52 weeks
    forecast_res = model_fit.get_forecast(steps=52, alpha=0.10)
    forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()
    dates_fc = pd.date_range(start="2025-01-05", periods=52, freq='W-SUN')
    forecast.index = dates_fc
    conf_int.index = dates_fc
    forecast_rounded = forecast.round(2)
    conf_int_rounded = conf_int.round(2)

    # Evaluation on test set
    eval_res = model_fit.get_forecast(steps=52, alpha=0.10)
    test_fc = eval_res.predicted_mean
    test_ci = eval_res.conf_int()
    test_fc.index = test.index
    test_fc_rounded = test_fc.round(2)
    test_ci_rounded = test_ci.round(2)

    r2 = r2_score(test["sales"], test_fc_rounded)
    mse = mean_squared_error(test["sales"], test_fc_rounded)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test["sales"], test_fc_rounded)
    mape = np.mean(np.abs((test["sales"] - test_fc_rounded) / test["sales"])) * 100

    return (
        forecast_rounded,
        conf_int_rounded,
        test_fc_rounded,
        test_ci_rounded,
        r2,
        rmse,
        mae,
        mape,
        model_fit,
        train.index,
        model_fit.resid
    )

(
    forecast_rounded,
    conf_int_rounded,
    test_fc_rounded,
    test_ci_rounded,
    r2,
    rmse,
    mae,
    mape,
    model_fit,
    train_index,
    residuals
) = train_and_forecast()

# ------------------------------ Plotting Helpers ------------------------------
def plot_forecast():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_rounded.index,
        y=forecast_rounded,
        mode="lines",
        name="Forecasted Sales"
    ))
    fig.add_trace(go.Scatter(
        x=list(forecast_rounded.index) + list(forecast_rounded.index[::-1]),
        y=list(conf_int_rounded.iloc[:,0]) + list(conf_int_rounded.iloc[:,1][::-1]),
        fill="toself",
        name="90% CI"
    ))
    fig.update_layout(
        title="2025 Forecast",
        xaxis_title="Week",
        yaxis_title="Sales ($)",
        hovermode="x unified"
    )
    return fig

# Similar helpers: plot_evaluation, plot_residuals, plot_hist, plot_qq, plot_acf omitted for brevity

def lookup_sales(date):
    date = pd.to_datetime(date)
    if date not in df.index:
        return f"No data for {date.date()}"
    sales = df.loc[date, "sales"]
    return f"Sales on {date.date()}: ${sales:.2f}"

# ------------------------------ Gradio UI ------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🍫 Chocolate Sales Forecast (ARIMA)")
    gr.Markdown("![Logo](https://i.imgur.com/oDM4ECC.jpeg)")

    with gr.Tab("2025 Forecast & Summary"):
        gr.Plot(value=plot_forecast)
        date_sel = gr.Date(
            label="Select a week",
            value=forecast_rounded.index.min().date(),
            minimum=forecast_rounded.index.min().date(),
            maximum=forecast_rounded.index.max().date()
        )
        out_sales = gr.Textbox(interactive=False, label="Sales")
        out_ci = gr.Textbox(interactive=False, label="90% CI")

        def update(date):
            d = pd.to_datetime(date)
            if d not in forecast_rounded.index:
                return "Invalid", ""
            v = forecast_rounded[d]
            ci = conf_int_rounded.loc[d]
            return f"${v:.2f}", f"[{ci[0]:.2f}, {ci[1]:.2f}]"

        date_sel.change(update, inputs=date_sel, outputs=[out_sales, out_ci])

    with gr.Tab("Historical Lookup"):
        hist_date = gr.Date(
            label="History date",
            value=df.index.max().date(),
            minimum=df.index.min().date(),
            maximum=df.index.max().date()
        )
        hist_out = gr.Textbox(interactive=False)
        hist_date.change(lookup_sales, inputs=hist_date, outputs=hist_out)

if __name__ == "__main__":
    demo.launch()

