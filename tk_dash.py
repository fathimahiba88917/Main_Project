
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ============================================================
# MAKEUP & COSMETICS ANALYTICS - TKINTER DASHBOARD
# Data files expected in the SAME folder as this Python file:
# Customer_500.csv
# Products_100.csv
# orders2500.csv
# orderdetails.csv
#
# IMPORTANT:
# Click "Refresh Data" after replacing/updating the CSV files.
# All KPIs, charts and the forecast will then be recalculated.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BG = "#F5F7FA"
SIDEBAR = "#111827"
SIDEBAR_2 = "#1F2937"
CARD = "#FFFFFF"
TEXT = "#172033"
MUTED = "#6B7280"
BORDER = "#E5E7EB"

NAV_COLORS = ["#7C3AED","#2563EB","#DB2777","#059669","#D97706","#DC2626","#0891B2","#4F46E5"]
NAV_HOVER = ["#6D28D9","#1D4ED8","#BE185D","#047857","#B45309","#B91C1C","#0E7490","#4338CA"]

ACCENT = "#C85C86"
ACCENT_DARK = "#A83E68"
SUCCESS = "#059669"
WARNING = "#D97706"

customers = products = orders = order_details = sales = monthly = None
forecast_monthly = None
forecast_yearly = None
forecast_model_name = ""


def read_data():
    """Read the latest CSV files every time Refresh Data is clicked."""
    global customers, products, orders, order_details, sales, monthly
    global forecast_monthly, forecast_yearly, forecast_model_name

    files = {
        "Customers": "Customer_500.csv",
        "Products": "Products_100.csv",
        "Orders": "orders2500.csv",
        "Order Details": "orderdetails.csv",
    }

    missing = [name for name, filename in files.items()
               if not (BASE_DIR / filename).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing CSV file(s):\n\n" +
            "\n".join(f"• {x}" for x in missing) +
            "\n\nPut all 4 CSV files in the same folder as this Python file."
        )

    customers = pd.read_csv(BASE_DIR / files["Customers"])
    products = pd.read_csv(BASE_DIR / files["Products"])
    orders = pd.read_csv(BASE_DIR / files["Orders"])
    order_details = pd.read_csv(BASE_DIR / files["Order Details"])

    def clean(df):
        df = df.copy()
        df.columns = (
            df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )
        return df

    customers = clean(customers)
    products = clean(products)
    orders = clean(orders)
    order_details = clean(order_details)

    for df, cols in [
        (customers, ["Customer_ID"]),
        (orders, ["Order_ID", "Customer_ID"]),
        (products, ["Product_ID"]),
        (order_details, ["Order_Detail_ID", "Order_ID", "Product_ID"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

    # Convert numeric columns safely so newly-added data is handled.
    for df, cols in [
        (products, ["Price", "Cost_Price", "Rating", "Stock_Quantity", "Shade_Count"]),
        (order_details, ["Quantity", "Discount_Percent"]),
        (customers, ["Age"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    sales = (
        orders.merge(order_details, on="Order_ID", how="inner")
        .merge(products, on="Product_ID", how="inner")
        .merge(
            customers[["Customer_ID", "Customer_Name", "Gender", "City", "State"]],
            on="Customer_ID",
            how="left",
        )
    )

    sales["Order_Date"] = pd.to_datetime(sales["Order_Date"], errors="coerce")
    sales["Gross_Sales"] = sales["Price"] * sales["Quantity"]
    sales["Discount_Amount"] = sales["Gross_Sales"] * sales["Discount_Percent"] / 100
    sales["Net_Sales"] = sales["Gross_Sales"] - sales["Discount_Amount"]
    sales["Total_Cost"] = sales["Cost_Price"] * sales["Quantity"]
    sales["Profit"] = sales["Net_Sales"] - sales["Total_Cost"]

    valid_dates = sales.dropna(subset=["Order_Date"]).copy()
    if valid_dates.empty:
        raise ValueError("No valid Order_Date values were found in the latest data.")

    monthly = (
        valid_dates.assign(Month=valid_dates["Order_Date"].dt.to_period("M"))
        .groupby("Month", as_index=False)["Net_Sales"]
        .sum()
        .sort_values("Month")
    )

    build_forecast()


def build_forecast():
    """
    Rebuild the forecast from the CURRENT monthly project data.
    Forecast horizon: through December 2030.
    Model: trend + month-of-year seasonality using least squares.
    This is deliberately simple and explainable for a student project.
    """
    global forecast_monthly, forecast_yearly, forecast_model_name

    hist = monthly.copy()
    hist["Date"] = hist["Month"].dt.to_timestamp()
    hist["t"] = np.arange(len(hist), dtype=float)

    # Use a linear trend plus 11 month dummy variables when enough
    # observations are available. This makes the forecast more useful
    # than a straight line while remaining easy to explain in viva.
    y = hist["Net_Sales"].astype(float).to_numpy()
    t = hist["t"].to_numpy()

    month_dummies = pd.get_dummies(
        hist["Date"].dt.month, prefix="m", drop_first=True, dtype=float
    )

    X = pd.DataFrame({"trend": t})
    X = pd.concat([X, month_dummies.reset_index(drop=True)], axis=1)

    # If there are too few observations for the seasonal model,
    # automatically fall back to a simple linear trend.
    if len(hist) >= max(12, len(X.columns) + 2):
        model = np.linalg.lstsq(
            np.column_stack([np.ones(len(X)), X.to_numpy()]), y, rcond=None
        )[0]
        forecast_model_name = "Trend + monthly seasonality (least squares)"
        last_t = len(hist) - 1

        future_dates = pd.date_range(
            hist["Date"].max() + pd.offsets.MonthBegin(1),
            "2030-12-01",
            freq="MS",
        )
        future_t = np.arange(last_t + 1, last_t + 1 + len(future_dates), dtype=float)

        future_month_dummies = pd.get_dummies(
            future_dates.month, prefix="m", drop_first=True, dtype=float
        )
        future_month_dummies = future_month_dummies.reindex(
            columns=month_dummies.columns, fill_value=0
        )

        Xf = pd.DataFrame({"trend": future_t})
        Xf = pd.concat([Xf, future_month_dummies.reset_index(drop=True)], axis=1)
        pred = np.column_stack([np.ones(len(Xf)), Xf.to_numpy()]) @ model
    else:
        # Fallback for very small new datasets.
        forecast_model_name = "Linear trend (least squares)"
        coef = np.polyfit(t, y, 1) if len(hist) >= 2 else [0, y[-1]]
        future_dates = pd.date_range(
            hist["Date"].max() + pd.offsets.MonthBegin(1),
            "2030-12-01",
            freq="MS",
        )
        future_t = np.arange(len(hist), len(hist) + len(future_dates), dtype=float)
        pred = np.polyval(coef, future_t)

    # Sales cannot logically be negative.
    pred = np.maximum(pred, 0)

    forecast_monthly = pd.DataFrame({
        "Date": future_dates,
        "Month": future_dates.to_period("M"),
        "Predicted_Sales": pred,
    })

    forecast_yearly = (
        forecast_monthly.assign(Year=forecast_monthly["Date"].dt.year)
        .groupby("Year", as_index=False)["Predicted_Sales"]
        .sum()
    )


# --------------------------- UI helpers ---------------------------

root = tk.Tk()
root.title("Makeup & Cosmetics Analytics Dashboard")
root.geometry("1500x900")
root.minsize(1200, 760)

# Start maximized on Windows so charts have enough horizontal and vertical space.
try:
    root.state("zoomed")
except tk.TclError:
    pass
root.configure(bg=BG)

header = tk.Frame(root, bg=SIDEBAR, height=92)
header.pack(fill="x")
header.pack_propagate(False)

title_frame = tk.Frame(header, bg=SIDEBAR)
title_frame.pack(side="left", padx=28, pady=12)

tk.Label(
    title_frame,
    text="MAKEUP & COSMETICS ANALYTICS",
    font=("Segoe UI", 21, "bold"),
    fg="white",
    bg=SIDEBAR,
).pack(anchor="w")

tk.Label(
    title_frame,
    text="Sales • Customers • Products • Brands • Profitability • Prediction",
    font=("Segoe UI", 9),
    fg="#C8D0D6",
    bg=SIDEBAR,
).pack(anchor="w", pady=(3, 0))

status_var = tk.StringVar(value="Loading latest project data...")
tk.Label(
    header,
    textvariable=status_var,
    font=("Segoe UI", 9, "bold"),
    fg="#E7B5C5",
    bg=SIDEBAR,
).pack(side="right", padx=28)


body = tk.Frame(root, bg=BG)
body.pack(fill="both", expand=True)

sidebar = tk.Frame(body, bg=SIDEBAR, width=245)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

# Scrollable main workspace so every chart keeps a readable size.
content_shell = tk.Frame(body, bg=BG)
content_shell.pack(side="right", fill="both", expand=True)

content_canvas = tk.Canvas(
    content_shell,
    bg=BG,
    highlightthickness=0,
    bd=0
)
content_scrollbar = tk.Scrollbar(
    content_shell,
    orient="vertical",
    command=content_canvas.yview,
    width=10
)
content_canvas.configure(yscrollcommand=content_scrollbar.set)

content_scrollbar.pack(side="right", fill="y")
content_canvas.pack(side="left", fill="both", expand=True)

content = tk.Frame(content_canvas, bg=BG)
content_window = content_canvas.create_window(
    (0, 0),
    window=content,
    anchor="nw"
)

def _resize_content(event=None):
    content_canvas.itemconfigure(content_window, width=content_canvas.winfo_width())
    content_canvas.configure(scrollregion=content_canvas.bbox("all"))

content.bind("<Configure>", _resize_content)
content_canvas.bind("<Configure>", _resize_content)

def _mousewheel(event):
    content_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

content_canvas.bind("<MouseWheel>", _mousewheel)
content.bind("<MouseWheel>", _mousewheel)


def clear_content():
    for w in content.winfo_children():
        w.destroy()


def heading(title, subtitle=""):
    top = tk.Frame(content, bg=BG)
    top.pack(fill="x", padx=30, pady=(22, 8))

    tk.Label(
        top, text=title, font=("Segoe UI", 22, "bold"),
        fg=TEXT, bg=BG
    ).pack(anchor="w")

    if subtitle:
        tk.Label(
            top, text=subtitle, font=("Segoe UI", 10),
            fg=MUTED, bg=BG
        ).pack(anchor="w", pady=(4, 0))

    # Recalculate scrollable content after the page is built.
    content.after_idle(lambda: content_canvas.configure(
        scrollregion=content_canvas.bbox("all")
    ))


def make_card(parent, label, value, column, accent=ACCENT):
    parent.columnconfigure(column, weight=1)
    f = tk.Frame(
        parent, bg=CARD,
        highlightbackground=BORDER,
        highlightthickness=1
    )
    f.grid(row=0, column=column, padx=7, pady=7, sticky="nsew")

    tk.Label(
        f, text=label, font=("Segoe UI", 8, "bold"),
        fg=MUTED, bg=CARD
    ).pack(anchor="w", padx=15, pady=(13, 2))

    tk.Label(
        f, text=value, font=("Segoe UI", 17, "bold"),
        fg=accent, bg=CARD
    ).pack(anchor="w", padx=15, pady=(0, 13))


def plot_into(parent, fig):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()

    widget = canvas.get_tk_widget()
    widget.configure(
        highlightthickness=0,
        bd=0
    )

    widget.pack(
        fill="both",
        expand=True,
        padx=4,
        pady=4
    )

    return canvas


def make_plot_area():
    frame = tk.Frame(content, bg=BG)
    frame.pack(fill="both", expand=False, padx=22, pady=(2, 14))
    return frame


def style_axes(ax):
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", alpha=0.12, linewidth=0.7)
    ax.tick_params(colors="#4B5563", labelsize=8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")


def finish_chart(fig, bottom=0.20, left=0.10, right=0.96, top=0.90):
    # Fixed margins prevent x-axis titles/tick labels from being clipped.
    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top
    )


def home():
    clear_content()
    heading(
        "Executive Dashboard",
        "Live view of the current CSV data. Use Refresh Data whenever you add or replace records."
    )

    top = tk.Frame(content, bg=BG)
    top.pack(fill="x", padx=22)

    total_sales = sales["Net_Sales"].sum()
    total_profit = sales["Profit"].sum()
    total_orders = sales["Order_ID"].nunique()
    total_customers = sales["Customer_ID"].nunique()
    units = sales["Quantity"].sum()
    margin = (total_profit / total_sales * 100) if total_sales else 0
    aov = total_sales / total_orders if total_orders else 0

    make_card(top, "TOTAL SALES", f"₹{total_sales:,.2f}", 0)
    make_card(top, "TOTAL PROFIT", f"₹{total_profit:,.2f}", 1)
    make_card(top, "TOTAL ORDERS", f"{total_orders:,}", 2)
    make_card(top, "CUSTOMERS", f"{total_customers:,}", 3)
    make_card(top, "UNITS SOLD", f"{units:,.0f}", 4)

    lower = tk.Frame(content, bg=BG)
    lower.pack(fill="both", expand=True, padx=30, pady=10)

    left = tk.Frame(lower, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
    left.pack(side="left", fill="both", expand=True, padx=(0, 7))

    right = tk.Frame(lower, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
    right.pack(side="right", fill="both", expand=True, padx=(7, 0))

    tk.Label(
        left, text="Project Overview", font=("Segoe UI", 16, "bold"),
        fg=TEXT, bg=CARD
    ).pack(anchor="w", padx=20, pady=(18, 8))

    overview = (
        "This dashboard presents the project dimensions described in the abstract: "
        "sales performance, customer purchasing behavior, product and brand performance, "
        "profitability, discounts, ratings and regional performance.\n\n"
        "Excel → data cleaning and preparation\n"
        "SQL → relational analysis and business queries\n"
        "Python → EDA, calculations, visualization and forecasting\n"
        "Power BI → interactive reporting\n"
        "Tkinter → interactive desktop presentation layer"
    )
    tk.Label(
        left, text=overview, justify="left", wraplength=470,
        font=("Segoe UI", 10), fg=TEXT, bg=CARD
    ).pack(anchor="w", padx=20)

    insights = (
        f"Profit Margin     {margin:.2f}%\n"
        f"Average Order     ₹{aov:,.2f}\n"
        f"Highest State     {sales.groupby('State')['Net_Sales'].sum().idxmax()}\n"
        f"Highest Brand     {sales.groupby('Brand')['Net_Sales'].sum().idxmax()}\n"
        f"Latest Data Month {sales['Order_Date'].max().strftime('%b %Y')}"
    )
    tk.Label(
        right, text="Quick Insights", font=("Segoe UI", 16, "bold"),
        fg=TEXT, bg=CARD
    ).pack(anchor="w", padx=20, pady=(18, 12))
    tk.Label(
        right, text=insights, justify="left",
        font=("Segoe UI", 11), fg=TEXT, bg=CARD
    ).pack(anchor="w", padx=20)


def sales_page():
    clear_content()

    heading(
        "Sales Performance",
        "Monthly sales trend and payment performance from the current project data."
    )

    # Monthly sales — give the plot enough height and bottom margin
    # so every date label and the X-axis title is fully visible.
    monthly_card = tk.Frame(
        content, bg=CARD, highlightbackground=BORDER,
        highlightthickness=1, height=330
    )
    monthly_card.pack(fill="x", padx=28, pady=(4, 12))
    monthly_card.pack_propagate(False)

    monthly_frame = tk.Frame(monthly_card, bg=CARD)
    monthly_frame.pack(fill="both", expand=True, padx=5, pady=5)

    fig, ax = plt.subplots(figsize=(13.2, 2.55), dpi=100)
    dates = monthly["Month"].dt.to_timestamp()

    ax.plot(
        dates, monthly["Net_Sales"],
        color="#2563EB", marker="o", markersize=3.5,
        linewidth=2.2
    )
    ax.set_title("Monthly Net Sales", fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Month", labelpad=8)
    ax.set_ylabel("Net Sales (₹)", labelpad=7)

    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", labelrotation=30, labelsize=8, pad=3)

    style_axes(ax)
    finish_chart(fig, bottom=0.29, left=0.075, right=0.985, top=0.90)
    plot_into(monthly_frame, fig)
    plt.close(fig)

    # Payment method chart — also keep the lower tick labels inside the card.
    payment_card = tk.Frame(
        content, bg=CARD, highlightbackground=BORDER,
        highlightthickness=1, height=330
    )
    payment_card.pack(fill="x", padx=28, pady=(0, 12))
    payment_card.pack_propagate(False)

    payment_frame = tk.Frame(payment_card, bg=CARD)
    payment_frame.pack(fill="both", expand=True, padx=5, pady=5)

    payment = (
        sales.groupby("Payment_Method", as_index=False)["Net_Sales"]
        .sum()
        .sort_values("Net_Sales", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(13.2, 2.55), dpi=100)
    bars = ax.barh(
        payment["Payment_Method"],
        payment["Net_Sales"],
        color=["#F59E0B", "#059669", "#DB2777", "#2563EB", "#7C3AED"]
    )
    ax.invert_yaxis()
    ax.set_title("Net Sales by Payment Method", fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Net Sales (₹)", labelpad=8)

    xmax = float(payment["Net_Sales"].max())
    ax.set_xlim(0, xmax * 1.12)

    for bar, value in zip(bars, payment["Net_Sales"]):
        ax.text(
            bar.get_width() + xmax * 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"₹{value/1_000_000:.2f}M",
            va="center", fontsize=8, fontweight="bold"
        )

    style_axes(ax)
    finish_chart(fig, bottom=0.20, left=0.13, right=0.965, top=0.88)
    plot_into(payment_frame, fig)
    plt.close(fig)

    tk.Label(
        content,
        text="All values are calculated directly from the project CSV data.",
        font=("Segoe UI", 8), fg=MUTED, bg=BG
    ).pack(anchor="w", padx=32, pady=(0, 16))

def product_page():
    clear_content()
    heading(
        "Product & Brand Performance",
        "Top products, categories and brands based on project sales data."
    )

    area = make_plot_area()

    top_row = tk.Frame(area, bg=BG, height=310)
    top_row.pack(fill="x", pady=(0, 12))
    top_row.pack_propagate(False)

    bottom_row = tk.Frame(area, bg=BG, height=290)
    bottom_row.pack(fill="x", pady=(0, 12))
    bottom_row.pack_propagate(False)

    p_panel = tk.Frame(
        top_row, bg=CARD, highlightbackground=BORDER, highlightthickness=1
    )
    p_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))

    c_panel = tk.Frame(
        top_row, bg=CARD, highlightbackground=BORDER, highlightthickness=1
    )
    c_panel.pack(side="right", fill="both", expand=True, padx=(6, 0))

    b_panel = tk.Frame(
        bottom_row, bg=CARD, highlightbackground=BORDER, highlightthickness=1
    )
    b_panel.pack(fill="both", expand=True)

    # Top products
    p = (
        sales.groupby("Product_Name")["Net_Sales"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(7.0, 2.75), dpi=100)
    ax.barh(p.index, p.values, color="#DB2777")
    ax.set_title("Top 10 Products by Net Sales", fontweight="bold", fontsize=12)
    ax.set_xlabel("Net Sales (₹)")
    ax.tick_params(axis="y", labelsize=7.5)
    style_axes(ax)
    finish_chart(fig, bottom=0.18, left=0.34, right=0.96, top=0.88)
    plot_into(p_panel, fig)
    plt.close(fig)

    # Categories: horizontal so category names never overlap.
    c = (
        sales.groupby("Category")["Net_Sales"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(7.0, 2.75), dpi=100)
    ax.barh(c.index, c.values, color="#7C3AED")
    ax.set_title("Top Categories by Net Sales", fontweight="bold", fontsize=12)
    ax.set_xlabel("Net Sales (₹)")
    ax.tick_params(axis="y", labelsize=8)
    style_axes(ax)
    finish_chart(fig, bottom=0.18, left=0.28, right=0.96, top=0.88)
    plot_into(c_panel, fig)
    plt.close(fig)

    # Brands
    b = (
        sales.groupby("Brand")["Net_Sales"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(12.5, 2.55), dpi=100)
    ax.barh(b.index, b.values, color="#059669")
    ax.set_title("Top 10 Brands by Net Sales", fontweight="bold", fontsize=12)
    ax.set_xlabel("Net Sales (₹)")
    ax.tick_params(axis="y", labelsize=8)
    style_axes(ax)
    finish_chart(fig, bottom=0.18, left=0.22, right=0.96, top=0.88)
    plot_into(b_panel, fig)
    plt.close(fig)


def customer_page():
    clear_content()
    heading(
        "Customer Analytics",
        "Demographics and purchasing behaviour of project customers."
    )

    area = make_plot_area()

    top_row = tk.Frame(area, bg=BG, height=390)
    top_row.pack(fill="x", pady=(0, 12))
    top_row.pack_propagate(False)

    bottom_row = tk.Frame(area, bg=BG, height=300)
    bottom_row.pack(fill="x", pady=(0, 12))
    bottom_row.pack_propagate(False)

    g_panel = tk.Frame(
        top_row, bg=CARD, highlightbackground=BORDER, highlightthickness=1
    )
    g_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))

    age_panel = tk.Frame(
        top_row, bg=CARD, highlightbackground=BORDER, highlightthickness=1
    )
    age_panel.pack(side="right", fill="both", expand=True, padx=(6, 0))

    repeat_panel = tk.Frame(
        bottom_row, bg=CARD, highlightbackground=BORDER, highlightthickness=1
    )
    repeat_panel.pack(fill="both", expand=True)

    g = customers["Gender"].value_counts()

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=100)
    ax.bar(g.index, g.values, color="#059669")
    ax.set_title("Customers by Gender", fontweight="bold", fontsize=12)
    ax.set_ylabel("Customer Count")
    style_axes(ax)
    finish_chart(fig, bottom=0.20, left=0.10, right=0.96, top=0.88)
    plot_into(g_panel, fig)
    plt.close(fig)

    labels = ["18-25", "26-35", "36-45", "46-55"]

    age = (
        pd.cut(
            customers["Age"],
            bins=[17, 25, 35, 45, 55],
            labels=labels
        )
        .value_counts()
        .reindex(labels)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=100)
    ax.bar(age.index.astype(str), age.values, color="#D97706")
    ax.set_title("Customers by Age Group", fontweight="bold", fontsize=12)
    ax.set_ylabel("Customer Count")
    style_axes(ax)
    finish_chart(fig, bottom=0.20, left=0.10, right=0.96, top=0.88)
    plot_into(age_panel, fig)
    plt.close(fig)

    # Customer type is calculated from Orders, not from sales rows.
    order_counts = orders.groupby("Customer_ID")["Order_ID"].nunique()

    repeat = pd.Series({
        "Repeat Customer": int((order_counts > 1).sum()),
        "One-Time Customer": int((order_counts == 1).sum())
    })

    fig, ax = plt.subplots(figsize=(12.5, 2.7), dpi=100)

    bars = ax.bar(
        repeat.index,
        repeat.values,
        color=["#2563EB", "#DB2777"],
        width=0.55
    )

    ax.set_title(
        "Repeat vs One-Time Customers",
        fontweight="bold",
        fontsize=12
    )
    ax.set_ylabel("Customer Count")

    for bar, val in zip(bars, repeat.values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            val,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    style_axes(ax)
    finish_chart(fig, bottom=0.22, left=0.08, right=0.96, top=0.87)
    plot_into(repeat_panel, fig)
    plt.close(fig)


def regional_page():
    clear_content()
    heading(
        "Regional Performance",
        "State and city sales using the location fields in the project."
    )

    area = make_plot_area()

    row = tk.Frame(area, bg=BG, height=410)
    row.pack(fill="x", pady=(0, 12))
    row.pack_propagate(False)

    spanel = tk.Frame(
        row, bg=CARD, highlightbackground=BORDER, highlightthickness=1
    )
    spanel.pack(side="left", fill="both", expand=True, padx=(0, 6))

    cpanel = tk.Frame(
        row, bg=CARD, highlightbackground=BORDER, highlightthickness=1
    )
    cpanel.pack(side="right", fill="both", expand=True, padx=(6, 0))

    st = (
        sales.groupby("State")["Net_Sales"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=100)
    ax.barh(st.index, st.values, color="#059669")
    ax.set_title("Top 10 States by Sales", fontweight="bold", fontsize=12)
    ax.set_xlabel("Net Sales (₹)")
    ax.tick_params(axis="y", labelsize=8)
    style_axes(ax)
    finish_chart(fig, bottom=0.18, left=0.28, right=0.96, top=0.88)
    plot_into(spanel, fig)
    plt.close(fig)

    c = (
        sales.groupby("City")["Net_Sales"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=100)
    ax.barh(c.index, c.values, color="#D97706")
    ax.set_title("Top 10 Cities by Sales", fontweight="bold", fontsize=12)
    ax.set_xlabel("Net Sales (₹)")
    ax.tick_params(axis="y", labelsize=8)
    style_axes(ax)
    finish_chart(fig, bottom=0.18, left=0.30, right=0.96, top=0.88)
    plot_into(cpanel, fig)
    plt.close(fig)


def profitability_page():
    clear_content()

    heading(
        "Profitability & Pricing",
        "Profitability by category and the relationship between product price and rating."
    )

    cards = tk.Frame(content, bg=BG)
    cards.pack(fill="x", padx=22, pady=(0, 8))

    total_sales = sales["Net_Sales"].sum()
    total_profit = sales["Profit"].sum()
    margin = total_profit / total_sales * 100 if total_sales else 0

    make_card(cards, "TOTAL SALES", f"₹{total_sales:,.2f}", 0, "#2563EB")
    make_card(cards, "TOTAL PROFIT", f"₹{total_profit:,.2f}", 1, "#059669")
    make_card(cards, "PROFIT MARGIN", f"{margin:.2f}%", 2, "#DB2777")

    area = make_plot_area()

    row = tk.Frame(area, bg=BG, height=410)
    row.pack(fill="x", pady=(0, 12))
    row.pack_propagate(False)

    cpanel = tk.Frame(
        row, bg=CARD, highlightbackground=BORDER, highlightthickness=1
    )
    cpanel.pack(side="left", fill="both", expand=True, padx=(0, 6))

    rpanel = tk.Frame(
        row, bg=CARD, highlightbackground=BORDER, highlightthickness=1
    )
    rpanel.pack(side="right", fill="both", expand=True, padx=(6, 0))

    c = (
        sales.groupby("Category")["Profit"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=100)
    ax.barh(c.index, c.values, color="#DC2626")
    ax.set_title("Top 10 Categories by Profit", fontweight="bold", fontsize=12)
    ax.set_xlabel("Profit (₹)")
    ax.tick_params(axis="y", labelsize=8)
    style_axes(ax)
    finish_chart(fig, bottom=0.18, left=0.28, right=0.96, top=0.88)
    plot_into(cpanel, fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=100)
    ax.scatter(
        products["Price"],
        products["Rating"],
        alpha=0.7,
        color="#7C3AED",
        s=42,
        edgecolors="white",
        linewidth=0.5
    )

    ax.set_title("Product Price vs Rating", fontweight="bold", fontsize=12)
    ax.set_xlabel("Price (₹)")
    ax.set_ylabel("Rating")
    ax.grid(alpha=0.12)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    finish_chart(fig, bottom=0.20, left=0.10, right=0.96, top=0.88)
    plot_into(rpanel, fig)
    plt.close(fig)


def demand_page():
    clear_content()

    heading(
        "Discount & Demand",
        "Discount behaviour, stock levels and quantity sold."
    )

    area = make_plot_area()

    top_row = tk.Frame(area, bg=BG, height=310)
    top_row.pack(fill="x", pady=(0, 12))
    top_row.pack_propagate(False)

    bottom_row = tk.Frame(area, bg=BG, height=290)
    bottom_row.pack(fill="x", pady=(0, 12))
    bottom_row.pack_propagate(False)

    discount_panel = tk.Frame(
        top_row, bg=CARD, highlightbackground=BORDER, highlightthickness=1
    )
    discount_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))

    qty_panel = tk.Frame(
        top_row, bg=CARD, highlightbackground=BORDER, highlightthickness=1
    )
    qty_panel.pack(side="right", fill="both", expand=True, padx=(6, 0))

    stock_panel = tk.Frame(
        bottom_row, bg=CARD, highlightbackground=BORDER, highlightthickness=1
    )
    stock_panel.pack(fill="both", expand=True)

    # Discount chart
    d = (
        sales.groupby("Discount_Percent")["Net_Sales"]
        .sum()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(7.0, 2.75), dpi=100)
    ax.bar(
        d.index.astype(str),
        d.values,
        color="#0891B2",
        width=0.65
    )

    ax.set_title("Net Sales by Discount %", fontweight="bold", fontsize=12)
    ax.set_xlabel("Discount (%)")
    ax.set_ylabel("Net Sales (₹)")
    ax.tick_params(axis="x", labelsize=8)

    style_axes(ax)
    finish_chart(fig, bottom=0.20, left=0.11, right=0.96, top=0.88)
    plot_into(discount_panel, fig)
    plt.close(fig)

    # Horizontal categories: this avoids the overlapping category labels.
    qty = (
        sales.groupby("Category")["Quantity"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(7.0, 2.75), dpi=100)
    bars = ax.barh(
        qty.index,
        qty.values,
        color="#D97706"
    )

    ax.set_title(
        "Top 10 Categories by Units Sold",
        fontweight="bold",
        fontsize=12
    )
    ax.set_xlabel("Units")
    ax.tick_params(axis="y", labelsize=8)

    for bar, val in zip(bars, qty.values):
        ax.text(
            val,
            bar.get_y() + bar.get_height()/2,
            f"{int(val):,}",
            va="center",
            ha="left",
            fontsize=7.5
        )

    style_axes(ax)
    finish_chart(fig, bottom=0.18, left=0.30, right=0.96, top=0.88)
    plot_into(qty_panel, fig)
    plt.close(fig)

    # Stock
    st = (
        products.nlargest(10, "Stock_Quantity")
        .sort_values("Stock_Quantity")
    )

    fig, ax = plt.subplots(figsize=(12.5, 2.55), dpi=100)
    ax.barh(
        st["Product_Name"],
        st["Stock_Quantity"],
        color="#4F46E5"
    )

    ax.set_title(
        "Top 10 Products by Current Stock",
        fontweight="bold",
        fontsize=12
    )
    ax.set_xlabel("Stock Quantity")
    ax.tick_params(axis="y", labelsize=7.5)

    style_axes(ax)
    finish_chart(fig, bottom=0.18, left=0.30, right=0.96, top=0.88)
    plot_into(stock_panel, fig)
    plt.close(fig)


def prediction_page():
    clear_content()

    heading(
        "Sales Prediction",
        "Dynamic monthly forecast through December 2030. "
        "Refresh Data to rebuild the model using the latest CSV records."
    )

    # Monthly forecast
    forecast_card = tk.Frame(
        content, bg=CARD, highlightbackground=BORDER,
        highlightthickness=1, height=330
    )
    forecast_card.pack(fill="x", padx=28, pady=(4, 12))
    forecast_card.pack_propagate(False)

    tk.Label(
        forecast_card,
        text="Monthly Sales Forecast Through 2030",
        font=("Segoe UI", 15, "bold"), fg=TEXT, bg=CARD
    ).pack(anchor="w", padx=18, pady=(12, 0))

    tk.Label(
        forecast_card,
        text="Blue = historical  •  Orange = forecast  •  Dotted = forecast start",
        font=("Segoe UI", 8), fg=MUTED, bg=CARD
    ).pack(anchor="w", padx=18, pady=(2, 2))

    chart_frame = tk.Frame(forecast_card, bg=CARD)
    chart_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

    historical = monthly.copy()
    historical["Date"] = historical["Month"].dt.to_timestamp()

    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(13.2, 2.55), dpi=100)
    ax.plot(
        historical["Date"], historical["Net_Sales"],
        color="#2563EB", marker="o", markersize=3.2,
        linewidth=2.0, label="Historical Sales"
    )
    ax.plot(
        forecast_monthly["Date"], forecast_monthly["Predicted_Sales"],
        color="#F97316", linestyle="--", linewidth=2.0,
        label="Forecast to Dec 2030"
    )
    ax.axvline(
        historical["Date"].max(),
        color="#6B7280", linestyle=":", linewidth=1.4,
        label="Forecast Begins"
    )

    ax.set_title(
        "Historical Sales vs Dynamic Forecast",
        fontsize=12, fontweight="bold", pad=7
    )
    ax.set_xlabel("Year", labelpad=7)
    ax.set_ylabel("Net Sales (₹)", labelpad=7)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=8, pad=3)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)

    style_axes(ax)
    finish_chart(fig, bottom=0.27, left=0.075, right=0.985, top=0.88)
    plot_into(chart_frame, fig)
    plt.close(fig)

    # Annual forecast — larger card and generous bottom margin.
    annual_card = tk.Frame(
        content, bg=CARD, highlightbackground=BORDER,
        highlightthickness=1, height=350
    )
    annual_card.pack(fill="x", padx=28, pady=(0, 12))
    annual_card.pack_propagate(False)

    tk.Label(
        annual_card,
        text="Predicted Annual Sales",
        font=("Segoe UI", 14, "bold"), fg=TEXT, bg=CARD
    ).pack(anchor="w", padx=16, pady=(12, 1))

    tk.Label(
        annual_card,
        text="Forecast totals generated from the current project data.",
        font=("Segoe UI", 8), fg=MUTED, bg=CARD
    ).pack(anchor="w", pady=(0, 2))

    annual_frame = tk.Frame(annual_card, bg=CARD)
    annual_frame.pack(fill="both", expand=True, padx=5, pady=(0, 7))

    years = forecast_yearly["Year"].astype(str)
    values = forecast_yearly["Predicted_Sales"]

    fig, ax = plt.subplots(figsize=(12.5, 2.65), dpi=100)

    palette = ["#60A5FA", "#34D399", "#FBBF24", "#FB7185", "#A78BFA"]
    bars = ax.bar(years, values, color=palette[:len(years)], width=0.58)

    ax.set_title(
        "Annual Forecast 2026–2030",
        fontsize=12, fontweight="bold", pad=8
    )
    ax.set_xlabel("Year", labelpad=9)
    ax.set_ylabel("Sales (₹)", labelpad=8)
    ax.tick_params(axis="x", labelsize=9, pad=5)

    ymax = float(values.max()) if len(values) else 1
    ax.set_ylim(0, ymax * 1.18)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.018,
            f"₹{value/1_000_000:.2f}M",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold"
        )

    style_axes(ax)
    finish_chart(fig, bottom=0.25, left=0.08, right=0.975, top=0.87)
    plot_into(annual_frame, fig)
    plt.close(fig)

    tk.Label(
        content,
        text=(
            "Prediction is recalculated from the latest CSV data whenever "
            "Refresh Data is clicked. The forecast is an analytical estimate."
        ),
        font=("Segoe UI", 8), fg=MUTED, bg=BG,
        wraplength=1300, justify="left"
    ).pack(anchor="w", padx=32, pady=(0, 16))

def heatmap_page():
    """Dedicated full-size correlation heatmap page."""
    clear_content()

    heading(
        "Correlation Heatmap",
        "Relationship between key numeric variables in the current cosmetics sales dataset."
    )

    # KPI cards make the page useful without shrinking the heatmap.
    corr = sales[
        [
            "Quantity",
            "Discount_Percent",
            "Price",
            "Cost_Price",
            "Rating",
            "Stock_Quantity",
            "Shade_Count",
            "Net_Sales",
            "Profit"
        ]
    ].corr()

    pairs = []
    cols = list(corr.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append(
                (cols[i], cols[j], abs(corr.iloc[i, j]), corr.iloc[i, j])
            )

    pairs.sort(key=lambda x: x[2], reverse=True)

    strongest = pairs[0] if pairs else ("N/A", "N/A", 0, 0)
    sales_profit = corr.loc["Net_Sales", "Profit"]
    price_cost = corr.loc["Price", "Cost_Price"]

    info = tk.Frame(content, bg=BG)
    info.pack(fill="x", padx=22, pady=(0, 10))

    make_card(
        info,
        "STRONGEST RELATIONSHIP",
        f"{strongest[0]} ↔ {strongest[1]}",
        0,
        "#7C3AED"
    )
    make_card(
        info,
        "STRONGEST CORRELATION",
        f"{strongest[3]:.2f}",
        1,
        "#DB2777"
    )
    make_card(
        info,
        "SALES ↔ PROFIT",
        f"{sales_profit:.2f}",
        2,
        "#059669"
    )
    make_card(
        info,
        "PRICE ↔ COST",
        f"{price_cost:.2f}",
        3,
        "#D97706"
    )

    # Large dedicated heatmap card.
    heatmap_card = tk.Frame(
        content,
        bg=CARD,
        highlightbackground=BORDER,
        highlightthickness=1,
        height=560
    )
    heatmap_card.pack(fill="x", padx=28, pady=(4, 12))
    heatmap_card.pack_propagate(False)

    tk.Label(
        heatmap_card,
        text="Correlation Matrix",
        font=("Segoe UI", 16, "bold"),
        fg=TEXT,
        bg=CARD
    ).pack(anchor="w", padx=18, pady=(12, 1))

    tk.Label(
        heatmap_card,
        text=(
            "Values range from -1 to +1. Values closer to +1 indicate a strong "
            "positive relationship; values closer to -1 indicate a strong negative relationship."
        ),
        font=("Segoe UI", 9),
        fg=MUTED,
        bg=CARD
    ).pack(anchor="w", padx=18, pady=(0, 4))

    hm_frame = tk.Frame(heatmap_card, bg=CARD)
    hm_frame.pack(fill="both", expand=True, padx=8, pady=4)

    fig, ax = plt.subplots(figsize=(12.8, 4.9), dpi=100)

    image = ax.imshow(
        corr,
        cmap="PuRd",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(
        cols,
        rotation=38,
        ha="right",
        fontsize=8
    )
    ax.set_yticklabels(
        cols,
        fontsize=8
    )

    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr.iloc[i, j]
            # White text on darker cells, dark text on lighter cells.
            text_color = "white" if abs(value) > 0.55 else "#111827"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=text_color
            )

    ax.set_title(
        "Correlation Matrix of Project Variables",
        fontsize=13,
        fontweight="bold",
        pad=8
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.025
    )
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Correlation", fontsize=8)

    finish_chart(
        fig,
        bottom=0.25,
        left=0.13,
        right=0.91,
        top=0.88
    )

    plot_into(hm_frame, fig)
    plt.close(fig)

    tk.Label(
        content,
        text=(
            "Interpretation: correlation shows association, not causation. "
            "Use the heatmap together with the sales, product, customer and profitability analyses."
        ),
        font=("Segoe UI", 9),
        fg=MUTED,
        bg=BG,
        wraplength=1300,
        justify="left"
    ).pack(anchor="w", padx=32, pady=(0, 16))

def refresh_data():
    try:
        read_data()
        status_var.set(
            f"Data refreshed • {len(customers):,} customers • "
            f"{len(products):,} products • {len(orders):,} orders • "
            f"Updated {pd.Timestamp.now().strftime('%d %b %Y %H:%M')}"
        )
        home()
        messagebox.showinfo(
            "Data Refreshed",
            "Latest CSV data has been loaded.\n\n"
            "All KPIs, charts and the prediction through 2030 have been recalculated."
        )
    except Exception as e:
        messagebox.showerror("Refresh Error", str(e))


# --------------------------- sidebar ---------------------------

tk.Label(
    sidebar,
    text="NAVIGATION",
    font=("Segoe UI", 9, "bold"),
    fg="#9DA9B3",
    bg=SIDEBAR
).pack(anchor="w", padx=22, pady=(22, 10))


nav_index = 0

nav_title = tk.Label(
    sidebar,
    text="NAVIGATION",
    font=("Segoe UI", 9, "bold"),
    fg="#9CA3AF",
    bg=SIDEBAR
)
nav_title.pack(anchor="w", padx=18, pady=(18, 8))

nav_area = tk.Frame(sidebar, bg=SIDEBAR)
nav_area.pack(fill="x", padx=10)

NAV_ITEMS = [
    ("▣   Sales Performance", sales_page),
    ("◈   Product & Brand", product_page),
    ("♙   Customer Analytics", customer_page),
    ("⌖   Regional Analysis", regional_page),
    ("₹   Profitability & Pricing", profitability_page),
    ("%   Discount & Demand", demand_page),
    ("↗   Sales Prediction", prediction_page),
    ("▤   Correlation Heatmap", heatmap_page),
]

for i in range(len(NAV_ITEMS)):
    nav_area.grid_rowconfigure(i, minsize=58)
nav_area.grid_columnconfigure(0, weight=1)


def nav_button(text, command):
    color = NAV_COLORS[nav_button.index]
    hover = NAV_HOVER[nav_button.index]

    button = tk.Button(
        nav_area,
        text=text,
        command=command,
        font=("Segoe UI", 10, "bold"),
        bg=color,
        fg="white",
        activebackground=hover,
        activeforeground="white",
        relief="flat",
        anchor="w",
        padx=18,
        pady=0,
        cursor="hand2",
        bd=0,
        highlightthickness=0
    )

    button.grid(
        row=nav_button.index,
        column=0,
        sticky="ew",
        padx=0,
        pady=3
    )

    nav_button.index += 1
    return button


nav_button.index = 0

for label, command in NAV_ITEMS:
    nav_button(label, command)


# Bottom controls stay fixed at the bottom of the sidebar.
controls = tk.Frame(sidebar, bg=SIDEBAR)
controls.pack(fill="x", side="bottom", padx=10, pady=12)

tk.Frame(
    controls,
    bg="#374151",
    height=1
).pack(fill="x", pady=(0, 10))

refresh_btn = tk.Button(
    controls,
    text="⟳   Refresh Data",
    command=refresh_data,
    font=("Segoe UI", 9, "bold"),
    bg="#334155",
    fg="#FFFFFF",
    activebackground="#0EA5E9",
    activeforeground="white",
    relief="flat",
    padx=10,
    pady=9,
    cursor="hand2",
    bd=0,
    highlightthickness=0,
)
refresh_btn.pack(fill="x", pady=(0, 6))

back_btn = tk.Button(
    controls,
    text="←   Back to Home",
    command=home,
    font=("Segoe UI", 9, "bold"),
    bg="#7C3AED",
    fg="white",
    activebackground="#6D28D9",
    activeforeground="white",
    relief="flat",
    padx=10,
    pady=9,
    cursor="hand2",
    bd=0,
    highlightthickness=0,
)
back_btn.pack(fill="x")


# --------------------------- startup ---------------------------

try:
    read_data()
    status_var.set(
        f"Live project data • {len(customers):,} customers • "
        f"{len(products):,} products • {len(orders):,} orders"
    )
    home()
except Exception as startup_error:
    status_var.set("Data loading failed")
    messagebox.showerror("Startup Error", str(startup_error))

root.mainloop()
