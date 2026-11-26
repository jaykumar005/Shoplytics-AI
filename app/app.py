import os
import sqlite3
from datetime import date
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "ecommerce.db")
MODEL_PATH = os.path.join(ROOT, "models", "daily_sales_model.joblib")
SAMPLE_DIR = os.path.join(ROOT, "data", "sample")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

st.set_page_config(page_title="Shoplytics AI", page_icon="🛒", layout="wide")


st.markdown("""
<style>
:root { --ink:#172033; --muted:#64748b; --brand:#635bff; --brand2:#16b8a6; }
.stApp { background: linear-gradient(180deg, #f7f8ff 0%, #f5f7fb 45%, #f8fafc 100%); color:var(--ink); }
[data-testid="stHeader"] { background: rgba(255,255,255,.72); }
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1450px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#111827 0%,#202b48 100%); }
section[data-testid="stSidebar"] * { color:#f8fafc !important; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label { background:rgba(255,255,255,.06); border-radius:10px; padding:8px 10px; margin:3px 0; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(99,91,255,.35); }
.hero {
  padding: 26px 30px; border-radius: 22px; color:white; margin-bottom:22px;
  background: linear-gradient(115deg,#242a65 0%,#635bff 55%,#18b8a6 100%);
  box-shadow: 0 14px 35px rgba(50,55,130,.18);
}
.hero h1 { color:white; font-size:2rem; margin:0 0 7px 0; letter-spacing:-.7px; }
.hero p { color:rgba(255,255,255,.84); margin:0; font-size:1rem; }
.section-title { font-size:1.18rem; font-weight:750; color:#172033; margin:1.1rem 0 .65rem 0; }
.kpi {
  background:rgba(255,255,255,.92); border:1px solid #e7eaf3; border-radius:17px;
  padding:17px 18px; box-shadow:0 5px 18px rgba(31,41,55,.045); min-height:112px;
}
.kpi-label { color:#64748b; font-size:.82rem; font-weight:650; margin-bottom:8px; }
.kpi-value { color:#172033; font-size:1.65rem; font-weight:800; letter-spacing:-.6px; }
.kpi-note { color:#94a3b8; font-size:.76rem; margin-top:4px; }
div[data-testid="stMetric"] {
  background:white; border:1px solid #e7eaf3; padding:15px 17px; border-radius:16px;
  box-shadow:0 5px 18px rgba(31,41,55,.045);
}
/* Strong, accessible contrast for KPI labels and values */
div[data-testid="stMetric"] { background:#ffffff !important; }
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] label,
div[data-testid="stMetricLabel"] span,
div[data-testid="stMetricLabel"] * {
  color:#334155 !important; opacity:1 !important; font-weight:750 !important;
}
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div,
div[data-testid="stMetricValue"] span,
div[data-testid="stMetricValue"] * {
  color:#172033 !important; opacity:1 !important; font-weight:850 !important;
}
/* Keep upload controls light and readable across Streamlit themes */
div[data-testid="stFileUploader"],
div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
  background:#ffffff !important; color:#334155 !important;
  border-color:#cbd5e1 !important;
}
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] small {
  color:#334155 !important; opacity:1 !important;
}
div[data-testid="stFileUploader"] button {
  color:#ffffff !important; background:#4f46e5 !important; border:0 !important;
}
/* Clearer body copy and sidebar text */
.stApp, .stApp p, .stApp label, .stApp [data-testid="stCaptionContainer"] {
  color:#334155;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
  color:#cbd5e1 !important; opacity:1 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label,
section[data-testid="stSidebar"] [data-testid="stRadio"] label * {
  color:#ffffff !important; opacity:1 !important; font-weight:600 !important;
}
div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
  background:#ffffff !important; color:#172033 !important;
  border:1px solid #e7eaf3; border-radius:16px; padding:8px;
  box-shadow:0 5px 18px rgba(31,41,55,.035);
}
.stButton > button {
  border:0; border-radius:11px; color:white; font-weight:700;
  background:linear-gradient(100deg,#635bff,#5147df); padding:.62rem 1.1rem;
  box-shadow:0 5px 13px rgba(99,91,255,.2);
}
.stButton > button:hover { color:white; border:0; background:linear-gradient(100deg,#5147df,#16a995); }
div[data-testid="stFileUploader"] { background:white; border:1px dashed #b8c0d8; border-radius:15px; padding:10px; }
div[data-testid="stAlert"] {
    background-color: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    border-radius: 13px !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span {
    color: #3730a3 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}
h1,h2,h3 { letter-spacing:-.45px; }

.workspaces-title { color:#172033; font-size:1.25rem; font-weight:800; margin:1.35rem 0 .8rem; }
.workspace-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:0 0 1.35rem; }
.workspace-card { min-height:150px; padding:19px 18px; border-radius:18px; color:#fff; box-shadow:0 9px 22px rgba(35,45,80,.12); }
.workspace-card .icon { font-size:1.65rem; margin-bottom:12px; }
.workspace-card h3 { color:#fff; font-size:1.02rem; margin:0 0 7px; letter-spacing:-.2px; }
.workspace-card p { color:rgba(255,255,255,.9); font-size:.82rem; line-height:1.45; margin:0; }
.workspace-card.purple { background:linear-gradient(135deg,#635bff,#8b5cf6); }
.workspace-card.blue { background:linear-gradient(135deg,#2563eb,#38bdf8); }
.workspace-card.teal { background:linear-gradient(135deg,#0f9f91,#2dd4bf); }
.workspace-card.orange { background:linear-gradient(135deg,#ea580c,#f59e0b); }
@media (max-width: 1100px) { .workspace-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width: 560px) { .workspace-grid { grid-template-columns:1fr; } }

footer { visibility:hidden; }
.hero p {
    color: #ffffff !important;
    opacity: 1 !important;
    font-weight: 600 !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
}
div[data-testid="stFileUploader"] button,
div[data-testid="stFileUploader"] button *,
div[data-testid="stFileUploader"] button span {
    color: #ffffff !important;
    background-color: #4f46e5 !important;
    font-weight: 600 !important;
}
/* Highlight all CSV/Excel upload labels */
div[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] {
    color: #3730a3 !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    background: #eef2ff !important;
    padding: 9px 12px !important;
    border-left: 4px solid #6366f1 !important;
    border-radius: 6px !important;
    margin-bottom: 10px !important;
}
/* Fix Data Center information banner */
div[data-testid="stAlert"] {
    background-color: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    border-radius: 10px !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] * {
    color: #3730a3 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* Final Data Center alert color fix */
div[data-testid="stAlert"] {
    background: #eef2ff !important;
    background-color: #eef2ff !important;
}

div[data-testid="stAlert"] > div,
div[data-testid="stAlert"] [role="alert"] {
    background: #eef2ff !important;
    background-color: #eef2ff !important;
    border-color: #c7d2fe !important;
}

div[data-testid="stAlert"] * {
    color: #3730a3 !important;
    opacity: 1 !important;
}

</style>
""", unsafe_allow_html=True)


def conn():
    return sqlite3.connect(DB_PATH)

def table_exists(c, table):
    return c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None

def normalize_orders(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ["order_id", "order_date", "quantity", "unit_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Orders file missing required columns: " + ", ".join(missing))
    df["order_id"] = df["order_id"].astype(str).str.strip()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce").dt.date.astype("string")
    for col in ["quantity", "unit_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "customer_id" not in df: df["customer_id"] = None
    if "product_id" not in df: df["product_id"] = None
    df["data_issue"] = ""
    df.loc[df["order_id"].isna() | (df["order_id"] == ""), "data_issue"] += "missing_order_id;"
    df.loc[df["order_date"].isna(), "data_issue"] += "invalid_date;"
    df.loc[df["quantity"].isna() | (df["quantity"] <= 0), "data_issue"] += "invalid_quantity;"
    df.loc[df["unit_price"].isna() | (df["unit_price"] < 0), "data_issue"] += "invalid_price;"
    df["sales_amount"] = df["quantity"] * df["unit_price"]
    df["sales_amount"] = df["sales_amount"].where(df["data_issue"] == "", np.nan)
    return df

def save_orders(df):
    c = conn()
    if table_exists(c, "orders"):
        old = pd.read_sql_query("SELECT * FROM orders", c)
        combined = pd.concat([old, df], ignore_index=True)
    else:
        combined = df.copy()
    combined = combined.drop_duplicates(subset=["order_id"], keep="last")
    combined.to_sql("orders", c, if_exists="replace", index=False)
    c.close()
    return len(df), len(combined)

def load_orders():
    c = conn()
    if not table_exists(c, "orders"):
        c.close()
        return pd.DataFrame()
    df = pd.read_sql_query("SELECT * FROM orders", c)
    c.close()
    if not df.empty:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    return df

def load_reference_file(file, table):
    if file is None: return 0
    df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    df.columns = [str(c).strip().lower() for c in df.columns]
    c = conn()
    df.to_sql(table, c, if_exists="replace", index=False)
    c.close()
    return len(df)

def format_table_dates(df):
    """Format displayed timestamps to minute precision, without changing stored data."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M")
    return out


def add_product_names(df):
    """Attach product_name when the products reference table is available."""
    if df.empty or "product_id" not in df.columns:
        return df
    c = conn()
    try:
        if not table_exists(c, "products"):
            return df
        products = pd.read_sql_query("SELECT * FROM products", c)
    finally:
        c.close()
    products.columns = [str(col).strip().lower() for col in products.columns]
    if not {"product_id", "product_name"}.issubset(products.columns):
        return df
    products = products[["product_id", "product_name"]].drop_duplicates("product_id")
    out = df.copy()
    out["product_id"] = out["product_id"].astype(str)
    products["product_id"] = products["product_id"].astype(str)
    out = out.merge(products, on="product_id", how="left")
    out["product_name"] = out["product_name"].fillna("Name not available")
    return out

def daily_series(df):
    valid = df.dropna(subset=["order_date", "sales_amount"]).copy()
    if valid.empty: return pd.DataFrame(columns=["date", "sales"])
    result = valid.groupby(valid["order_date"].dt.floor("D"))["sales_amount"].sum().reset_index()
    result.columns = ["date", "sales"]
    result = result.sort_values("date")
    # Fill missing calendar dates with zero sales so lags are continuous.
    idx = pd.date_range(result["date"].min(), result["date"].max(), freq="D")
    result = result.set_index("date").reindex(idx, fill_value=0).rename_axis("date").reset_index()
    return result

def style_chart(fig, currency_axis=None):
    """Apply a consistent light dashboard theme to every Plotly chart."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Arial, sans-serif", color="#243047", size=13),
        title=dict(font=dict(size=18, color="#172033"), x=0.02, xanchor="left"),
        margin=dict(l=24, r=24, t=64, b=24),
        hoverlabel=dict(bgcolor="#172033", font_color="#ffffff"),
        legend=dict(bgcolor="rgba(255,255,255,0.75)", font=dict(color="#334155")),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#dbe2ee",
        tickfont=dict(color="#475569"),
        title_font=dict(color="#334155"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e8edf5",
        zerolinecolor="#dbe2ee",
        linecolor="#dbe2ee",
        tickfont=dict(color="#475569"),
        title_font=dict(color="#334155"),
    )
    if currency_axis == "y":
        fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
    elif currency_axis == "x":
        fig.update_xaxes(tickprefix="₹", tickformat=",.0f")
    return fig


def make_features(series):
    d = series.copy()
    d["lag_1"] = d["sales"].shift(1)
    d["lag_7"] = d["sales"].shift(7)
    d["rolling_7"] = d["sales"].shift(1).rolling(7).mean()
    d["day_of_week"] = d["date"].dt.dayofweek
    d["day_of_year"] = d["date"].dt.dayofyear
    d["month"] = d["date"].dt.month
    return d.dropna().reset_index(drop=True)

def train_model(series):
    feat = make_features(series)
    if len(feat) < 30:
        raise ValueError(f"Not enough training rows after lag features: {len(feat)}. Add more daily history.")
    cols = ["lag_1", "lag_7", "rolling_7", "day_of_week", "day_of_year", "month"]
    split = max(int(len(feat) * 0.8), 1)
    if len(feat) - split < 5:
        split = len(feat) - 5
    X_train, X_test = feat[cols].iloc[:split], feat[cols].iloc[split:]
    y_train, y_test = feat["sales"].iloc[:split], feat["sales"].iloc[split:]
    model = RandomForestRegressor(n_estimators=250, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
        "r2": float(r2_score(y_test, pred)) if len(y_test) >= 2 else None,
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
        "features": cols
    }
    joblib.dump({"model": model, "metrics": metrics, "trained_through": str(series["date"].max().date())}, MODEL_PATH)
    return metrics

def forecast_next_days(series, days=7):
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    history = series.copy().sort_values("date").reset_index(drop=True)
    outputs = []
    for _ in range(days):
        next_date = history["date"].max() + pd.Timedelta(days=1)
        sales_values = history["sales"].astype(float).tolist()
        row = pd.DataFrame([{
            "lag_1": sales_values[-1],
            "lag_7": sales_values[-7],
            "rolling_7": float(np.mean(sales_values[-7:])),
            "day_of_week": next_date.dayofweek,
            "day_of_year": next_date.dayofyear,
            "month": next_date.month
        }])
        value = max(0.0, float(model.predict(row)[0]))
        outputs.append({"date": next_date, "predicted_sales": value})
        history = pd.concat([history, pd.DataFrame([{"date": next_date, "sales": value}])], ignore_index=True)
    return pd.DataFrame(outputs)

st.markdown("""
<div class="hero">
  <h1>🛒 Shoplytics AI</h1>
  <p>One workspace for data quality, business insights and machine-learning sales forecasts.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose page", ["Overview", "Upload & Update Data", "Data Quality", "Sales Analytics", "ML Forecast"])
    st.divider()
    st.caption("Database: local SQLite")
    st.caption("Data stays on this computer.")

if page == "Upload & Update Data":
    st.header("Data Center")
    st.info("Upload an orders file. Each new daily file is merged into the existing database. Duplicate order_id values are updated rather than duplicated.")
    orders_file = st.file_uploader("Orders CSV or Excel (required)", type=["csv", "xlsx", "xls"], key="orders")
    c1, c2 = st.columns(2)
    with c1:
        customers_file = st.file_uploader("Customers CSV/Excel (optional)", type=["csv", "xlsx", "xls"])
        products_file = st.file_uploader("Products CSV/Excel (optional)", type=["csv", "xlsx", "xls"])
    with c2:
        payments_file = st.file_uploader("Payments CSV/Excel (optional)", type=["csv", "xlsx", "xls"])
        deliveries_file = st.file_uploader("Deliveries CSV/Excel (optional)", type=["csv", "xlsx", "xls"])
    if st.button("Import / Update Database", type="primary", disabled=orders_file is None):
        try:
            raw = pd.read_csv(orders_file) if orders_file.name.lower().endswith(".csv") else pd.read_excel(orders_file)
            cleaned = normalize_orders(raw)
            incoming, total = save_orders(cleaned)
            ref_counts = {}
            for f, t in [(customers_file, "customers"), (products_file, "products"), (payments_file, "payments"), (deliveries_file, "deliveries")]:
                if f is not None: ref_counts[t] = load_reference_file(f, t)
            st.success(f"Imported {incoming} incoming rows. Database now contains {total} unique orders.")
            if ref_counts: st.write("Reference tables imported:", ref_counts)
            st.write("Rows with data issues:", int((cleaned["data_issue"] != "").sum()))
        except Exception as e:
            st.error(f"Import failed: {e}")
    st.divider()
    st.subheader("Load the supplied sample dataset")
    st.write("This loads the five CSV files from the original uploaded project into the local database.")
    if st.button("Load original project sample data"):
        try:
            raw = pd.read_csv(os.path.join(SAMPLE_DIR, "orders.csv"))
            cleaned = normalize_orders(raw)
            incoming, total = save_orders(cleaned)
            for table in ["customers", "products", "payments", "deliveries"]:
                path = os.path.join(SAMPLE_DIR, table + ".csv")
                if os.path.exists(path):
                    pd.read_csv(path).to_sql(table, conn(), if_exists="replace", index=False)
            st.success(f"Sample orders loaded. {total} unique orders now in database.")
        except Exception as e:
            st.error(str(e))

elif page == "Overview":
    df = load_orders()
    st.header("Business Overview")
    if df.empty:
        st.warning("No data loaded yet. Go to 'Upload & Update Data' and load the included sample dataset or upload your own orders CSV.")
    else:
        valid = df.dropna(subset=["sales_amount", "order_date"])
        total_sales = valid["sales_amount"].sum()
        total_orders = valid["order_id"].nunique()
        customers = valid["customer_id"].nunique() if "customer_id" in valid else 0
        avg_order = total_sales / total_orders if total_orders else 0
        a,b,c,d = st.columns(4)
        a.metric("Gross sales", f"₹{total_sales:,.0f}")
        b.metric("Unique orders", f"{total_orders:,}")
        c.metric("Customers", f"{customers:,}")
        d.metric("Average order value", f"₹{avg_order:,.0f}")
        st.caption(f"Data range: {valid['order_date'].min().date()} to {valid['order_date'].max().date()}")
        st.markdown("""
        <div class="workspaces-title">Explore Your Workspaces</div>
        <div class="workspace-grid">
          <div class="workspace-card purple">
            <div class="icon">📥</div>
            <h3>Data Management</h3>
            <p>Upload CSV or Excel files and keep daily order records updated in one database.</p>
          </div>
          <div class="workspace-card blue">
            <div class="icon">🧹</div>
            <h3>Data Quality</h3>
            <p>Validate records, identify missing or invalid values and download quality reports.</p>
          </div>
          <div class="workspace-card teal">
            <div class="icon">📊</div>
            <h3>Sales Analytics</h3>
            <p>Explore sales trends, monthly performance and top-selling products.</p>
          </div>
          <div class="workspace-card orange">
            <div class="icon">🤖</div>
            <h3>AI Forecasting</h3>
            <p>Train a Random Forest model and forecast upcoming daily sales when enough history is available.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
        daily = daily_series(df)
        if not daily.empty:
            fig = px.line(daily, x="date", y="sales", title="Daily Sales Trend", markers=True)
            fig.update_traces(line=dict(color="#635bff", width=3), marker=dict(size=6, color="#16b8a6"))
            st.plotly_chart(style_chart(fig, currency_axis="y"), use_container_width=True)
        st.subheader("Latest imported orders")
        latest_orders = add_product_names(df.sort_values("order_date", ascending=False).head(10))
        st.dataframe(format_table_dates(latest_orders), use_container_width=True)

elif page == "Data Quality":
    df = load_orders()
    st.header("Data Health & Validation")
    if df.empty:
        st.warning("Load data first.")
    else:
        issue_rows = df[df["data_issue"].fillna("") != ""]
        a,b,c = st.columns(3)
        a.metric("Total rows", f"{len(df):,}")
        b.metric("Rows with issues", f"{len(issue_rows):,}")
        c.metric("Duplicate order IDs", f"{int(df['order_id'].duplicated().sum()):,}")
        if not issue_rows.empty:
            st.dataframe(format_table_dates(issue_rows), use_container_width=True)
        else:
            st.success("No flagged issues in the imported order records.")
        st.download_button("Download current orders with quality flags", df.to_csv(index=False).encode("utf-8"), "orders_quality_report.csv", "text/csv")

elif page == "Sales Analytics":
    df = load_orders()
    st.header("Sales Analytics")
    if df.empty:
        st.warning("Load data first.")
    else:
        valid = df.dropna(subset=["sales_amount", "order_date"]).copy()
        if valid.empty:
            st.warning("No valid sales rows available.")
        else:
            valid["month"] = valid["order_date"].dt.to_period("M").astype(str)
            monthly = valid.groupby("month", as_index=False)["sales_amount"].sum()
            fig = px.bar(monthly, x="month", y="sales_amount", title="Monthly Sales", text_auto=".2s")
            fig.update_traces(marker_color="#635bff", marker_line_width=0)
            st.plotly_chart(style_chart(fig, currency_axis="y"), use_container_width=True)
            if "product_id" in valid.columns:
                product_sales = valid.groupby("product_id", as_index=False)["sales_amount"].sum()
                product_sales = add_product_names(product_sales)
                product_sales["product_label"] = product_sales.apply(
                    lambda row: f"{row['product_name']} ({row['product_id']})"
                    if row.get("product_name") and row["product_name"] != "Name not available"
                    else str(row["product_id"]), axis=1
                )
                top = product_sales.nlargest(10, "sales_amount").sort_values("sales_amount")
                fig = px.bar(top, x="sales_amount", y="product_label", orientation="h",
                             title="Top 10 Products by Sales", text_auto=".2s")
                fig.update_traces(marker_color="#16b8a6", marker_line_width=0)
                st.plotly_chart(style_chart(fig, currency_axis="x"), use_container_width=True)
                st.subheader("Product sales details")
                st.dataframe(
                    format_table_dates(product_sales.sort_values("sales_amount", ascending=False).head(20))
                    [["product_id", "product_name", "sales_amount"]]
                    if "product_name" in product_sales.columns
                    else product_sales.sort_values("sales_amount", ascending=False).head(20),
                    use_container_width=True
                )
            st.dataframe(format_table_dates(monthly), use_container_width=True)

elif page == "ML Forecast":
    df = load_orders()
    st.header("AI Sales Forecast")
    if df.empty:
        st.warning("Load data first.")
    else:
        series = daily_series(df)
        st.write(f"Continuous daily history available: **{len(series)} days**")
        st.info("The model uses past daily sales (previous day, previous 7 days, 7-day average and calendar features). This is a time-series forecasting baseline, not a guarantee of future sales.")
        if len(series) < 60:
            st.warning("At least 60 calendar days of history are required by this MVP before training. Your supplied sample currently has only one month, so the app will collect more daily data first.")
        if st.button("Train / Retrain Random Forest model", type="primary", disabled=len(series) < 60):
            try:
                metrics = train_model(series)
                st.success("Model trained and saved.")
                st.json(metrics)
            except Exception as e:
                st.error(f"Training failed: {e}")
        if os.path.exists(MODEL_PATH):
            bundle = joblib.load(MODEL_PATH)
            st.subheader("Last model evaluation")
            m = bundle["metrics"]
            x,y,z = st.columns(3)
            x.metric("MAE", f"₹{m['mae']:,.2f}")
            y.metric("RMSE", f"₹{m['rmse']:,.2f}")
            z.metric("Test days", str(m["test_rows"]))
            st.caption("Model trained through " + bundle["trained_through"])
            days = st.slider("Forecast horizon (days)", 1, 14, 7)
            if st.button("Generate forecast"):
                fc = forecast_next_days(series, days)
                fig = px.line(fc, x="date", y="predicted_sales", markers=True, title="Forecasted Daily Sales")
                fig.update_traces(line=dict(color="#16b8a6", width=3), marker=dict(size=8, color="#635bff"))
                st.plotly_chart(style_chart(fig, currency_axis="y"), use_container_width=True)
                st.dataframe(fc, use_container_width=True)
                st.download_button("Download forecast CSV", fc.to_csv(index=False).encode("utf-8"), "sales_forecast.csv", "text/csv")
        else:
            st.caption("No trained model found yet. Collect sufficient history, then train it here.")

st.sidebar.divider()
st.sidebar.caption("MVP note: live store API connectors are the next integration phase. Use data from stores you own or are authorized to access.")
