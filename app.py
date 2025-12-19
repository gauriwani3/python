import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from datetime import datetime
from contextlib import contextmanager
from typing import Optional
import warnings
import logging

# -----------------------------
# Config & Setup
# -----------------------------
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Database Configuration
# -----------------------------

PG_HOST = os.getenv("PG_HOST")
PG_PORT = int(os.getenv("PG_PORT"))
PG_DB   = os.getenv("PG_DB")
PG_USER = os.getenv("PG_USER")
PG_PASS = os.getenv("PG_PASS")

PRODUCTION_PLAN_TABLE = "hil_tlj_cast_house.production_plan_batches"

# -----------------------------
# DB Context Manager
# -----------------------------
@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASS
        )
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()

# -----------------------------
# Fetch Production Data
# -----------------------------
def fetch_production_data(limit: int = 2000) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM {PRODUCTION_PLAN_TABLE}
        LIMIT %s
    """
    try:
        with get_db_connection() as conn:
            return pd.read_sql(query, conn, params=(limit,))
    except Exception as e:
        logger.error(f"Error fetching production data: {e}")
        return pd.DataFrame()

# =====================================================
# Streamlit App
# =====================================================
st.set_page_config(page_title="Production Dashboard", layout="wide")
st.title("📊 Production Plan Dashboard")

# -----------------------------
# Load Data
# -----------------------------
df = fetch_production_data()

if df.empty:
    st.warning("No data available in production_plan_batches table.")
    st.stop()

# -----------------------------
# Show Columns
# -----------------------------
st.subheader("📌 Available Columns")
st.write(df.columns.tolist())

# -----------------------------
# Select Datetime Column
# -----------------------------
datetime_col = st.selectbox(
    "Select datetime column for analysis",
    options=df.columns.tolist()
)

# Convert selected column to datetime
df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
df = df.dropna(subset=[datetime_col])

if df.empty:
    st.error("Selected column does not contain valid datetime values.")
    st.stop()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔍 Filters")

min_date = df[datetime_col].min().date()
max_date = df[datetime_col].max().date()

start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

filtered_df = df[
    (df[datetime_col].dt.date >= start_date) &
    (df[datetime_col].dt.date <= end_date)
]

# -----------------------------
# KPI Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(filtered_df))
col2.metric("Start Date", start_date)
col3.metric("End Date", end_date)

# -----------------------------
# Aggregations
# -----------------------------
daily_batches = (
    filtered_df
    .groupby(filtered_df[datetime_col].dt.date)
    .size()
    .reset_index(name="batch_count")
)

status_df = pd.DataFrame()
if "status" in filtered_df.columns:
    status_df = (
        filtered_df.groupby("status")
        .size()
        .reset_index(name="count")
    )

# -----------------------------
# Plotly Charts
# -----------------------------
st.subheader("📈 Daily Production Trend")
fig_line = px.line(
    daily_batches,
    x=datetime_col,
    y="batch_count",
    markers=True,
    title="Daily Production Batches"
)
st.plotly_chart(fig_line, use_container_width=True)

if not status_df.empty:
    st.subheader("📊 Batch Status Distribution")
    fig_bar = px.bar(
        status_df,
        x="status",
        y="count",
        text="count",
        title="Batch Status Count"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# Raw Data Preview
# -----------------------------
with st.expander("📄 View Filtered Data"):
    st.dataframe(filtered_df)
