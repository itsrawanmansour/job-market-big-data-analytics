from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
ML_DIR = BASE_DIR / "data" / "output" / "ml"

CLEANED_POSTINGS_PATH = PROCESSED_DIR / "cleaned_postings.parquet"
JOBS_WITH_SKILLS_PATH = PROCESSED_DIR / "jobs_with_skills.parquet"
JOB_SKILL_SUMMARY_PATH = PROCESSED_DIR / "job_skill_summary.parquet"

ML_SUMMARY_PATH = ML_DIR / "ml_summary.txt"
CONFUSION_MATRIX_PATH = ML_DIR / "confusion_matrix.csv"
SAMPLE_PREDICTIONS_PATH = ML_DIR / "sample_predictions.csv"

SALARY_MIN = 10000
SALARY_MAX = 500000

PALETTE = [
    "#2563EB",
    "#14B8A6",
    "#F59E0B",
    "#EF4444",
    "#8B5CF6",
    "#22C55E",
    "#EC4899",
    "#06B6D4",
    "#64748B",
    "#84CC16",
]

CONTINUOUS_SCALE = [
    "#EFF6FF",
    "#BFDBFE",
    "#60A5FA",
    "#2563EB",
    "#1E3A8A",
]

st.set_page_config(
    page_title="Job Market Big Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>
    :root {
        --main-bg: #0B1220;
        --panel-bg: #111827;
        --card-bg: #FFFFFF;
        --soft-bg: #F8FAFC;
        --text-main: #0F172A;
        --text-muted: #64748B;
        --text-white: #FFFFFF;
        --border: #E2E8F0;
        --blue: #2563EB;
        --teal: #14B8A6;
    }

    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", Arial, sans-serif !important;
    }

    .stApp {
        background: linear-gradient(180deg, #0B1220 0%, #111827 45%, #0F172A 100%) !important;
        color: var(--text-white) !important;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }

    p, span, div {
        color: inherit;
    }

    .hero {
        padding: 28px 32px;
        border-radius: 26px;
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 45%, #14B8A6 100%);
        color: #FFFFFF !important;
        margin-bottom: 20px;
        box-shadow: 0 22px 55px rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.16);
    }

    .hero-title {
        font-size: 36px;
        font-weight: 850;
        margin-bottom: 8px;
        letter-spacing: -0.7px;
        color: #FFFFFF !important;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #E0F2FE !important;
        max-width: 1050px;
        line-height: 1.65;
    }

    .section-title {
        font-size: 25px;
        font-weight: 850;
        color: #FFFFFF !important;
        margin-top: 12px;
        margin-bottom: 6px;
        letter-spacing: -0.3px;
    }

    .section-note {
        color: #CBD5E1 !important;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .metric-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 18px 18px;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.18);
        min-height: 112px;
    }

    .metric-label {
        color: #475569 !important;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #0F172A !important;
        font-size: 29px;
        font-weight: 850;
        letter-spacing: -0.5px;
    }

    .metric-foot {
        color: #64748B !important;
        font-size: 12px;
        margin-top: 8px;
    }

    .insight-box {
        padding: 18px 20px;
        border-radius: 18px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        margin: 8px 0 14px 0;
        color: #0F172A !important;
        line-height: 1.6;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.14);
    }

    .recommendation {
        padding: 18px 20px;
        border-radius: 18px;
        background: linear-gradient(135deg, #EFF6FF 0%, #ECFDF5 100%);
        border: 1px solid #BFDBFE;
        margin-bottom: 14px;
        color: #0F172A !important;
        line-height: 1.6;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.14);
    }

    .warning-box {
        padding: 16px 18px;
        border-radius: 16px;
        background: #FFF7ED;
        border: 1px solid #FED7AA;
        color: #9A3412 !important;
        margin-bottom: 16px;
    }

    .insight-box b,
    .recommendation b,
    .warning-box b {
        color: #0F172A !important;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border-radius: 16px;
        padding: 12px;
        border: 1px solid #E2E8F0;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricDelta"] {
        color: #0F172A !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        border-radius: 14px !important;
        padding: 10px 16px !important;
        border: 1px solid #CBD5E1 !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab"] p {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #DBEAFE 0%, #CCFBF1 100%) !important;
        color: #0F172A !important;
        border: 1px solid #60A5FA !important;
    }

    .stTabs [aria-selected="true"] p {
        color: #0F172A !important;
    }

    section[data-testid="stSidebar"] {
        background: #F8FAFC !important;
        border-right: 1px solid #CBD5E1 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #0F172A !important;
    }

    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #475569 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
    }

    div[data-baseweb="select"] span {
        color: #0F172A !important;
    }

    div[data-baseweb="popover"] * {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
    }

    .stSlider label,
    .stSelectbox label,
    .stMultiSelect label,
    .stNumberInput label {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    div[data-testid="stDataFrame"] {
        background: #FFFFFF !important;
        border-radius: 14px !important;
    }

    .stDataFrame,
    .stTable {
        color: #0F172A !important;
    }

    .stAlert {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border-radius: 14px !important;
    }

    .stAlert * {
        color: #0F172A !important;
    }

    div[data-testid="stMarkdownContainer"] {
        color: inherit;
    }

    div[data-testid="stMarkdownContainer"] p {
        color: inherit;
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #14B8A6 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
    }

    button[kind="primary"] p {
        color: #FFFFFF !important;
    }

    .plot-container, .js-plotly-plot {
        border-radius: 18px !important;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data(show_spinner=False)
def load_processed_data():
    cleaned_postings = pd.read_parquet(CLEANED_POSTINGS_PATH)
    jobs_with_skills = pd.read_parquet(JOBS_WITH_SKILLS_PATH)

    if JOB_SKILL_SUMMARY_PATH.exists():
        job_skill_summary = pd.read_parquet(JOB_SKILL_SUMMARY_PATH)
    else:
        job_skill_summary = pd.DataFrame()

    cleaned_postings["salary_value"] = pd.to_numeric(cleaned_postings["salary_value"], errors="coerce")
    cleaned_postings["views"] = pd.to_numeric(cleaned_postings["views"], errors="coerce").fillna(0)
    cleaned_postings["applies"] = pd.to_numeric(cleaned_postings["applies"], errors="coerce").fillna(0)

    jobs_with_skills["salary_value"] = pd.to_numeric(jobs_with_skills["salary_value"], errors="coerce")
    jobs_with_skills["views"] = pd.to_numeric(jobs_with_skills["views"], errors="coerce").fillna(0)
    jobs_with_skills["applies"] = pd.to_numeric(jobs_with_skills["applies"], errors="coerce").fillna(0)

    return cleaned_postings, jobs_with_skills, job_skill_summary


@st.cache_data(show_spinner=False)
def load_ml_outputs():
    summary_text = ""
    confusion_matrix = pd.DataFrame()
    sample_predictions = pd.DataFrame()

    if ML_SUMMARY_PATH.exists():
        summary_text = ML_SUMMARY_PATH.read_text(encoding="utf-8")

    if CONFUSION_MATRIX_PATH.exists():
        confusion_matrix = pd.read_csv(CONFUSION_MATRIX_PATH)

    if SAMPLE_PREDICTIONS_PATH.exists():
        sample_predictions = pd.read_csv(SAMPLE_PREDICTIONS_PATH)

    return summary_text, confusion_matrix, sample_predictions


def parse_metric(summary_text, metric_name):
    match = re.search(rf"{metric_name}:\s*([0-9.]+)", summary_text)
    if match:
        return float(match.group(1))
    return None


def valid_salary_data(df):
    return df[
        (df["salary_value"].notna()) &
        (df["salary_value"] >= SALARY_MIN) &
        (df["salary_value"] <= SALARY_MAX)
    ].copy()


def format_number(value):
    if pd.isna(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def metric_card(label, value, footnote=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def styled_plotly(fig, height=430):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=65, b=30),
        font=dict(family="Inter, Segoe UI, Arial", size=13, color="#0F172A"),
        title=dict(font=dict(size=18, color="#0F172A")),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter", font_color="#0F172A"),
        legend=dict(
            font=dict(color="#0F172A"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(color="#0F172A", title_font=dict(color="#0F172A"), tickfont=dict(color="#0F172A")),
        yaxis=dict(color="#0F172A", title_font=dict(color="#0F172A"), tickfont=dict(color="#0F172A"))
    )
    return fig


def plot_vertical_bar(df, x, y, title, text_col=None, height=430):
    if df.empty:
        st.info("No data available for this chart.")
        return

    fig = px.bar(
        df,
        x=x,
        y=y,
        color=x,
        color_discrete_sequence=PALETTE,
        text=text_col or y,
        title=title,
        hover_data=df.columns
    )
    fig.update_traces(textposition="outside", marker_line_width=0, textfont_color="#0F172A")
    fig.update_xaxes(tickangle=35)
    fig = styled_plotly(fig, height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


def plot_horizontal_bar(df, x, y, title, height=480):
    if df.empty:
        st.info("No data available for this chart.")
        return

    data = df.copy().iloc[::-1]

    fig = px.bar(
        data,
        x=x,
        y=y,
        orientation="h",
        color=x,
        color_continuous_scale=CONTINUOUS_SCALE,
        text=x,
        title=title,
        hover_data=data.columns
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", textfont_color="#0F172A")
    fig = styled_plotly(fig, height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


def plot_donut(df, names, values, title, height=420):
    if df.empty:
        st.info("No data available for this chart.")
        return

    fig = px.pie(
        df,
        names=names,
        values=values,
        hole=0.55,
        color_discrete_sequence=PALETTE,
        title=title
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(color="#FFFFFF", size=13),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>"
    )
    fig = styled_plotly(fig, height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


def plot_treemap(df, path_col, values_col, title, height=520):
    if df.empty:
        st.info("No data available for this chart.")
        return

    fig = px.treemap(
        df,
        path=[path_col],
        values=values_col,
        color=values_col,
        color_continuous_scale=CONTINUOUS_SCALE,
        title=title
    )
    fig.update_traces(
        textinfo="label+value",
        textfont=dict(color="#FFFFFF", size=15),
        marker=dict(line=dict(color="#FFFFFF", width=1)),
        hovertemplate="<b>%{label}</b><br>Value: %{value:,}<extra></extra>"
    )
    fig = styled_plotly(fig, height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


def plot_bubble(df, x, y, size, color, title, height=520):
    if df.empty:
        st.info("No data available for this chart.")
        return

    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size,
        color=color,
        color_discrete_sequence=PALETTE,
        hover_data=df.columns,
        title=title,
        size_max=45
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="#FFFFFF")))
    fig = styled_plotly(fig, height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


def plot_heatmap(matrix, title, height=500):
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale=CONTINUOUS_SCALE,
        title=title,
        labels=dict(x="Predicted Salary Level", y="Actual Salary Level", color="Count")
    )
    fig.update_traces(textfont=dict(color="#FFFFFF"))
    fig = styled_plotly(fig, height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


def create_filters(cleaned_postings, jobs_with_skills):
    st.sidebar.markdown("## Filters")
    st.sidebar.caption("Use the filters to update the whole dashboard.")

    work_types = ["All"] + sorted(cleaned_postings["work_type_final"].dropna().unique().tolist())
    remote_statuses = ["All"] + sorted(cleaned_postings["remote_status"].dropna().unique().tolist())
    skills = ["All"] + sorted(jobs_with_skills["skill_name"].dropna().unique().tolist())

    selected_work_type = st.sidebar.selectbox("Work Type", work_types)
    selected_remote = st.sidebar.selectbox("Remote Status", remote_statuses)
    selected_skill = st.sidebar.selectbox("Skill", skills)

    salary_range = st.sidebar.slider(
        "Salary Range",
        min_value=SALARY_MIN,
        max_value=SALARY_MAX,
        value=(SALARY_MIN, SALARY_MAX),
        step=5000
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("The salary range filter affects salary-related visuals only.")

    return selected_work_type, selected_remote, selected_skill, salary_range


def apply_filters(cleaned_postings, jobs_with_skills, selected_work_type, selected_remote, selected_skill):
    filtered_jobs = cleaned_postings.copy()

    if selected_work_type != "All":
        filtered_jobs = filtered_jobs[filtered_jobs["work_type_final"] == selected_work_type]

    if selected_remote != "All":
        filtered_jobs = filtered_jobs[filtered_jobs["remote_status"] == selected_remote]

    if selected_skill != "All":
        skill_job_ids = jobs_with_skills[jobs_with_skills["skill_name"] == selected_skill]["job_id"].unique()
        filtered_jobs = filtered_jobs[filtered_jobs["job_id"].isin(skill_job_ids)]

    job_ids = filtered_jobs["job_id"].unique()
    filtered_skills = jobs_with_skills[jobs_with_skills["job_id"].isin(job_ids)]

    return filtered_jobs, filtered_skills


def overview_tab(filtered_jobs, filtered_skills, salary_range):
    st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">A quick summary of the filtered job market dataset.</div>', unsafe_allow_html=True)

    salary_jobs = valid_salary_data(filtered_jobs)
    salary_jobs = salary_jobs[
        (salary_jobs["salary_value"] >= salary_range[0]) &
        (salary_jobs["salary_value"] <= salary_range[1])
    ]

    total_jobs = filtered_jobs["job_id"].nunique()
    total_companies = filtered_jobs[filtered_jobs["company_name"] != "Unknown Company"]["company_name"].nunique()
    total_locations = filtered_jobs["location"].nunique()
    total_skills = filtered_skills["skill_name"].nunique()
    average_salary = salary_jobs["salary_value"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card("Total Job Postings", f"{total_jobs:,}", "Filtered records")
    with c2:
        metric_card("Total Companies", f"{total_companies:,}", "Unique companies")
    with c3:
        metric_card("Total Locations", f"{total_locations:,}", "Unique locations")
    with c4:
        metric_card("Total Skills", f"{total_skills:,}", "Mapped skill groups")
    with c5:
        metric_card("Average Salary", format_number(average_salary), "Valid salary records")

    work_types = (
        filtered_jobs
        .groupby("work_type_final")
        .size()
        .reset_index(name="jobs_count")
        .sort_values("jobs_count", ascending=False)
    )

    salary_levels = (
        filtered_jobs
        .groupby("salary_level")
        .size()
        .reset_index(name="jobs_count")
        .sort_values("jobs_count", ascending=False)
    )

    top_skills = (
        filtered_skills
        .groupby("skill_name")
        .agg(jobs_count=("job_id", "nunique"))
        .reset_index()
        .sort_values("jobs_count", ascending=False)
        .head(10)
    )

    c6, c7 = st.columns([1, 1])

    with c6:
        plot_donut(work_types, "work_type_final", "jobs_count", "Work Type Share")

    with c7:
        plot_donut(salary_levels, "salary_level", "jobs_count", "Salary Level Share")

    plot_treemap(top_skills, "skill_name", "jobs_count", "Top Skills Treemap", height=480)


def job_trends_tab(filtered_jobs):
    st.markdown('<div class="section-title">Job Market Trends</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Explore the most frequent jobs, companies, locations, and work types.</div>', unsafe_allow_html=True)

    top_titles = (
        filtered_jobs
        .groupby("title")
        .size()
        .reset_index(name="jobs_count")
        .sort_values("jobs_count", ascending=False)
        .head(10)
    )

    top_companies = (
        filtered_jobs[filtered_jobs["company_name"] != "Unknown Company"]
        .groupby("company_name")
        .size()
        .reset_index(name="jobs_count")
        .sort_values("jobs_count", ascending=False)
        .head(10)
    )

    top_locations = (
        filtered_jobs
        .groupby("location")
        .size()
        .reset_index(name="jobs_count")
        .sort_values("jobs_count", ascending=False)
        .head(10)
    )

    work_types = (
        filtered_jobs
        .groupby("work_type_final")
        .size()
        .reset_index(name="jobs_count")
        .sort_values("jobs_count", ascending=False)
    )

    c1, c2 = st.columns([1.15, 1])

    with c1:
        plot_horizontal_bar(top_titles, "jobs_count", "title", "Top 10 Job Titles", height=470)

    with c2:
        plot_treemap(top_companies, "company_name", "jobs_count", "Top Companies Treemap", height=470)

    c3, c4 = st.columns([1.1, 1])

    with c3:
        plot_bubble(top_locations, "jobs_count", "location", "jobs_count", "location", "Top Locations Bubble Chart", height=470)

    with c4:
        plot_donut(work_types, "work_type_final", "jobs_count", "Jobs by Work Type", height=470)


def skills_tab(filtered_skills):
    st.markdown('<div class="section-title">Skills Demand</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Analyze the most requested skills and their relationship with salary and remote jobs.</div>', unsafe_allow_html=True)

    top_skills = (
        filtered_skills
        .groupby("skill_name")
        .agg(jobs_count=("job_id", "nunique"))
        .reset_index()
        .sort_values("jobs_count", ascending=False)
        .head(20)
    )

    high_salary_skills = (
        valid_salary_data(filtered_skills)
        .query("salary_level == 'High'")
        .groupby("skill_name")
        .agg(
            jobs_count=("job_id", "nunique"),
            avg_salary=("salary_value", "mean")
        )
        .reset_index()
        .sort_values("jobs_count", ascending=False)
        .head(12)
    )

    remote_skills = (
        filtered_skills[filtered_skills["remote_status"] == "Remote"]
        .groupby("skill_name")
        .agg(jobs_count=("job_id", "nunique"))
        .reset_index()
        .sort_values("jobs_count", ascending=False)
        .head(12)
    )

    plot_horizontal_bar(top_skills, "jobs_count", "skill_name", "Top 20 In-Demand Skills", height=650)

    c1, c2 = st.columns(2)

    with c1:
        plot_bubble(
            high_salary_skills,
            "avg_salary",
            "jobs_count",
            "jobs_count",
            "skill_name",
            "High-Salary Skills: Salary vs Demand",
            height=470
        )

    with c2:
        plot_treemap(remote_skills, "skill_name", "jobs_count", "Remote Job Skills Treemap", height=470)

    st.markdown("### Skills by Selected Job Title")

    job_titles = ["Select a job title"] + sorted(filtered_skills["title"].dropna().unique().tolist())
    selected_title = st.selectbox("Choose Job Title", job_titles)

    if selected_title != "Select a job title":
        title_skills = (
            filtered_skills[filtered_skills["title"] == selected_title]
            .groupby("skill_name")
            .agg(jobs_count=("job_id", "nunique"))
            .reset_index()
            .sort_values("jobs_count", ascending=False)
            .head(15)
        )

        plot_horizontal_bar(title_skills, "jobs_count", "skill_name", f"Top Skills for {selected_title}", height=520)
    else:
        st.info("Select a job title to view its related skills.")


def salary_tab(filtered_jobs, filtered_skills, salary_range):
    st.markdown('<div class="section-title">Salary Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Salary visuals use valid salary records only and apply the selected salary range.</div>', unsafe_allow_html=True)

    salary_jobs = valid_salary_data(filtered_jobs)
    salary_skills = valid_salary_data(filtered_skills)

    salary_jobs = salary_jobs[
        (salary_jobs["salary_value"] >= salary_range[0]) &
        (salary_jobs["salary_value"] <= salary_range[1])
    ]

    salary_skills = salary_skills[
        (salary_skills["salary_value"] >= salary_range[0]) &
        (salary_skills["salary_value"] <= salary_range[1])
    ]

    salary_by_title = (
        salary_jobs
        .groupby("title")
        .agg(
            jobs_count=("job_id", "nunique"),
            avg_salary=("salary_value", "mean"),
            median_salary=("salary_value", "median")
        )
        .reset_index()
    )
    salary_by_title = salary_by_title[salary_by_title["jobs_count"] >= 5]
    salary_by_title = salary_by_title.sort_values("avg_salary", ascending=False).head(12)

    salary_by_skill = (
        salary_skills
        .groupby("skill_name")
        .agg(
            jobs_count=("job_id", "nunique"),
            avg_salary=("salary_value", "mean"),
            median_salary=("salary_value", "median")
        )
        .reset_index()
    )
    salary_by_skill = salary_by_skill[salary_by_skill["jobs_count"] >= 5]
    salary_by_skill = salary_by_skill.sort_values("avg_salary", ascending=False).head(12)

    salary_by_location = (
        salary_jobs
        .groupby("location")
        .agg(
            jobs_count=("job_id", "nunique"),
            avg_salary=("salary_value", "mean"),
            median_salary=("salary_value", "median")
        )
        .reset_index()
    )
    salary_by_location = salary_by_location[salary_by_location["jobs_count"] >= 10]
    salary_by_location = salary_by_location.sort_values("avg_salary", ascending=False).head(12)

    salary_level = (
        filtered_jobs
        .groupby("salary_level")
        .size()
        .reset_index(name="jobs_count")
        .sort_values("jobs_count", ascending=False)
    )

    remote_salary = (
        salary_jobs
        .groupby("remote_status")
        .agg(
            jobs_count=("job_id", "nunique"),
            avg_salary=("salary_value", "mean"),
            median_salary=("salary_value", "median")
        )
        .reset_index()
        .sort_values("avg_salary", ascending=False)
    )

    c1, c2 = st.columns([1.1, 1])

    with c1:
        plot_horizontal_bar(salary_by_title, "avg_salary", "title", "Highest Average Salary by Job Title", height=520)

    with c2:
        plot_donut(salary_level, "salary_level", "jobs_count", "Salary Level Distribution", height=520)

    c3, c4 = st.columns(2)

    with c3:
        plot_bubble(
            salary_by_skill,
            "avg_salary",
            "jobs_count",
            "jobs_count",
            "skill_name",
            "Salary by Skill: Average Salary vs Demand",
            height=500
        )

    with c4:
        plot_bubble(
            salary_by_location,
            "avg_salary",
            "jobs_count",
            "jobs_count",
            "location",
            "Salary by Location: Average Salary vs Job Count",
            height=500
        )

    plot_vertical_bar(remote_salary, "remote_status", "avg_salary", "Remote vs Non-Remote Average Salary", height=420)


def engagement_tab(filtered_jobs, filtered_skills):
    st.markdown('<div class="section-title">Engagement Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">This section studies job visibility through views and applications.</div>', unsafe_allow_html=True)

    total_views = filtered_jobs["views"].sum()
    total_applies = filtered_jobs["applies"].sum()
    avg_views = filtered_jobs["views"].mean()
    avg_applies = filtered_jobs["applies"].mean()
    correlation = filtered_jobs[["views", "applies"]].corr().iloc[0, 1] if len(filtered_jobs) > 1 else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card("Total Views", format_number(total_views), "All filtered jobs")
    with c2:
        metric_card("Total Applies", format_number(total_applies), "All filtered jobs")
    with c3:
        metric_card("Avg Views", f"{avg_views:.2f}", "Per job")
    with c4:
        metric_card("Avg Applies", f"{avg_applies:.2f}", "Per job")
    with c5:
        metric_card("Correlation", f"{correlation:.4f}" if pd.notna(correlation) else "N/A", "Views vs applies")

    scatter_data = filtered_jobs.copy()
    scatter_data = scatter_data[(scatter_data["views"] >= 0) & (scatter_data["applies"] >= 0)]

    if not scatter_data.empty:
        sample_data = scatter_data.sample(min(len(scatter_data), 6000), random_state=42)

        fig = px.scatter(
            sample_data,
            x="views",
            y="applies",
            color="remote_status",
            size=np.maximum(sample_data["views"], 1),
            color_discrete_sequence=PALETTE,
            hover_data=["title", "company_name", "location", "salary_level", "work_type_final"],
            title="Views vs Applications"
        )
        fig.update_traces(marker=dict(line=dict(width=1, color="#FFFFFF")))
        fig = styled_plotly(fig, 560)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    engagement_by_skill = (
        filtered_skills
        .groupby("skill_name")
        .agg(
            jobs_count=("job_id", "nunique"),
            avg_views=("views", "mean"),
            avg_applies=("applies", "mean")
        )
        .reset_index()
        .sort_values("avg_views", ascending=False)
        .head(15)
    )

    plot_bubble(
        engagement_by_skill,
        "avg_views",
        "avg_applies",
        "jobs_count",
        "skill_name",
        "Skills Engagement: Average Views vs Average Applications",
        height=520
    )

    c6, c7 = st.columns(2)

    with c6:
        st.markdown("### Top Viewed Jobs")
        top_viewed = (
            filtered_jobs
            .sort_values("views", ascending=False)
            [["job_id", "title", "company_name", "location", "views", "applies", "remote_status", "salary_level"]]
            .head(20)
        )
        st.dataframe(top_viewed, use_container_width=True, hide_index=True)

    with c7:
        st.markdown("### Top Applied Jobs")
        top_applied = (
            filtered_jobs
            .sort_values("applies", ascending=False)
            [["job_id", "title", "company_name", "location", "views", "applies", "remote_status", "salary_level"]]
            .head(20)
        )
        st.dataframe(top_applied, use_container_width=True, hide_index=True)


@st.cache_resource(show_spinner=False)
def train_dashboard_prediction_model(cleaned_postings, jobs_with_skills):
    skills_text = (
        jobs_with_skills
        .dropna(subset=["skill_name"])
        .groupby("job_id")["skill_name"]
        .apply(lambda values: " ".join(sorted(set([str(v).replace(" ", "_") for v in values]))))
        .reset_index()
        .rename(columns={"skill_name": "skills_text"})
    )

    data = cleaned_postings.merge(skills_text, on="job_id", how="left")

    data["skills_text"] = data["skills_text"].fillna("Unknown_Skill")
    data["salary_value"] = pd.to_numeric(data["salary_value"], errors="coerce")
    data["views"] = pd.to_numeric(data["views"], errors="coerce").fillna(0)
    data["applies"] = pd.to_numeric(data["applies"], errors="coerce").fillna(0)

    data = data[
        (data["salary_value"].notna()) &
        (data["salary_value"] >= SALARY_MIN) &
        (data["salary_value"] <= SALARY_MAX) &
        (data["salary_level"].isin(["Low", "Medium", "High"]))
    ].copy()

    top_titles = data["title"].value_counts().head(50).index.tolist()
    top_locations = data["location"].value_counts().head(50).index.tolist()

    data["title_group"] = data["title"].apply(lambda value: value if value in top_titles else "Other Title")
    data["location_group"] = data["location"].apply(lambda value: value if value in top_locations else "Other Location")

    for column in ["work_type_final", "remote_status", "experience_level_final"]:
        data[column] = data[column].fillna("Unknown").astype(str)

    features = [
        "title_group",
        "location_group",
        "work_type_final",
        "remote_status",
        "views",
        "applies",
        "experience_level_final",
        "skills_text"
    ]

    X = data[features]
    y = data["salary_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    categorical_features = [
        "title_group",
        "location_group",
        "work_type_final",
        "remote_status",
        "experience_level_final"
    ]

    numeric_features = ["views", "applies"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
            ("skills", CountVectorizer(), "skills_text")
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=140,
                max_depth=13,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            ))
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    metadata = {
        "accuracy": accuracy,
        "f1": f1,
        "title_options": sorted(top_titles) + ["Other Title"],
        "location_options": sorted(top_locations) + ["Other Location"],
        "work_type_options": sorted(data["work_type_final"].dropna().unique().tolist()),
        "remote_options": sorted(data["remote_status"].dropna().unique().tolist()),
        "experience_options": sorted(data["experience_level_final"].dropna().unique().tolist()),
        "skill_options": sorted(jobs_with_skills["skill_name"].dropna().unique().tolist()),
        "classes": model.named_steps["classifier"].classes_.tolist()
    }

    return model, metadata


def ml_tab(cleaned_postings, jobs_with_skills):
    st.markdown('<div class="section-title">Machine Learning Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Review the official Spark MLlib results and test an interactive salary level prediction.</div>', unsafe_allow_html=True)

    summary_text, confusion_matrix, sample_predictions = load_ml_outputs()

    official_accuracy = parse_metric(summary_text, "Accuracy")
    official_f1 = parse_metric(summary_text, "F1-score")

    c1, c2 = st.columns(2)

    with c1:
        metric_card("Spark ML Accuracy", f"{official_accuracy:.4f}" if official_accuracy is not None else "N/A", "Official Spark MLlib model")
    with c2:
        metric_card("Spark ML F1-score", f"{official_f1:.4f}" if official_f1 is not None else "N/A", "Weighted classification score")

    if not confusion_matrix.empty:
        st.markdown("### Confusion Matrix")

        order = ["Low", "Medium", "High"]
        matrix = (
            confusion_matrix
            .pivot(index="actual_salary_level", columns="predicted_salary_level", values="count")
            .reindex(index=order, columns=order)
            .fillna(0)
        )

        plot_heatmap(matrix, "Salary Level Prediction Confusion Matrix", height=520)

    if not sample_predictions.empty:
        st.markdown("### Sample Predictions")
        st.dataframe(sample_predictions.head(30), use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("### Interactive Salary Level Prediction")

    model, metadata = train_dashboard_prediction_model(cleaned_postings, jobs_with_skills)

    c3, c4 = st.columns(2)

    with c3:
        title_group = st.selectbox("Job Title", metadata["title_options"])
        location_group = st.selectbox("Location", metadata["location_options"])
        work_type = st.selectbox("Work Type", metadata["work_type_options"])
        remote_status = st.selectbox("Remote Status", metadata["remote_options"])

    with c4:
        experience_level = st.selectbox("Experience Level", metadata["experience_options"])
        views = st.number_input("Expected Views", min_value=0, max_value=100000, value=15, step=1)
        applies = st.number_input("Expected Applications", min_value=0, max_value=100000, value=2, step=1)
        selected_skills = st.multiselect("Required Skills", metadata["skill_options"], default=metadata["skill_options"][:1])

    if st.button("Predict Salary Level", type="primary"):
        skills_text = " ".join([skill.replace(" ", "_") for skill in selected_skills]) if selected_skills else "Unknown_Skill"

        input_data = pd.DataFrame([
            {
                "title_group": title_group,
                "location_group": location_group,
                "work_type_final": work_type,
                "remote_status": remote_status,
                "views": views,
                "applies": applies,
                "experience_level_final": experience_level,
                "skills_text": skills_text
            }
        ])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = model.named_steps["classifier"].classes_

        st.success(f"Predicted Salary Level: {prediction}")

        probability_df = pd.DataFrame({
            "salary_level": classes,
            "probability": probabilities
        }).sort_values("probability", ascending=False)

        c5, c6 = st.columns([1, 1])

        with c5:
            fig = px.bar(
                probability_df,
                x="salary_level",
                y="probability",
                color="salary_level",
                color_discrete_sequence=PALETTE,
                title="Prediction Probability by Salary Level",
                text="probability"
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", textfont_color="#0F172A")
            fig.update_yaxes(range=[0, 1])
            fig = styled_plotly(fig, 430)
            st.plotly_chart(fig, use_container_width=True)

        with c6:
            plot_donut(probability_df, "salary_level", "probability", "Prediction Probability Share", height=430)

    st.markdown(
        """
        <div class="warning-box">
        The interactive model is used for dashboard demonstration. The official evaluation results are taken from the Spark MLlib model shown above.
        </div>
        """,
        unsafe_allow_html=True
    )


def recommendations_tab(filtered_jobs, filtered_skills):
    st.markdown('<div class="section-title">Simple Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">These recommendations are generated from the current filtered dashboard results.</div>', unsafe_allow_html=True)

    top_skills = (
        filtered_skills
        .groupby("skill_name")
        .agg(jobs_count=("job_id", "nunique"))
        .reset_index()
        .sort_values("jobs_count", ascending=False)
        .head(5)
    )

    salary_skills = (
        valid_salary_data(filtered_skills)
        .groupby("skill_name")
        .agg(
            jobs_count=("job_id", "nunique"),
            avg_salary=("salary_value", "mean")
        )
        .reset_index()
    )

    salary_skills = salary_skills[salary_skills["jobs_count"] >= 5]
    salary_skills = salary_skills.sort_values("avg_salary", ascending=False).head(5)

    remote_skills = (
        filtered_skills[filtered_skills["remote_status"] == "Remote"]
        .groupby("skill_name")
        .agg(jobs_count=("job_id", "nunique"))
        .reset_index()
        .sort_values("jobs_count", ascending=False)
        .head(5)
    )

    if not top_skills.empty:
        skills_text = ", ".join(top_skills["skill_name"].tolist())
        st.markdown(
            f"""
            <div class="recommendation">
            <b>In-demand skills:</b><br>
            The most in-demand skills in the current filtered data are <b>{skills_text}</b>.
            Job seekers can use this insight to understand which skill areas appear most often.
            </div>
            """,
            unsafe_allow_html=True
        )

    if not salary_skills.empty:
        salary_text = ", ".join(salary_skills["skill_name"].tolist())
        st.markdown(
            f"""
            <div class="recommendation">
            <b>High-salary skills:</b><br>
            Higher salary jobs are more associated with skills such as <b>{salary_text}</b>.
            These skills may be helpful for long-term career planning.
            </div>
            """,
            unsafe_allow_html=True
        )

    if not remote_skills.empty:
        remote_text = ", ".join(remote_skills["skill_name"].tolist())
        st.markdown(
            f"""
            <div class="recommendation">
            <b>Remote work skills:</b><br>
            For remote jobs, the most common skills are <b>{remote_text}</b>.
            This is useful for job seekers who prefer remote opportunities.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="warning-box">
        These recommendations are based only on the dataset results. They should be used as supportive insights, not as final career decisions.
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    cleaned_postings, jobs_with_skills, job_skill_summary = load_processed_data()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Job Market Big Data Analytics Dashboard</div>
            <div class="hero-subtitle">
                Big Data Analytics for Job Market Trends, In-Demand Skills, and Salary Insights.
                This dashboard presents processed LinkedIn job posting data through interactive charts, salary analysis, engagement insights, and salary level prediction.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_work_type, selected_remote, selected_skill, salary_range = create_filters(cleaned_postings, jobs_with_skills)

    filtered_jobs, filtered_skills = apply_filters(
        cleaned_postings,
        jobs_with_skills,
        selected_work_type,
        selected_remote,
        selected_skill
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview",
        "Job Trends",
        "Skills Demand",
        "Salary Insights",
        "Engagement",
        "ML Prediction",
        "Recommendations"
    ])

    with tab1:
        overview_tab(filtered_jobs, filtered_skills, salary_range)

    with tab2:
        job_trends_tab(filtered_jobs)

    with tab3:
        skills_tab(filtered_skills)

    with tab4:
        salary_tab(filtered_jobs, filtered_skills, salary_range)

    with tab5:
        engagement_tab(filtered_jobs, filtered_skills)

    with tab6:
        ml_tab(cleaned_postings, jobs_with_skills)

    with tab7:
        recommendations_tab(filtered_jobs, filtered_skills)


if __name__ == "__main__":
    main()