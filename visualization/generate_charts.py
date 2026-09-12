from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]

ANALYTICS_DIR = BASE_DIR / "data" / "output" / "analytics"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_bar_chart(df, x_col, y_col, title, xlabel, ylabel, file_name, top_n=None):
    data = df.copy()

    if top_n is not None:
        data = data.head(top_n)

    plt.figure(figsize=(12, 6))
    plt.bar(data[x_col].astype(str), data[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / file_name, dpi=300)
    plt.close()


def save_horizontal_bar_chart(df, x_col, y_col, title, xlabel, ylabel, file_name, top_n=None):
    data = df.copy()

    if top_n is not None:
        data = data.head(top_n)

    data = data.iloc[::-1]

    plt.figure(figsize=(12, 7))
    plt.barh(data[y_col].astype(str), data[x_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / file_name, dpi=300)
    plt.close()


def save_engagement_summary_chart(df):
    data = df[df["metric"].isin(["Average Views", "Average Applies"])].copy()

    plt.figure(figsize=(8, 5))
    plt.bar(data["metric"], data["value"])
    plt.title("Average Views and Applications")
    plt.xlabel("Metric")
    plt.ylabel("Average Value")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "engagement_summary.png", dpi=300)
    plt.close()


def main():
    top_job_titles = pd.read_csv(ANALYTICS_DIR / "top_job_titles.csv")
    top_skills = pd.read_csv(ANALYTICS_DIR / "top_skills.csv")
    work_type_distribution = pd.read_csv(ANALYTICS_DIR / "work_type_distribution.csv")
    salary_level_distribution = pd.read_csv(ANALYTICS_DIR / "salary_level_distribution.csv")
    salary_by_skill = pd.read_csv(ANALYTICS_DIR / "salary_by_skill.csv")
    remote_salary_comparison = pd.read_csv(ANALYTICS_DIR / "remote_salary_comparison.csv")
    engagement_summary = pd.read_csv(ANALYTICS_DIR / "engagement_summary.csv")

    save_bar_chart(
        top_job_titles,
        "title",
        "jobs_count",
        "Top 10 Job Titles",
        "Job Title",
        "Number of Jobs",
        "top_job_titles.png"
    )

    save_horizontal_bar_chart(
        top_skills,
        "jobs_count",
        "skill_name",
        "Top 20 In-Demand Skills",
        "Number of Jobs",
        "Skill",
        "top_skills.png",
        top_n=20
    )

    save_bar_chart(
        work_type_distribution,
        "work_type_final",
        "jobs_count",
        "Work Type Distribution",
        "Work Type",
        "Number of Jobs",
        "work_type_distribution.png"
    )

    save_bar_chart(
        salary_level_distribution,
        "salary_level",
        "jobs_count",
        "Salary Level Distribution",
        "Salary Level",
        "Number of Jobs",
        "salary_level_distribution.png"
    )

    save_horizontal_bar_chart(
        salary_by_skill,
        "avg_salary",
        "skill_name",
        "Average Salary by Skill",
        "Average Salary",
        "Skill",
        "salary_by_skill.png",
        top_n=10
    )

    save_bar_chart(
        remote_salary_comparison,
        "remote_status",
        "avg_salary",
        "Remote vs Non-Remote Average Salary",
        "Remote Status",
        "Average Salary",
        "remote_salary_comparison.png"
    )

    save_engagement_summary_chart(engagement_summary)

    print("Charts generated successfully.")
    print(f"Charts saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()