from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYTICS_DIR = BASE_DIR / "data" / "output" / "analytics"
OUTPUT_DIR = BASE_DIR / "data" / "output"

CLEANED_POSTINGS_PATH = PROCESSED_DIR / "cleaned_postings.parquet"
JOBS_WITH_SKILLS_PATH = PROCESSED_DIR / "jobs_with_skills.parquet"
JOB_SKILL_SUMMARY_PATH = PROCESSED_DIR / "job_skill_summary.parquet"

SALARY_MIN = 10000
SALARY_MAX = 500000

ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    cleaned_postings = pd.read_parquet(CLEANED_POSTINGS_PATH)
    jobs_with_skills = pd.read_parquet(JOBS_WITH_SKILLS_PATH)
    job_skill_summary = pd.read_parquet(JOB_SKILL_SUMMARY_PATH)

    return cleaned_postings, jobs_with_skills, job_skill_summary


def save_output(df, file_name):
    parquet_path = ANALYTICS_DIR / file_name
    csv_path = ANALYTICS_DIR / file_name.replace(".parquet", ".csv")

    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return parquet_path


def filter_valid_salary(data):
    filtered = data.copy()
    filtered["salary_value"] = pd.to_numeric(filtered["salary_value"], errors="coerce")

    filtered = filtered[
        (filtered["salary_value"].notna()) &
        (filtered["salary_value"] >= SALARY_MIN) &
        (filtered["salary_value"] <= SALARY_MAX)
    ]

    return filtered


def salary_filter_summary(cleaned_postings):
    salary_values = pd.to_numeric(cleaned_postings["salary_value"], errors="coerce")

    available_salary_count = salary_values.notna().sum()
    valid_salary_count = (
        (salary_values.notna()) &
        (salary_values >= SALARY_MIN) &
        (salary_values <= SALARY_MAX)
    ).sum()

    removed_outliers_count = available_salary_count - valid_salary_count

    result = pd.DataFrame([
        {
            "metric": "Salary records before filtering",
            "value": int(available_salary_count)
        },
        {
            "metric": "Salary records after filtering",
            "value": int(valid_salary_count)
        },
        {
            "metric": "Removed invalid or outlier salary records",
            "value": int(removed_outliers_count)
        },
        {
            "metric": "Minimum accepted salary",
            "value": SALARY_MIN
        },
        {
            "metric": "Maximum accepted salary",
            "value": SALARY_MAX
        }
    ])

    return result


def top_job_titles(cleaned_postings):
    result = (
        cleaned_postings
        .groupby("title")
        .size()
        .reset_index(name="jobs_count")
        .sort_values(by="jobs_count", ascending=False)
        .head(10)
    )

    return result


def top_companies(cleaned_postings):
    data = cleaned_postings.copy()
    data = data[data["company_name"].notna()]
    data = data[data["company_name"].astype(str).str.strip() != ""]
    data = data[data["company_name"] != "Unknown Company"]

    result = (
        data
        .groupby("company_name")
        .size()
        .reset_index(name="jobs_count")
        .sort_values(by="jobs_count", ascending=False)
        .head(10)
    )

    return result


def top_locations(cleaned_postings):
    result = (
        cleaned_postings
        .groupby("location")
        .size()
        .reset_index(name="jobs_count")
        .sort_values(by="jobs_count", ascending=False)
        .head(10)
    )

    return result


def work_type_distribution(cleaned_postings):
    result = (
        cleaned_postings
        .groupby("work_type_final")
        .size()
        .reset_index(name="jobs_count")
        .sort_values(by="jobs_count", ascending=False)
    )

    return result


def top_skills(jobs_with_skills):
    result = (
        jobs_with_skills
        .groupby("skill_name")
        .agg(jobs_count=("job_id", "nunique"))
        .reset_index()
        .sort_values(by="jobs_count", ascending=False)
        .head(20)
    )

    return result


def skills_by_title(jobs_with_skills):
    data = (
        jobs_with_skills
        .groupby(["title", "skill_name"])
        .agg(jobs_count=("job_id", "nunique"))
        .reset_index()
    )

    data["rank"] = data.groupby("title")["jobs_count"].rank(method="first", ascending=False)

    result = (
        data[data["rank"] <= 5]
        .sort_values(["title", "rank"])
        .reset_index(drop=True)
    )

    return result


def high_salary_skills(jobs_with_skills):
    data = filter_valid_salary(jobs_with_skills)
    data = data[data["salary_level"] == "High"]

    result = (
        data
        .groupby("skill_name")
        .agg(
            jobs_count=("job_id", "nunique"),
            avg_salary=("salary_value", "mean"),
            median_salary=("salary_value", "median")
        )
        .reset_index()
        .sort_values(by="jobs_count", ascending=False)
        .head(20)
    )

    return result


def remote_skills(jobs_with_skills):
    data = jobs_with_skills[jobs_with_skills["remote_status"] == "Remote"]

    result = (
        data
        .groupby("skill_name")
        .agg(jobs_count=("job_id", "nunique"))
        .reset_index()
        .sort_values(by="jobs_count", ascending=False)
        .head(20)
    )

    return result


def salary_by_title(cleaned_postings):
    data = filter_valid_salary(cleaned_postings)

    result = (
        data
        .groupby("title")
        .agg(
            jobs_count=("job_id", "nunique"),
            avg_salary=("salary_value", "mean"),
            median_salary=("salary_value", "median")
        )
        .reset_index()
    )

    result = result[result["jobs_count"] >= 5]
    result = result.sort_values(by="avg_salary", ascending=False).head(20)

    return result


def salary_by_skill(jobs_with_skills):
    data = filter_valid_salary(jobs_with_skills)

    result = (
        data
        .groupby("skill_name")
        .agg(
            jobs_count=("job_id", "nunique"),
            avg_salary=("salary_value", "mean"),
            median_salary=("salary_value", "median")
        )
        .reset_index()
    )

    result = result[result["jobs_count"] >= 5]
    result = result.sort_values(by="avg_salary", ascending=False)

    return result


def salary_by_location(cleaned_postings):
    data = filter_valid_salary(cleaned_postings)

    result = (
        data
        .groupby("location")
        .agg(
            jobs_count=("job_id", "nunique"),
            avg_salary=("salary_value", "mean"),
            median_salary=("salary_value", "median")
        )
        .reset_index()
    )

    result = result[result["jobs_count"] >= 10]
    result = result.sort_values(by="avg_salary", ascending=False).head(20)

    return result


def salary_level_distribution(cleaned_postings):
    result = (
        cleaned_postings
        .groupby("salary_level")
        .size()
        .reset_index(name="jobs_count")
        .sort_values(by="jobs_count", ascending=False)
    )

    return result


def remote_salary_comparison(cleaned_postings):
    data = filter_valid_salary(cleaned_postings)

    result = (
        data
        .groupby("remote_status")
        .agg(
            jobs_count=("job_id", "nunique"),
            avg_salary=("salary_value", "mean"),
            median_salary=("salary_value", "median")
        )
        .reset_index()
        .sort_values(by="avg_salary", ascending=False)
    )

    return result


def top_viewed_jobs(cleaned_postings):
    columns = [
        "job_id",
        "title",
        "company_name",
        "location",
        "views",
        "applies",
        "salary_value",
        "salary_level",
        "work_type_final",
        "remote_status"
    ]

    result = (
        cleaned_postings[columns]
        .sort_values(by="views", ascending=False)
        .head(20)
    )

    return result


def top_applied_jobs(cleaned_postings):
    columns = [
        "job_id",
        "title",
        "company_name",
        "location",
        "views",
        "applies",
        "salary_value",
        "salary_level",
        "work_type_final",
        "remote_status"
    ]

    result = (
        cleaned_postings[columns]
        .sort_values(by="applies", ascending=False)
        .head(20)
    )

    return result


def engagement_by_skill(jobs_with_skills):
    result = (
        jobs_with_skills
        .groupby("skill_name")
        .agg(
            jobs_count=("job_id", "nunique"),
            total_views=("views", "sum"),
            avg_views=("views", "mean"),
            total_applies=("applies", "sum"),
            avg_applies=("applies", "mean")
        )
        .reset_index()
        .sort_values(by="avg_views", ascending=False)
    )

    return result


def engagement_summary(cleaned_postings):
    data = cleaned_postings.copy()

    views_applies_data = data[["views", "applies"]].dropna()

    if len(views_applies_data) > 1:
        correlation = views_applies_data["views"].corr(views_applies_data["applies"])
    else:
        correlation = None

    summary = pd.DataFrame([
        {
            "metric": "Total Views",
            "value": float(data["views"].sum())
        },
        {
            "metric": "Total Applies",
            "value": float(data["applies"].sum())
        },
        {
            "metric": "Average Views",
            "value": float(data["views"].mean())
        },
        {
            "metric": "Average Applies",
            "value": float(data["applies"].mean())
        },
        {
            "metric": "Views-Applies Correlation",
            "value": float(correlation) if correlation is not None else None
        }
    ])

    return summary


def generate_text_summary(results):
    lines = []
    lines.append("PHASE 8 - ANALYTICS AND INSIGHTS SUMMARY")
    lines.append("=" * 50)

    lines.append("\n0. SALARY FILTER SUMMARY")
    for _, row in results["salary_filter_summary"].iterrows():
        lines.append(f"{row['metric']}: {int(row['value']):,}")

    lines.append("\n1. TOP 10 JOB TITLES")
    for _, row in results["top_job_titles"].iterrows():
        lines.append(f"{row['title']}: {int(row['jobs_count']):,}")

    lines.append("\n2. TOP 10 COMPANIES")
    for _, row in results["top_companies"].iterrows():
        lines.append(f"{row['company_name']}: {int(row['jobs_count']):,}")

    lines.append("\n3. TOP 10 LOCATIONS")
    for _, row in results["top_locations"].iterrows():
        lines.append(f"{row['location']}: {int(row['jobs_count']):,}")

    lines.append("\n4. WORK TYPE DISTRIBUTION")
    for _, row in results["work_type_distribution"].iterrows():
        lines.append(f"{row['work_type_final']}: {int(row['jobs_count']):,}")

    lines.append("\n5. TOP 20 SKILLS")
    for _, row in results["top_skills"].iterrows():
        lines.append(f"{row['skill_name']}: {int(row['jobs_count']):,}")

    lines.append("\n6. TOP HIGH-SALARY SKILLS")
    for _, row in results["high_salary_skills"].head(10).iterrows():
        lines.append(
            f"{row['skill_name']}: {int(row['jobs_count']):,} jobs, "
            f"average salary {row['avg_salary']:,.2f}, "
            f"median salary {row['median_salary']:,.2f}"
        )

    lines.append("\n7. TOP REMOTE JOB SKILLS")
    for _, row in results["remote_skills"].head(10).iterrows():
        lines.append(f"{row['skill_name']}: {int(row['jobs_count']):,}")

    lines.append("\n8. TOP SALARY BY JOB TITLE")
    for _, row in results["salary_by_title"].head(10).iterrows():
        lines.append(
            f"{row['title']}: average salary {row['avg_salary']:,.2f}, "
            f"median salary {row['median_salary']:,.2f}, "
            f"jobs {int(row['jobs_count']):,}"
        )

    lines.append("\n9. TOP SALARY BY SKILL")
    for _, row in results["salary_by_skill"].head(10).iterrows():
        lines.append(
            f"{row['skill_name']}: average salary {row['avg_salary']:,.2f}, "
            f"median salary {row['median_salary']:,.2f}, "
            f"jobs {int(row['jobs_count']):,}"
        )

    lines.append("\n10. TOP SALARY BY LOCATION")
    for _, row in results["salary_by_location"].head(10).iterrows():
        lines.append(
            f"{row['location']}: average salary {row['avg_salary']:,.2f}, "
            f"median salary {row['median_salary']:,.2f}, "
            f"jobs {int(row['jobs_count']):,}"
        )

    lines.append("\n11. SALARY LEVEL DISTRIBUTION")
    for _, row in results["salary_level_distribution"].iterrows():
        lines.append(f"{row['salary_level']}: {int(row['jobs_count']):,}")

    lines.append("\n12. REMOTE SALARY COMPARISON")
    for _, row in results["remote_salary_comparison"].iterrows():
        lines.append(
            f"{row['remote_status']}: average salary {row['avg_salary']:,.2f}, "
            f"median salary {row['median_salary']:,.2f}, "
            f"jobs {int(row['jobs_count']):,}"
        )

    lines.append("\n13. ENGAGEMENT SUMMARY")
    for _, row in results["engagement_summary"].iterrows():
        value = row["value"]

        if pd.isna(value):
            lines.append(f"{row['metric']}: Not available")
        else:
            lines.append(f"{row['metric']}: {value:,.4f}")

    lines.append("\nAnalytics outputs generated successfully.")

    output_text = "\n".join(lines)

    summary_path = OUTPUT_DIR / "analytics_summary.txt"
    summary_path.write_text(output_text, encoding="utf-8")

    print(output_text)
    print(f"\nAnalytics summary saved to: {summary_path}")


def main():
    cleaned_postings, jobs_with_skills, job_skill_summary = load_data()

    results = {
        "salary_filter_summary": salary_filter_summary(cleaned_postings),
        "top_job_titles": top_job_titles(cleaned_postings),
        "top_companies": top_companies(cleaned_postings),
        "top_locations": top_locations(cleaned_postings),
        "work_type_distribution": work_type_distribution(cleaned_postings),
        "top_skills": top_skills(jobs_with_skills),
        "skills_by_title": skills_by_title(jobs_with_skills),
        "high_salary_skills": high_salary_skills(jobs_with_skills),
        "remote_skills": remote_skills(jobs_with_skills),
        "salary_by_title": salary_by_title(cleaned_postings),
        "salary_by_skill": salary_by_skill(jobs_with_skills),
        "salary_by_location": salary_by_location(cleaned_postings),
        "salary_level_distribution": salary_level_distribution(cleaned_postings),
        "remote_salary_comparison": remote_salary_comparison(cleaned_postings),
        "top_viewed_jobs": top_viewed_jobs(cleaned_postings),
        "top_applied_jobs": top_applied_jobs(cleaned_postings),
        "engagement_by_skill": engagement_by_skill(jobs_with_skills),
        "engagement_summary": engagement_summary(cleaned_postings)
    }

    output_files = {
        "salary_filter_summary": "salary_filter_summary.parquet",
        "top_job_titles": "top_job_titles.parquet",
        "top_companies": "top_companies.parquet",
        "top_locations": "top_locations.parquet",
        "work_type_distribution": "work_type_distribution.parquet",
        "top_skills": "top_skills.parquet",
        "skills_by_title": "skills_by_title.parquet",
        "high_salary_skills": "high_salary_skills.parquet",
        "remote_skills": "remote_skills.parquet",
        "salary_by_title": "salary_by_title.parquet",
        "salary_by_skill": "salary_by_skill.parquet",
        "salary_by_location": "salary_by_location.parquet",
        "salary_level_distribution": "salary_level_distribution.parquet",
        "remote_salary_comparison": "remote_salary_comparison.parquet",
        "top_viewed_jobs": "top_viewed_jobs.parquet",
        "top_applied_jobs": "top_applied_jobs.parquet",
        "engagement_by_skill": "engagement_by_skill.parquet",
        "engagement_summary": "engagement_summary.parquet"
    }

    for key, file_name in output_files.items():
        save_output(results[key], file_name)

    generate_text_summary(results)


if __name__ == "__main__":
    main()