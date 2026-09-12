from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "output"

CLEANED_POSTINGS_PATH = PROCESSED_DIR / "cleaned_postings.parquet"
JOBS_WITH_SKILLS_PATH = PROCESSED_DIR / "jobs_with_skills.parquet"
JOB_SKILL_SUMMARY_PATH = PROCESSED_DIR / "job_skill_summary.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_file_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def load_processed_data():
    check_file_exists(CLEANED_POSTINGS_PATH)
    check_file_exists(JOBS_WITH_SKILLS_PATH)

    cleaned_postings = pd.read_parquet(CLEANED_POSTINGS_PATH)
    jobs_with_skills = pd.read_parquet(JOBS_WITH_SKILLS_PATH)

    return cleaned_postings, jobs_with_skills


def create_job_skill_summary(jobs_with_skills):
    salary_data = jobs_with_skills.copy()

    summary = (
        salary_data
        .groupby("skill_name", dropna=False)
        .agg(
            jobs_count=("job_id", "nunique"),
            avg_salary=("salary_value", "mean"),
            median_salary=("salary_value", "median"),
            total_views=("views", "sum"),
            avg_views=("views", "mean"),
            total_applies=("applies", "sum"),
            avg_applies=("applies", "mean"),
            remote_jobs_count=("remote_status", lambda x: (x == "Remote").sum()),
            high_salary_jobs=("salary_level", lambda x: (x == "High").sum())
        )
        .reset_index()
    )

    summary = summary.sort_values(by="jobs_count", ascending=False)

    summary.to_parquet(JOB_SKILL_SUMMARY_PATH, index=False, engine="pyarrow")

    return summary


def generate_processed_storage_summary(cleaned_postings, jobs_with_skills, job_skill_summary):
    lines = []
    lines.append(" PROCESSED STORAGE SUMMARY")
    lines.append("=" * 50)

    lines.append("\n1. PROCESSED FILES")
    lines.append(f"cleaned_postings.parquet: {CLEANED_POSTINGS_PATH}")
    lines.append(f"jobs_with_skills.parquet: {JOBS_WITH_SKILLS_PATH}")
    lines.append(f"job_skill_summary.parquet: {JOB_SKILL_SUMMARY_PATH}")

    lines.append("\n2. FILE RECORD COUNTS")
    lines.append(f"Rows in cleaned_postings.parquet: {len(cleaned_postings):,}")
    lines.append(f"Rows in jobs_with_skills.parquet: {len(jobs_with_skills):,}")
    lines.append(f"Rows in job_skill_summary.parquet: {len(job_skill_summary):,}")

    lines.append("\n3. DATA COVERAGE")
    lines.append(f"Unique jobs in cleaned_postings: {cleaned_postings['job_id'].nunique():,}")
    lines.append(f"Unique jobs in jobs_with_skills: {jobs_with_skills['job_id'].nunique():,}")
    lines.append(f"Unique skills in jobs_with_skills: {jobs_with_skills['skill_name'].nunique():,}")

    lines.append("\n4. TOP 10 SKILLS FROM JOB SKILL SUMMARY")
    top_skills = job_skill_summary.head(10)

    for _, row in top_skills.iterrows():
        lines.append(f"{row['skill_name']}: {int(row['jobs_count']):,} jobs")

    lines.append("\nProcessed storage verification completed successfully.")

    output_text = "\n".join(lines)

    output_file = OUTPUT_DIR / "processed_storage_summary.txt"
    output_file.write_text(output_text, encoding="utf-8")

    print(output_text)
    print(f"\nProcessed storage summary saved to: {output_file}")


def main():
    cleaned_postings, jobs_with_skills = load_processed_data()
    job_skill_summary = create_job_skill_summary(jobs_with_skills)
    generate_processed_storage_summary(cleaned_postings, jobs_with_skills, job_skill_summary)


if __name__ == "__main__":
    main()