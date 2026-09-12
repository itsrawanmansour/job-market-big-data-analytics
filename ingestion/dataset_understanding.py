from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

POSTINGS_PATH = BASE_DIR / "data" / "raw" / "postings.csv"
JOB_SKILLS_PATH = BASE_DIR / "data" / "raw" / "job_skills.csv"
SKILLS_PATH = BASE_DIR / "data" / "raw" / "skills.csv"

OUTPUT_DIR = BASE_DIR / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

postings_columns = [
    "job_id",
    "company_name",
    "title",
    "description",
    "min_salary",
    "med_salary",
    "max_salary",
    "pay_period",
    "location",
    "views",
    "applies",
    "remote_allowed",
    "formatted_work_type",
    "formatted_experience_level",
    "work_type",
    "currency",
    "normalized_salary",
]

print("Reading postings.csv...")
postings = pd.read_csv(
    POSTINGS_PATH,
    usecols=lambda col: col in postings_columns,
    low_memory=False,
)

print("Reading job_skills.csv...")
job_skills = pd.read_csv(JOB_SKILLS_PATH, low_memory=False)

print("Reading skills.csv...")
skills = pd.read_csv(SKILLS_PATH, low_memory=False)

def is_not_empty(series):
    return series.notna() & series.astype(str).str.strip().ne("")

def safe_unique_count(df, column):
    if column not in df.columns:
        return 0
    return df.loc[is_not_empty(df[column]), column].nunique()

def missing_count(df, column):
    if column not in df.columns:
        return "Column not found"
    return int(df[column].isna().sum())

salary_cols = ["min_salary", "med_salary", "max_salary", "normalized_salary"]
salary_mask_any = False

for col in salary_cols:
    if col in postings.columns:
        numeric_col = pd.to_numeric(postings[col], errors="coerce")
        salary_mask_any = salary_mask_any | numeric_col.notna()

normalized_salary = pd.to_numeric(
    postings.get("normalized_salary", pd.Series(dtype=float)),
    errors="coerce"
)

remote_series = postings.get("remote_allowed", pd.Series(dtype=object)).fillna("")
remote_mask = remote_series.astype(str).str.strip().isin(["1", "1.0", "true", "True", "TRUE"])

work_type_col = "formatted_work_type" if "formatted_work_type" in postings.columns else "work_type"

posted_job_ids = set(postings["job_id"].dropna().astype(str))
job_skill_job_ids = set(job_skills["job_id"].dropna().astype(str))
jobs_with_skills = posted_job_ids.intersection(job_skill_job_ids)

skill_abrs_job_skills = set(job_skills["skill_abr"].dropna().astype(str).str.strip())
skill_abrs_mapping = set(skills["skill_abr"].dropna().astype(str).str.strip())
mapped_skill_abrs = skill_abrs_job_skills.intersection(skill_abrs_mapping)
missing_skill_abrs = skill_abrs_job_skills.difference(skill_abrs_mapping)

skills_joined = job_skills.merge(skills, on="skill_abr", how="left")

profile_lines = []

profile_lines.append("PHASE 1 - DATASET UNDERSTANDING")
profile_lines.append("=" * 50)

profile_lines.append("\n1. BASIC DATASET SIZE")
profile_lines.append(f"Number of rows in postings.csv: {len(postings):,}")
profile_lines.append(f"Number of columns loaded from postings.csv: {len(postings.columns):,}")
profile_lines.append(f"Number of rows in job_skills.csv: {len(job_skills):,}")
profile_lines.append(f"Number of unique job_id in job_skills.csv: {job_skills['job_id'].nunique():,}")
profile_lines.append(f"Number of rows in skills.csv: {len(skills):,}")
profile_lines.append(f"Number of unique skills in skills.csv: {skills['skill_abr'].nunique():,}")

profile_lines.append("\n2. MAIN DATASET INFORMATION")
profile_lines.append(f"Number of unique companies: {safe_unique_count(postings, 'company_name'):,}")
profile_lines.append(f"Number of unique locations: {safe_unique_count(postings, 'location'):,}")
profile_lines.append(f"Number of unique job titles: {safe_unique_count(postings, 'title'):,}")

profile_lines.append("\n3. SALARY AVAILABILITY")
profile_lines.append(f"Jobs with normalized_salary: {int(normalized_salary.notna().sum()):,}")
profile_lines.append(f"Jobs without normalized_salary: {int(normalized_salary.isna().sum()):,}")
profile_lines.append(f"Jobs with any salary value: {int(salary_mask_any.sum()):,}")
profile_lines.append(f"Jobs without any salary value: {int((~salary_mask_any).sum()):,}")

profile_lines.append("\n4. WORK TYPE DISTRIBUTION")
if work_type_col in postings.columns:
    work_counts = postings[work_type_col].fillna("Unknown").astype(str).str.strip()
    work_counts = work_counts.replace("", "Unknown").value_counts().head(10)
    for name, count in work_counts.items():
        profile_lines.append(f"{name}: {count:,}")
else:
    profile_lines.append("Work type column not found")

profile_lines.append("\n5. REMOTE JOBS")
remote_count = int(remote_mask.sum())
total_jobs = len(postings)
profile_lines.append(f"Remote jobs count: {remote_count:,}")
profile_lines.append(f"Non-remote / unknown jobs count: {total_jobs - remote_count:,}")
profile_lines.append(f"Remote jobs percentage: {(remote_count / total_jobs) * 100:.2f}%")

profile_lines.append("\n6. MISSING VALUES IN IMPORTANT COLUMNS")
important_columns = [
    "company_name",
    "title",
    "description",
    "location",
    "views",
    "applies",
    "normalized_salary",
    "formatted_work_type",
    "formatted_experience_level",
    "remote_allowed",
]
for col in important_columns:
    profile_lines.append(f"{col}: {missing_count(postings, col)}")

profile_lines.append("\n7. JOB-SKILL MAPPING COVERAGE")
profile_lines.append(f"Jobs in postings that have at least one skill: {len(jobs_with_skills):,}")
profile_lines.append(f"Jobs in postings without skills: {len(posted_job_ids) - len(jobs_with_skills):,}")
profile_lines.append(f"Skill abbreviations in job_skills.csv: {len(skill_abrs_job_skills):,}")
profile_lines.append(f"Skill abbreviations successfully mapped to skills.csv: {len(mapped_skill_abrs):,}")
profile_lines.append(f"Skill abbreviations missing from skills.csv: {len(missing_skill_abrs):,}")

profile_lines.append("\n8. TOP 10 JOB TITLES")
top_titles = postings["title"].dropna().astype(str).str.strip().value_counts().head(10)
for name, count in top_titles.items():
    profile_lines.append(f"{name}: {count:,}")

profile_lines.append("\n9. TOP 10 COMPANIES")
top_companies = postings["company_name"].dropna().astype(str).str.strip()
top_companies = top_companies[top_companies.ne("")].value_counts().head(10)
for name, count in top_companies.items():
    profile_lines.append(f"{name}: {count:,}")

profile_lines.append("\n10. TOP 10 LOCATIONS")
top_locations = postings["location"].dropna().astype(str).str.strip().value_counts().head(10)
for name, count in top_locations.items():
    profile_lines.append(f"{name}: {count:,}")

profile_lines.append("\n11. TOP 10 SKILLS")
top_skills = skills_joined["skill_name"].dropna().astype(str).str.strip().value_counts().head(10)
for name, count in top_skills.items():
    profile_lines.append(f"{name}: {count:,}")

output_text = "\n".join(profile_lines)

print("\n" + output_text)

output_file = OUTPUT_DIR / "dataset_profile.txt"
output_file.write_text(output_text, encoding="utf-8")

print(f"\nDataset profile saved to: {output_file}")