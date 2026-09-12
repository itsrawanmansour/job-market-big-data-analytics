from pathlib import Path
import shutil
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

SOURCE_DIR = BASE_DIR / "data" / "source"
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "output"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

source_files = {
    "postings": [
        SOURCE_DIR / "postings.csv",
        RAW_DIR / "postings.csv"
    ],
    "job_skills": [
        SOURCE_DIR / "jobs" / "job_skills.csv",
        RAW_DIR / "job_skills.csv"
    ],
    "skills": [
        SOURCE_DIR / "mappings" / "skills.csv",
        RAW_DIR / "skills.csv"
    ]
}

raw_files = {
    "postings": RAW_DIR / "postings.csv",
    "job_skills": RAW_DIR / "job_skills.csv",
    "skills": RAW_DIR / "skills.csv"
}

required_columns = {
    "postings": [
        "job_id",
        "company_name",
        "title",
        "description",
        "max_salary",
        "med_salary",
        "min_salary",
        "pay_period",
        "location",
        "views",
        "applies",
        "remote_allowed",
        "formatted_work_type",
        "formatted_experience_level",
        "work_type",
        "currency",
        "normalized_salary"
    ],
    "job_skills": [
        "job_id",
        "skill_abr"
    ],
    "skills": [
        "skill_abr",
        "skill_name"
    ]
}

def find_source_file(name):
    for path in source_files[name]:
        if path.exists():
            return path
    raise FileNotFoundError(f"Source file for {name} was not found.")

def copy_to_raw(name):
    source_path = find_source_file(name)
    target_path = raw_files[name]

    if source_path.resolve() == target_path.resolve():
        return source_path, target_path, "Already exists in raw folder"

    shutil.copy2(source_path, target_path)
    return source_path, target_path, "Copied to raw folder"

def read_header(path):
    return list(pd.read_csv(path, nrows=0).columns)

def count_rows(path):
    total = 0
    for chunk in pd.read_csv(path, chunksize=50000, low_memory=False):
        total += len(chunk)
    return total

def validate_columns(name, columns):
    missing = [col for col in required_columns[name] if col not in columns]
    return missing

summary = {
    "phase": "Phase 4 - Data Ingestion",
    "status": "success",
    "files": {}
}

text_lines = []
text_lines.append("PHASE 4 - DATA INGESTION")
text_lines.append("=" * 50)

try:
    for name in ["postings", "job_skills", "skills"]:
        source_path, raw_path, copy_status = copy_to_raw(name)
        columns = read_header(raw_path)
        missing_columns = validate_columns(name, columns)
        rows_count = count_rows(raw_path)

        if missing_columns:
            summary["status"] = "failed"

        file_info = {
            "source_path": str(source_path),
            "raw_path": str(raw_path),
            "copy_status": copy_status,
            "rows_count": rows_count,
            "columns_count": len(columns),
            "missing_required_columns": missing_columns
        }

        summary["files"][name] = file_info

        text_lines.append(f"\n{name}.csv")
        text_lines.append(f"Source path: {source_path}")
        text_lines.append(f"Raw path: {raw_path}")
        text_lines.append(f"Copy status: {copy_status}")
        text_lines.append(f"Rows count: {rows_count:,}")
        text_lines.append(f"Columns count: {len(columns):,}")

        if missing_columns:
            text_lines.append(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            text_lines.append("Required columns validation: Passed")

    if summary["status"] == "success":
        text_lines.append("\nRaw data loaded successfully.")
        text_lines.append("Dataset profile generated.")
    else:
        text_lines.append("\nData ingestion completed with validation errors.")

except Exception as error:
    summary["status"] = "failed"
    summary["error"] = str(error)
    text_lines.append(f"\nError: {error}")

summary_path = OUTPUT_DIR / "ingestion_summary.json"
text_path = OUTPUT_DIR / "ingestion_summary.txt"

summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")
text_path.write_text("\n".join(text_lines), encoding="utf-8")

print("\n".join(text_lines))
print(f"\nIngestion summary saved to: {text_path}")
print(f"Ingestion JSON saved to: {summary_path}")