
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
import os
import shutil



BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "output"

POSTINGS_PATH = str(RAW_DIR / "postings.csv")
JOB_SKILLS_PATH = str(RAW_DIR / "job_skills.csv")
SKILLS_PATH = str(RAW_DIR / "skills.csv")

CLEANED_POSTINGS_PATH = str(PROCESSED_DIR / "cleaned_postings.parquet")
JOBS_WITH_SKILLS_PATH = str(PROCESSED_DIR / "jobs_with_skills.parquet")

SUMMARY_PATH = OUTPUT_DIR / "spark_processing_summary.txt"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_spark_session():
    os.environ["HADOOP_HOME"] = "C:\\hadoop"
    os.environ["hadoop.home.dir"] = "C:\\hadoop"
    os.environ["PATH"] = "C:\\hadoop\\bin;" + os.environ["PATH"]

    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
    os.environ["PYSPARK_PYTHON"] = "python"

    spark = (
        SparkSession.builder
        .appName("JobMarketBigDataProcessing")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.blockManager.port", "10025")
        .config("spark.driver.port", "10026")
        .config("spark.ui.port", "4040")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark
def read_raw_data(spark):
    postings = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(POSTINGS_PATH)
    )

    job_skills = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(JOB_SKILLS_PATH)
    )

    skills = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(SKILLS_PATH)
    )

    return postings, job_skills, skills


def select_needed_posting_columns(postings):
    needed_columns = [
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
    ]

    existing_columns = [col for col in needed_columns if col in postings.columns]
    return postings.select(*existing_columns)


def clean_text_columns(postings):
    cleaned = postings

    text_columns = [
        "company_name",
        "title",
        "description",
        "location",
        "pay_period",
        "formatted_work_type",
        "formatted_experience_level",
        "work_type",
        "currency"
    ]

    for column in text_columns:
        if column in cleaned.columns:
            cleaned = cleaned.withColumn(
                column,
                F.trim(F.regexp_replace(F.col(column).cast("string"), r"\s+", " "))
            )

    cleaned = cleaned.withColumn(
        "company_name",
        F.when(F.col("company_name").isNull() | (F.col("company_name") == ""), "Unknown Company")
        .otherwise(F.col("company_name"))
    )

    cleaned = cleaned.withColumn(
        "description",
        F.when(F.col("description").isNull() | (F.col("description") == ""), "No Description")
        .otherwise(F.col("description"))
    )

    return cleaned


def cast_numeric_columns(postings):
    numeric_double_columns = [
        "max_salary",
        "med_salary",
        "min_salary",
        "normalized_salary"
    ]

    numeric_integer_columns = [
        "views",
        "applies"
    ]

    cleaned = postings

    for column in numeric_double_columns:
        if column in cleaned.columns:
            cleaned = cleaned.withColumn(column, F.col(column).cast(DoubleType()))

    for column in numeric_integer_columns:
        if column in cleaned.columns:
            cleaned = cleaned.withColumn(column, F.col(column).cast(IntegerType()))

    cleaned = cleaned.withColumn(
        "views",
        F.when(F.col("views").isNull(), 0).otherwise(F.col("views"))
    )

    cleaned = cleaned.withColumn(
        "applies",
        F.when(F.col("applies").isNull(), 0).otherwise(F.col("applies"))
    )

    return cleaned


def clean_remote_column(postings):
    cleaned = postings.withColumn(
        "remote_status",
        F.when(F.col("remote_allowed").cast("string").isin("1", "1.0", "true", "True", "TRUE"), "Remote")
        .otherwise("Not Remote")
    )

    return cleaned


def create_salary_value(postings):
    cleaned = postings.withColumn(
        "salary_value",
        F.when(F.col("normalized_salary").isNotNull(), F.col("normalized_salary"))
        .when(F.col("med_salary").isNotNull(), F.col("med_salary"))
        .when(
            F.col("min_salary").isNotNull() & F.col("max_salary").isNotNull(),
            (F.col("min_salary") + F.col("max_salary")) / F.lit(2)
        )
        .otherwise(None)
    )

    return cleaned


def calculate_salary_thresholds(postings):
    salary_rows = (
        postings
        .where(F.col("salary_value").isNotNull())
        .select("salary_value")
    )

    count_salary = salary_rows.count()

    if count_salary == 0:
        return None, None

    p33, p66 = salary_rows.approxQuantile("salary_value", [0.33, 0.66], 0.01)
    return p33, p66


def create_salary_level(postings, p33, p66):
    if p33 is None or p66 is None:
        return postings.withColumn("salary_level", F.lit("Unknown"))

    cleaned = postings.withColumn(
        "salary_level",
        F.when(F.col("salary_value").isNull(), "Unknown")
        .when(F.col("salary_value") <= F.lit(p33), "Low")
        .when((F.col("salary_value") > F.lit(p33)) & (F.col("salary_value") <= F.lit(p66)), "Medium")
        .otherwise("High")
    )

    return cleaned


def clean_postings(postings):
    original_count = postings.count()
    cleaned = select_needed_posting_columns(postings)

    before_dedup_count = cleaned.count()
    cleaned = cleaned.dropDuplicates(["job_id"])

    after_dedup_count = cleaned.count()
    cleaned = clean_text_columns(cleaned)
    cleaned = cast_numeric_columns(cleaned)
    cleaned = clean_remote_column(cleaned)
    cleaned = create_salary_value(cleaned)

    p33, p66 = calculate_salary_thresholds(cleaned)

    cleaned = create_salary_level(cleaned, p33, p66)

    cleaned = cleaned.withColumn(
        "work_type_final",
        F.when(F.col("formatted_work_type").isNotNull() & (F.col("formatted_work_type") != ""), F.col("formatted_work_type"))
        .when(F.col("work_type").isNotNull() & (F.col("work_type") != ""), F.col("work_type"))
        .otherwise("Unknown")
    )
    cleaned = cleaned.withColumn(
        "experience_level_final",
        F.when(
            F.col("formatted_experience_level").isNull() | (F.col("formatted_experience_level") == ""),
            "Unknown"
        ).otherwise(F.col("formatted_experience_level"))
    )
    processing_stats = {
        "original_postings_count": original_count,
        "before_dedup_count": before_dedup_count,
        "after_dedup_count": after_dedup_count,
        "removed_duplicates": before_dedup_count - after_dedup_count,
        "salary_p33": p33,
        "salary_p66": p66
    }

    return cleaned, processing_stats


def clean_job_skills(job_skills):
    cleaned = (
        job_skills
        .select(
            F.col("job_id").alias("skill_job_id"),
            F.trim(F.col("skill_abr").cast("string")).alias("skill_abr")
        )
        .where(F.col("skill_job_id").isNotNull())
        .where(F.col("skill_abr").isNotNull() & (F.col("skill_abr") != ""))
        .dropDuplicates(["skill_job_id", "skill_abr"])
    )

    return cleaned


def clean_skills(skills):
    cleaned = (
        skills
        .select(
            F.trim(F.col("skill_abr").cast("string")).alias("map_skill_abr"),
            F.trim(F.col("skill_name").cast("string")).alias("skill_name")
        )
        .where(F.col("map_skill_abr").isNotNull() & (F.col("map_skill_abr") != ""))
        .where(F.col("skill_name").isNotNull() & (F.col("skill_name") != ""))
        .dropDuplicates(["map_skill_abr"])
    )

    return cleaned


def create_jobs_with_skills(cleaned_postings, cleaned_job_skills, cleaned_skills):
    jobs_with_skills = (
        cleaned_postings
        .join(
            cleaned_job_skills,
            cleaned_postings["job_id"] == cleaned_job_skills["skill_job_id"],
            "inner"
        )
        .join(
            cleaned_skills,
            cleaned_job_skills["skill_abr"] == cleaned_skills["map_skill_abr"],
            "left"
        )
        .select(
            cleaned_postings["job_id"],
            cleaned_postings["title"],
            cleaned_postings["company_name"],
            cleaned_postings["location"],
            cleaned_job_skills["skill_abr"],
            cleaned_skills["skill_name"],
            cleaned_postings["salary_value"],
            cleaned_postings["normalized_salary"],
            cleaned_postings["salary_level"],
            cleaned_postings["views"],
            cleaned_postings["applies"],
            cleaned_postings["work_type_final"].alias("work_type"),
            cleaned_postings["remote_status"],
            cleaned_postings["experience_level_final"].alias("experience_level"),
            cleaned_postings["pay_period"],
            cleaned_postings["currency"]
        )
    )

    return jobs_with_skills


def write_processed_data(cleaned_postings, jobs_with_skills):
    cleaned_postings_file = PROCESSED_DIR / "cleaned_postings.parquet"
    jobs_with_skills_file = PROCESSED_DIR / "jobs_with_skills.parquet"

    if cleaned_postings_file.exists():
        if cleaned_postings_file.is_dir():
            shutil.rmtree(cleaned_postings_file)
        else:
            cleaned_postings_file.unlink()

    if jobs_with_skills_file.exists():
        if jobs_with_skills_file.is_dir():
            shutil.rmtree(jobs_with_skills_file)
        else:
            jobs_with_skills_file.unlink()

    cleaned_postings_pd = cleaned_postings.toPandas()
    jobs_with_skills_pd = jobs_with_skills.toPandas()

    cleaned_postings_pd.to_parquet(
        cleaned_postings_file,
        index=False,
        engine="pyarrow"
    )

    jobs_with_skills_pd.to_parquet(
        jobs_with_skills_file,
        index=False,
        engine="pyarrow"
    )
def generate_summary(cleaned_postings, jobs_with_skills, processing_stats):
    cleaned_count = cleaned_postings.count()
    jobs_with_skills_count = jobs_with_skills.count()

    salary_level_counts = (
        cleaned_postings
        .groupBy("salary_level")
        .count()
        .orderBy("salary_level")
        .collect()
    )

    remote_counts = (
        cleaned_postings
        .groupBy("remote_status")
        .count()
        .orderBy("remote_status")
        .collect()
    )

    work_type_counts = (
        cleaned_postings
        .groupBy("work_type_final")
        .count()
        .orderBy(F.desc("count"))
        .limit(10)
        .collect()
    )

    skill_count = jobs_with_skills.select("skill_name").where(F.col("skill_name").isNotNull()).distinct().count()
    jobs_with_at_least_one_skill = jobs_with_skills.select("job_id").distinct().count()

    lines = []
    lines.append("PHASE 6 - PYSPARK PROCESSING SUMMARY")
    lines.append("=" * 50)

    lines.append("\n1. POSTINGS CLEANING")
    lines.append(f"Original postings count: {processing_stats['original_postings_count']:,}")
    lines.append(f"Postings before duplicate removal: {processing_stats['before_dedup_count']:,}")
    lines.append(f"Postings after duplicate removal: {processing_stats['after_dedup_count']:,}")
    lines.append(f"Removed duplicate jobs: {processing_stats['removed_duplicates']:,}")
    lines.append(f"Final cleaned postings count: {cleaned_count:,}")

    lines.append("\n2. SALARY THRESHOLDS")
    lines.append(f"Salary percentile 33: {processing_stats['salary_p33']}")
    lines.append(f"Salary percentile 66: {processing_stats['salary_p66']}")

    lines.append("\n3. SALARY LEVEL DISTRIBUTION")
    for row in salary_level_counts:
        lines.append(f"{row['salary_level']}: {row['count']:,}")

    lines.append("\n4. REMOTE STATUS DISTRIBUTION")
    for row in remote_counts:
        lines.append(f"{row['remote_status']}: {row['count']:,}")

    lines.append("\n5. TOP WORK TYPES")
    for row in work_type_counts:
        lines.append(f"{row['work_type_final']}: {row['count']:,}")

    lines.append("\n6. JOB-SKILL JOIN OUTPUT")
    lines.append(f"Jobs with skills table rows: {jobs_with_skills_count:,}")
    lines.append(f"Jobs with at least one mapped skill: {jobs_with_at_least_one_skill:,}")
    lines.append(f"Distinct mapped skills: {skill_count:,}")

    lines.append("\n7. OUTPUT FILES")
    lines.append(f"cleaned_postings.parquet: {CLEANED_POSTINGS_PATH}")
    lines.append(f"jobs_with_skills.parquet: {JOBS_WITH_SKILLS_PATH}")

    lines.append("\nPySpark processing completed successfully.")

    summary_text = "\n".join(lines)
    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")
    print(summary_text)


def main():
    spark = create_spark_session()

    try:
        postings, job_skills, skills = read_raw_data(spark)

        cleaned_postings, processing_stats = clean_postings(postings)
        cleaned_job_skills = clean_job_skills(job_skills)
        cleaned_skills = clean_skills(skills)

        jobs_with_skills = create_jobs_with_skills(
            cleaned_postings,
            cleaned_job_skills,
            cleaned_skills
        )

        write_processed_data(cleaned_postings, jobs_with_skills)

        generate_summary(cleaned_postings, jobs_with_skills, processing_stats)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()