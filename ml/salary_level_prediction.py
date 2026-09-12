from pathlib import Path
import os
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, CountVectorizer, VectorAssembler, IndexToString
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "output"
ML_OUTPUT_DIR = OUTPUT_DIR / "ml"

CLEANED_POSTINGS_PATH = PROCESSED_DIR / "cleaned_postings.parquet"
JOBS_WITH_SKILLS_PATH = PROCESSED_DIR / "jobs_with_skills.parquet"

ML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SALARY_MIN = 10000
SALARY_MAX = 500000


def create_spark_session():
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
    os.environ["PYSPARK_PYTHON"] = "python"

    spark = (
        SparkSession.builder
        .appName("SalaryLevelPrediction")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.blockManager.port", "10025")
        .config("spark.driver.port", "10026")
        .config("spark.ui.port", "4041")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_processed_data():
    cleaned_postings = pd.read_parquet(CLEANED_POSTINGS_PATH)
    jobs_with_skills = pd.read_parquet(JOBS_WITH_SKILLS_PATH)

    return cleaned_postings, jobs_with_skills


def aggregate_skills(jobs_with_skills):
    skill_data = jobs_with_skills[["job_id", "skill_name"]].copy()
    skill_data = skill_data.dropna()
    skill_data["skill_name"] = skill_data["skill_name"].astype(str).str.strip()
    skill_data = skill_data[skill_data["skill_name"] != ""]

    aggregated = (
        skill_data
        .groupby("job_id")["skill_name"]
        .apply(lambda values: sorted(list(set(values))))
        .reset_index()
        .rename(columns={"skill_name": "skills_array"})
    )

    return aggregated


def prepare_ml_dataframe(cleaned_postings, skills_aggregated):
    columns = [
        "job_id",
        "title",
        "location",
        "work_type_final",
        "remote_status",
        "views",
        "applies",
        "experience_level_final",
        "salary_value",
        "salary_level"
    ]

    data = cleaned_postings[columns].copy()

    data["salary_value"] = pd.to_numeric(data["salary_value"], errors="coerce")
    data["views"] = pd.to_numeric(data["views"], errors="coerce").fillna(0)
    data["applies"] = pd.to_numeric(data["applies"], errors="coerce").fillna(0)

    data = data[
        (data["salary_value"].notna()) &
        (data["salary_value"] >= SALARY_MIN) &
        (data["salary_value"] <= SALARY_MAX)
    ]

    data = data[data["salary_level"].isin(["Low", "Medium", "High"])]

    data = data.merge(skills_aggregated, on="job_id", how="left")

    data["skills_array"] = data["skills_array"].apply(
        lambda value: value if isinstance(value, list) and len(value) > 0 else ["Unknown Skill"]
    )

    text_columns = [
        "title",
        "location",
        "work_type_final",
        "remote_status",
        "experience_level_final"
    ]

    for column in text_columns:
        data[column] = data[column].fillna("Unknown").astype(str).str.strip()
        data[column] = data[column].replace("", "Unknown")

    top_titles = data["title"].value_counts().head(50).index
    top_locations = data["location"].value_counts().head(50).index

    data["title_group"] = data["title"].apply(lambda value: value if value in top_titles else "Other Title")
    data["location_group"] = data["location"].apply(lambda value: value if value in top_locations else "Other Location")

    final_columns = [
        "job_id",
        "title",
        "title_group",
        "location",
        "location_group",
        "work_type_final",
        "remote_status",
        "views",
        "applies",
        "experience_level_final",
        "skills_array",
        "salary_value",
        "salary_level"
    ]

    return data[final_columns]


def create_spark_dataframe(spark, ml_data):
    return spark.createDataFrame(ml_data)


def split_data(df):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    return train_df, test_df


def build_pipeline():
    label_indexer = StringIndexer(
        inputCol="salary_level",
        outputCol="label",
        handleInvalid="keep"
    )

    categorical_input_cols = [
        "title_group",
        "location_group",
        "work_type_final",
        "remote_status",
        "experience_level_final"
    ]

    categorical_index_cols = [
        "title_index",
        "location_index",
        "work_type_index",
        "remote_index",
        "experience_index"
    ]

    categorical_encoded_cols = [
        "title_vec",
        "location_vec",
        "work_type_vec",
        "remote_vec",
        "experience_vec"
    ]

    categorical_indexer = StringIndexer(
        inputCols=categorical_input_cols,
        outputCols=categorical_index_cols,
        handleInvalid="keep"
    )

    encoder = OneHotEncoder(
        inputCols=categorical_index_cols,
        outputCols=categorical_encoded_cols
    )

    skills_vectorizer = CountVectorizer(
        inputCol="skills_array",
        outputCol="skills_vec",
        minDF=1.0
    )

    assembler = VectorAssembler(
        inputCols=[
            "views",
            "applies",
            "title_vec",
            "location_vec",
            "work_type_vec",
            "remote_vec",
            "experience_vec",
            "skills_vec"
        ],
        outputCol="features"
    )

    classifier = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        numTrees=50,
        maxDepth=8,
        seed=42
    )

    pipeline = Pipeline(
        stages=[
            label_indexer,
            categorical_indexer,
            encoder,
            skills_vectorizer,
            assembler,
            classifier
        ]
    )

    return pipeline


def train_model(pipeline, train_df):
    model = pipeline.fit(train_df)
    return model


def evaluate_model(model, test_df):
    predictions = model.transform(test_df)

    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    accuracy = accuracy_evaluator.evaluate(predictions)
    f1_score = f1_evaluator.evaluate(predictions)

    label_model = model.stages[0]
    labels = label_model.labels

    converter = IndexToString(
        inputCol="prediction",
        outputCol="predicted_salary_level",
        labels=labels
    )

    predictions = converter.transform(predictions)

    actual_converter = IndexToString(
        inputCol="label",
        outputCol="actual_salary_level",
        labels=labels
    )

    predictions = actual_converter.transform(predictions)

    return predictions, accuracy, f1_score, labels


def create_confusion_matrix(predictions):
    confusion = (
        predictions
        .groupBy("actual_salary_level", "predicted_salary_level")
        .count()
        .orderBy("actual_salary_level", "predicted_salary_level")
    )

    confusion_pd = confusion.toPandas()
    return confusion_pd


def create_sample_predictions(predictions):
    sample_columns = [
        "job_id",
        "title",
        "location",
        "work_type_final",
        "remote_status",
        "views",
        "applies",
        "experience_level_final",
        "salary_value",
        "actual_salary_level",
        "predicted_salary_level"
    ]

    sample = (
        predictions
        .select(*sample_columns)
        .limit(30)
        .toPandas()
    )

    return sample


def get_label_distribution(df, column_name):
    result = (
        df
        .groupBy(column_name)
        .count()
        .orderBy(column_name)
        .toPandas()
    )

    return result


def save_outputs(summary_text, confusion_matrix, sample_predictions, train_distribution, test_distribution):
    summary_path = ML_OUTPUT_DIR / "ml_summary.txt"
    confusion_path = ML_OUTPUT_DIR / "confusion_matrix.csv"
    sample_path = ML_OUTPUT_DIR / "sample_predictions.csv"
    train_dist_path = ML_OUTPUT_DIR / "train_salary_level_distribution.csv"
    test_dist_path = ML_OUTPUT_DIR / "test_salary_level_distribution.csv"

    summary_path.write_text(summary_text, encoding="utf-8")

    confusion_matrix.to_csv(confusion_path, index=False, encoding="utf-8-sig")
    sample_predictions.to_csv(sample_path, index=False, encoding="utf-8-sig")
    train_distribution.to_csv(train_dist_path, index=False, encoding="utf-8-sig")
    test_distribution.to_csv(test_dist_path, index=False, encoding="utf-8-sig")


def generate_summary(
    ml_data_count,
    train_count,
    test_count,
    accuracy,
    f1_score,
    labels,
    confusion_matrix,
    train_distribution,
    test_distribution
):
    lines = []

    lines.append("PHASE 9 - MACHINE LEARNING SUMMARY")
    lines.append("=" * 50)

    lines.append("\n1. MODEL OBJECTIVE")
    lines.append("Target variable: salary_level")
    lines.append("Classes: Low, Medium, High")
    lines.append("Algorithm: Random Forest Classifier")
    lines.append("Framework: Spark MLlib")

    lines.append("\n2. DATA USED FOR TRAINING")
    lines.append(f"Salary records used after filtering: {ml_data_count:,}")
    lines.append(f"Training records: {train_count:,}")
    lines.append(f"Testing records: {test_count:,}")
    lines.append(f"Accepted salary range: {SALARY_MIN:,} to {SALARY_MAX:,}")

    lines.append("\n3. FEATURES USED")
    lines.append("title_group")
    lines.append("location_group")
    lines.append("work_type_final")
    lines.append("remote_status")
    lines.append("views")
    lines.append("applies")
    lines.append("experience_level_final")
    lines.append("skills_array")

    lines.append("\n4. LABEL ORDER USED BY MODEL")
    for index, label in enumerate(labels):
        lines.append(f"{index}: {label}")

    lines.append("\n5. MODEL PERFORMANCE")
    lines.append(f"Accuracy: {accuracy:.4f}")
    lines.append(f"F1-score: {f1_score:.4f}")

    lines.append("\n6. TRAINING SALARY LEVEL DISTRIBUTION")
    for _, row in train_distribution.iterrows():
        lines.append(f"{row['salary_level']}: {int(row['count']):,}")

    lines.append("\n7. TESTING SALARY LEVEL DISTRIBUTION")
    for _, row in test_distribution.iterrows():
        lines.append(f"{row['salary_level']}: {int(row['count']):,}")

    lines.append("\n8. CONFUSION MATRIX")
    for _, row in confusion_matrix.iterrows():
        lines.append(
            f"Actual {row['actual_salary_level']} predicted as "
            f"{row['predicted_salary_level']}: {int(row['count']):,}"
        )

    lines.append("\nMachine learning stage completed successfully.")

    return "\n".join(lines)


def main():
    spark = create_spark_session()

    try:
        cleaned_postings, jobs_with_skills = load_processed_data()
        skills_aggregated = aggregate_skills(jobs_with_skills)
        ml_data = prepare_ml_dataframe(cleaned_postings, skills_aggregated)

        spark_df = create_spark_dataframe(spark, ml_data)

        train_df, test_df = split_data(spark_df)

        train_count = train_df.count()
        test_count = test_df.count()

        pipeline = build_pipeline()
        model = train_model(pipeline, train_df)

        predictions, accuracy, f1_score, labels = evaluate_model(model, test_df)

        confusion_matrix = create_confusion_matrix(predictions)
        sample_predictions = create_sample_predictions(predictions)

        train_distribution = get_label_distribution(train_df, "salary_level")
        test_distribution = get_label_distribution(test_df, "salary_level")

        summary_text = generate_summary(
            ml_data_count=len(ml_data),
            train_count=train_count,
            test_count=test_count,
            accuracy=accuracy,
            f1_score=f1_score,
            labels=labels,
            confusion_matrix=confusion_matrix,
            train_distribution=train_distribution,
            test_distribution=test_distribution
        )

        save_outputs(
            summary_text,
            confusion_matrix,
            sample_predictions,
            train_distribution,
            test_distribution
        )

        print(summary_text)
        print(f"\nML outputs saved in: {ML_OUTPUT_DIR}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()