"""
SmartTriage Data Cleaning & Stratification Pipeline.
Responsible for text normalization, missing-value sanitization,
stratified train/val/test partitioning, and persisting processed datasets.
"""

from pathlib import Path
import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Define file paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "raw_issues.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_FILE = PROCESSED_DATA_DIR / "val.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
METADATA_FILE = PROCESSED_DATA_DIR / "metadata.json"


def normalize_text(text: str) -> str:
    """
    Cleans raw markdown and technical issue text while preserving
    critical semantic identifiers like error codes, URLs, and class names.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Convert markdown links [text](http...) to just 'text'
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # 2. Replace raw URLs with a generic <URL> token to prevent overfitting to specific domains
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)

    # 3. Strip backticks from inline code blocks while preserving the code tokens
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # 4. Remove triple-backtick markdown blocks entirely but keep their contents
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```", "", text)

    # 5. Normalize excessive whitespace, tabs, and newlines to a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_and_split_data(
    input_file: Path = RAW_DATA_FILE,
    output_dir: Path = PROCESSED_DATA_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict:
    """
    Loads raw issues, cleans text, performs stratified splitting on the target category,
    and saves processed datasets and metadata.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw dataset from: {input_file}")
    df = pd.read_csv(input_file)

    # Sanitize missing values
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    # Apply text normalization
    print("Normalizing issue titles and bodies...")
    df["clean_title"] = df["title"].apply(normalize_text)
    df["clean_body"] = df["body"].apply(normalize_text)

    # Construct unified composite input: title has high density, body provides context
    df["text"] = df["clean_title"] + " " + df["clean_body"]

    # Compute text length metrics for validation
    df["token_count"] = df["text"].apply(lambda t: len(t.split()))

    # Stratified Split Phase 1: Separate Train (70%) from Temporary (30% for Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(df, test_size=temp_ratio, stratify=df["category"], random_state=random_seed)

    # Stratified Split Phase 2: Split Temporary 50/50 into Validation (15%) and Test (15%)
    val_fraction_of_temp = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - val_fraction_of_temp), stratify=temp_df["category"], random_state=random_seed
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Persist processed partitions
    print(f"Saving train partition ({len(train_df)} rows) to: {TRAIN_FILE}")
    train_df.to_csv(TRAIN_FILE, index=False, encoding="utf-8")

    print(f"Saving val partition ({len(val_df)} rows) to: {VAL_FILE}")
    val_df.to_csv(VAL_FILE, index=False, encoding="utf-8")

    print(f"Saving test partition ({len(test_df)} rows) to: {TEST_FILE}")
    test_df.to_csv(TEST_FILE, index=False, encoding="utf-8")

    # Generate metadata contract for downstream models
    category_labels = sorted(df["category"].unique().tolist())
    priority_labels = sorted(df["priority"].unique().tolist())

    metadata = {
        "dataset_name": "SmartTriage-Raw-Issues",
        "total_samples": len(df),
        "split_counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "category_to_id": {label: idx for idx, label in enumerate(category_labels)},
        "priority_to_id": {label: idx for idx, label in enumerate(priority_labels)},
        "avg_token_count": round(float(df["token_count"].mean()), 2),
        "max_token_count": int(df["token_count"].max()),
        "min_token_count": int(df["token_count"].min()),
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved dataset metadata contract to: {METADATA_FILE}")
    return metadata


def main():
    print("--- STARTING DATA CLEANING & STRATIFICATION ---")
    metadata = clean_and_split_data()
    print("\n--- PIPELINE EXECUTION COMPLETED ---")
    print(f"Total Rows Processed: {metadata['total_samples']}")
    print(
        f"Splits -> Train: {metadata['split_counts']['train']}, Val: {metadata['split_counts']['val']}, Test: {metadata['split_counts']['test']}"
    )
    print(f"Target Categories: {list(metadata['category_to_id'].keys())}")
    print(f"Target Priorities: {list(metadata['priority_to_id'].keys())}")
    print(f"Average Issue Word Count: {metadata['avg_token_count']} words")


if __name__ == "__main__":
    main()
