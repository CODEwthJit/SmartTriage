"""
SmartTriage Semantic Vector Search & Duplicate Detection Engine.
Indexes historical issues using dense Transformer embeddings (all-MiniLM-L6-v2),
computes normalized Cosine Similarity, and identifies duplicate bug reports in real time.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
MODEL_REGISTRY_DIR = PROJECT_ROOT / "models" / "registry"
INDEX_FILE = MODEL_REGISTRY_DIR / "vector_index.joblib"


class DuplicateSearchEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.embeddings = None  # Shape: (N, 384)
        self.metadata = None  # List of issue dicts

    def build_index(self, data_file: Path = TRAIN_FILE) -> int:
        """Encodes all historical training issues and builds an L2-normalized vector matrix."""
        print(f"Building vector index from: {data_file}")
        df = pd.read_csv(data_file)

        # Store searchable issue metadata
        self.metadata = df[["issue_id", "title", "body", "category", "priority", "duplicate_group_id"]].to_dict(
            orient="records"
        )

        # Combine title and body for rich semantic representation
        texts = [f"{row['title']} {row['body']}" for row in self.metadata]

        print(f"Encoding {len(texts)} issues into 384-dimensional dense vectors...")
        raw_embeddings = self.encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalization for instant cosine dot-products
        )

        self.embeddings = np.array(raw_embeddings, dtype=np.float32)
        print(f"Index built successfully. Matrix shape: {self.embeddings.shape}")
        return len(self.metadata)

    def query(self, title: str, body: str = "", top_k: int = 3, threshold: float = 0.70) -> list[dict]:
        """
        Embeds an incoming issue and calculates dot-product cosine similarity
        against all indexed issues in sub-millisecond time.
        """
        if self.embeddings is None:
            raise ValueError("Index is not loaded. Call build_index() or load() first.")

        query_text = f"{title} {body}".strip()
        query_vec = self.encoder.encode([query_text], normalize_embeddings=True)[0]  # Shape: (384,)

        # Dot product with L2-normalized vectors equals exact Cosine Similarity:
        # similarities = query_vec @ embeddings.T -> Shape: (N,)
        similarities = np.dot(self.embeddings, query_vec)

        # Retrieve top-K indices sorted in descending order
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            issue = self.metadata[idx]
            results.append(
                {
                    "issue_id": int(issue["issue_id"]),
                    "title": issue["title"],
                    "category": issue["category"],
                    "priority": issue["priority"],
                    "similarity_score": round(score, 4),
                    "is_duplicate_warning": bool(score >= threshold),
                    "duplicate_group_id": int(issue["duplicate_group_id"]),
                }
            )

        return results

    def save(self, output_path: Path = INDEX_FILE):
        """Serializes precomputed embeddings and metadata for zero-overhead inference."""
        MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"model_name": self.model_name, "embeddings": self.embeddings, "metadata": self.metadata}
        joblib.dump(payload, output_path)
        print(f"Saved vector index to: {output_path}")

    def load(self, index_path: Path = INDEX_FILE):
        """Loads serialized embeddings and metadata from disk."""
        payload = joblib.load(index_path)
        self.model_name = payload["model_name"]
        self.embeddings = payload["embeddings"]
        self.metadata = payload["metadata"]
        print(f"Loaded vector index with {len(self.metadata)} items from: {index_path}")


def main():
    engine = DuplicateSearchEngine()
    engine.build_index(TRAIN_FILE)
    engine.save(INDEX_FILE)

    # Test Query 1: Variant of NPE checkout bug
    test_query_1 = "Cart checkout crashes with NullPointerException when basket is empty"
    print("\n--- Testing Semantic Search Query 1 ---")
    print(f'Query: "{test_query_1}"')
    matches = engine.query(title=test_query_1, top_k=3, threshold=0.70)
    for i, m in enumerate(matches, 1):
        print(
            f"  Top {i}: [{m['similarity_score']:.4f}] Issue #{m['issue_id']} - {m['title']} (Dup Warning: {m['is_duplicate_warning']})"
        )

    # Test Query 2: Variant of SQL Injection security vulnerability
    test_query_2 = "SQL injection vulnerability discovered in search filter parameter"
    print("\n--- Testing Semantic Search Query 2 ---")
    print(f'Query: "{test_query_2}"')
    matches = engine.query(title=test_query_2, top_k=3, threshold=0.70)
    for i, m in enumerate(matches, 1):
        print(
            f"  Top {i}: [{m['similarity_score']:.4f}] Issue #{m['issue_id']} - {m['title']} (Dup Warning: {m['is_duplicate_warning']})"
        )


if __name__ == "__main__":
    main()
