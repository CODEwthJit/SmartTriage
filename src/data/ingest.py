"""
SmartTriage Data Ingestion Module.
Responsible for acquiring raw issue data, establishing schema contracts,
and persisting immutable raw datasets to data/raw/raw_issues.csv.
"""

from pathlib import Path
import random
import pandas as pd

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "raw_issues.csv"

# Pre-defined semantic templates representing real-world engineering issues
ISSUE_TEMPLATES = {
    "bug": [
        (
            "NullPointerException on user checkout",
            "When calling POST /api/v1/checkout with an empty cart item list, server throws NullPointerException at CheckoutService.java:142. Expected HTTP 400 Bad Request.",
            "P0-Critical",
        ),
        (
            "Database connection pool exhausted under high concurrency",
            "HikariCP pool drops active connections after 500 concurrent requests. Error: Connection is not available, request timed out after 30000ms.",
            "P1-High",
        ),
        (
            "CSS overflow issue on mobile navigation drawer",
            "On iOS Safari viewport < 375px, the slide-over menu items clip outside the right screen border.",
            "P3-Low",
        ),
        (
            "Memory leak in background token refresh worker",
            "Process RSS memory grows continuously by ~50MB/hour. Profiling indicates unclosed TCP sessions in AuthService.",
            "P1-High",
        ),
        (
            "Broken hyperlink in settings footer",
            "Clicking 'Terms of Service' in user settings redirects to 404 page.",
            "P3-Low",
        ),
        (
            "UnicodeDecodeError when parsing multibyte UTF-8 input",
            "The JSON parser crashes on emoji input with UnicodeDecodeError: 'utf-8' codec can't decode byte 0xed in position 12.",
            "P2-Medium",
        ),
        (
            "Deadlock in order fulfillment distributed lock",
            "Redis distributed lock fails to release when worker process receives SIGTERM during checkout transaction.",
            "P0-Critical",
        ),
        (
            "Checkbox state does not persist across tab switching",
            "Form input resets when toggling between 'Account' and 'Billing' sub-tabs.",
            "P2-Medium",
        ),
    ],
    "feature": [
        (
            "Add support for OAuth2 Google Login",
            "Users need the ability to authenticate via Google OAuth2 with PKCE flow. Needs configuration for client ID and secret.",
            "P2-Medium",
        ),
        (
            "Support CSV export for monthly billing reports",
            "Accounting teams require monthly transactions exported as standard RFC 4180 CSV files with tax breakdown.",
            "P2-Medium",
        ),
        (
            "Dark mode theme preference toggle",
            "Implement automatic dark mode switching based on CSS prefers-color-scheme media query.",
            "P3-Low",
        ),
        (
            "Add webhook notifications for payment success",
            "Developers need outbound HTTP webhooks triggered on invoice.paid events with signature verification header.",
            "P1-High",
        ),
        (
            "Implement bulk item deletion in admin console",
            "Allow administrators to select multiple user records and execute soft-delete in a single batch operation.",
            "P2-Medium",
        ),
    ],
    "documentation": [
        (
            "Update API documentation for v2 authentication header",
            "The current Swagger docs show 'Authorization: Token <key>', but v2 requires 'Authorization: Bearer <jwt>'.",
            "P2-Medium",
        ),
        (
            "Fix broken example code in Quickstart tutorial",
            "The quickstart snippet for initializing the client client.connect() causes TypeError: missing 1 required positional argument 'api_key'.",
            "P2-Medium",
        ),
        (
            "Add architecture diagram to README",
            "The repository needs an updated high-level diagram explaining the microservice event bus flow.",
            "P3-Low",
        ),
        (
            "Document Docker Compose local setup steps",
            "Add troubleshooting section for port conflicts on Windows WSL2 when starting PostgreSQL container.",
            "P3-Low",
        ),
    ],
    "performance": [
        (
            "Slow SQL query on user activity feed endpoint",
            "SELECT * FROM activities WHERE user_id = ? ORDER BY timestamp DESC takes 4.2 seconds on accounts with >50k events. Missing composite index.",
            "P1-High",
        ),
        (
            "High CPU utilization in regex router matching",
            "Regex matching in URL routing consumes 85% CPU during high request spikes due to catastrophic backtracking in greedy pattern.",
            "P1-High",
        ),
        (
            "Optimize Docker image layer caching to speed up CI",
            "Re-installing heavy Python packages on every single commit causes CI build times to exceed 18 minutes.",
            "P2-Medium",
        ),
        (
            "API gateway p99 latency spikes above 1200ms",
            "Downstream microservice connection timeouts lead to thread starvation in the upstream edge gateway.",
            "P1-High",
        ),
    ],
    "security": [
        (
            "SQL Injection vulnerability in search filter query",
            "The 'filter' parameter in /api/v1/search is concatenated directly into the SQL string without parameterized binding. Exploitable via ' OR 1=1 --",
            "P0-Critical",
        ),
        (
            "Missing CSRF token validation on password change endpoint",
            "POST /api/v1/user/password-reset accepts cross-origin state-changing requests without checking X-CSRF-Token.",
            "P0-Critical",
        ),
        (
            "Insecure direct object reference (IDOR) on document download",
            "Authenticated users can access arbitrary tenant invoices by altering the numeric invoice_id in the URL.",
            "P0-Critical",
        ),
        (
            "CORS policy allows wildcard origin with credentials",
            "Access-Control-Allow-Origin is set to '*' while Access-Control-Allow-Credentials is true, violating CORS specification.",
            "P1-High",
        ),
    ],
}

# Duplicate clusters for evaluating semantic duplicate detection
DUPLICATE_PAIRS = [
    {
        "group_id": 101,
        "category": "bug",
        "priority": "P0-Critical",
        "variants": [
            (
                "NullPointerException when cart checkout is empty",
                "CheckoutService.java:142 throws NPE when checkout items list is empty.",
            ),
            (
                "Cart checkout crashes with NPE on empty submission",
                "Submitting checkout without items causes 500 error due to NullPointerException in CheckoutService.",
            ),
            (
                "Empty cart checkout unhandled exception",
                "System fails on checkout with empty basket, throwing null pointer at line 142 of checkout service.",
            ),
        ],
    },
    {
        "group_id": 102,
        "category": "security",
        "priority": "P0-Critical",
        "variants": [
            (
                "SQL injection in search endpoint",
                "Raw query concatenation found in search filter parameter allows arbitrary database execution.",
            ),
            (
                "Unescaped SQL query in /search filter parameter",
                "Vulnerability report: Search filter parameter allows SQL injection due to lack of prepared statements.",
            ),
            (
                "SQLi vulnerability in search API",
                "Users can inject SQL statements via the filter parameter on the search route.",
            ),
        ],
    },
    {
        "group_id": 103,
        "category": "performance",
        "priority": "P1-High",
        "variants": [
            (
                "Slow query on activity feed",
                "Activity feed query takes over 4 seconds for active users because composite index is missing on (user_id, timestamp).",
            ),
            (
                "Activity feed latency is 4000ms+",
                "Users with large activity histories experience extreme delays loading feed due to unindexed query.",
            ),
            (
                "Database timeout loading activity stream",
                "Fetching user activities times out on accounts with >50k events. Index missing.",
            ),
        ],
    },
]


def generate_dataset(num_samples: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic, statistically sound dataset of software issues
    with natural class imbalance and semantic duplicate clusters.
    """
    random.seed(random_seed)
    records = []
    issue_id_counter = 1000

    # 1. Inject duplicate clusters first (ground truth for duplicate retrieval)
    for dup in DUPLICATE_PAIRS:
        for title, body in dup["variants"]:
            records.append(
                {
                    "issue_id": issue_id_counter,
                    "title": title,
                    "body": body,
                    "category": dup["category"],
                    "priority": dup["priority"],
                    "duplicate_group_id": dup["group_id"],
                    "author_association": random.choice(["CONTRIBUTOR", "MEMBER", "NONE"]),
                    "comments_count": random.randint(1, 15),
                }
            )
            issue_id_counter += 1

    # 2. Sample remaining issues with realistic industry class proportions:
    # Bugs ~45%, Features ~25%, Docs ~15%, Performance ~10%, Security ~5%
    category_weights = {"bug": 0.45, "feature": 0.25, "documentation": 0.15, "performance": 0.10, "security": 0.05}

    categories = list(category_weights.keys())
    weights = list(category_weights.values())

    remaining_count = num_samples - len(records)
    for _ in range(remaining_count):
        cat = random.choices(categories, weights=weights, k=1)[0]
        title_tmpl, body_tmpl, priority = random.choice(ISSUE_TEMPLATES[cat])

        # Add slight natural variation (e.g. environment prefixes, stack snippets)
        prefixes = ["", "[Prod] ", "[Bug Report] ", "[Feature Request] ", "Urgent: ", "Question: "]
        title = f"{random.choice(prefixes)}{title_tmpl}" if cat in ["bug", "security"] else title_tmpl

        # Add random comments count correlated with priority
        comments = random.randint(5, 30) if priority == "P0-Critical" else random.randint(0, 8)

        records.append(
            {
                "issue_id": issue_id_counter,
                "title": title,
                "body": body_tmpl,
                "category": cat,
                "priority": priority,
                "duplicate_group_id": -1,  # -1 indicates unique non-duplicate issue
                "author_association": random.choice(["NONE", "CONTRIBUTOR", "MEMBER", "FIRST_TIME_CONTRIBUTOR"]),
                "comments_count": comments,
            }
        )
        issue_id_counter += 1

    df = pd.DataFrame(records)
    # Shuffle dataset to prevent ordering bias
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df


def main():
    print(f"Creating raw data directory at: {RAW_DATA_DIR}")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating reproducible raw issues dataset (N=1500)...")
    df = generate_dataset(num_samples=1500, random_seed=42)

    print(f"Persisting immutable dataset to: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\n--- INGESTION SUMMARY ---")
    print(f"Total Rows: {len(df)}")
    print(f"Data Shape: {df.shape}")
    print("\nCategory Distribution:")
    print(df["category"].value_counts(normalize=True).round(3))
    print("\nPriority Distribution:")
    print(df["priority"].value_counts(normalize=True).round(3))
    print(f"\nDuplicate ground-truth clusters injected: {len(DUPLICATE_PAIRS)} groups.")
    print("Ingestion complete successfully.")


if __name__ == "__main__":
    main()
