"""Smoke test: verifies Gemini + BigQuery access before we build anything.

Run from the repo root:  python scripts/smoke_test.py
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

OK, FAIL = "[ OK ]", "[FAIL]"
failures = 0


def check(name, fn):
    global failures
    try:
        detail = fn()
        print(f"{OK} {name}" + (f" -> {detail}" if detail else ""))
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"{FAIL} {name}\n       {type(e).__name__}: {e}")


# ---------- Gemini ----------
def gemini_models():
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions and "gemini" in m.name:
            names.append(m.name.replace("models/", ""))
    print("       Models your key can use for generateContent:")
    for n in sorted(names):
        print(f"         - {n}")
    return f"{len(names)} gemini models"


def gemini_generate():
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.getenv("GEMINI_MODEL_FAST", "gemini-2.5-flash")
    resp = client.models.generate_content(
        model=model, contents="Reply with exactly one word: pong"
    )
    return f"{model} says {resp.text.strip()!r}"


def gemini_embed():
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # Embedding model name may change; check the models list if this fails.
    model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    resp = client.models.embed_content(model=model, contents="hello world")
    return f"{model} dim={len(resp.embeddings[0].values)}"


# ---------- BigQuery ----------
DATASET = os.getenv("BQ_DATASET", "bigquery-public-data.thelook_ecommerce")
TABLES = ["orders", "order_items", "products", "users"]


def bq_client():
    from google.cloud import bigquery

    return bigquery.Client(project=os.getenv("GCP_PROJECT_ID") or None)


def bq_query():
    from google.cloud import bigquery

    client = bq_client()
    sql = f"SELECT COUNT(*) AS n FROM `{DATASET}.users`"
    dry = client.query(
        sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    )
    rows = list(client.query(sql).result())
    return f"project={client.project}, dry-run bytes={dry.total_bytes_processed}, users={rows[0].n}"


def bq_schemas():
    client = bq_client()
    for t in TABLES:
        table = client.get_table(f"{DATASET}.{t}")
        cols = ", ".join(f"{f.name}:{f.field_type}" for f in table.schema)
        print(f"       {t} ({table.num_rows} rows): {cols}")
    return None


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print(f"{FAIL} GEMINI_API_KEY missing. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    print("== Gemini ==")
    check("list models", gemini_models)
    check("generate_content", gemini_generate)
    check("embed_content", gemini_embed)

    print("\n== BigQuery ==")
    check("auth + simple query", bq_query)
    check("table schemas", bq_schemas)

    print()
    if failures:
        print(f"{failures} check(s) failed. Paste the output to Claude and we'll fix it.")
        sys.exit(1)
    print("All good. Ready to build.")
