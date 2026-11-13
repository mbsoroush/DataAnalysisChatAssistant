# Assumptions & Design Decisions

The full design is in [ARCHITECTURE.md](ARCHITECTURE.md); this file records the assumptions it rests on.

The project specification leaves several points open and invites assumptions. Each is stated here with the reasoning, so it can be challenged or changed without redesigning the system.

## 1. Dataset facts (verified against `bigquery-public-data.thelook_ecommerce`)

| Table | Rows | Key columns |
|---|---|---|
| `users` | 100,000 | id, first_name, last_name, email, age, gender, state, street_address, postal_code, city, country, latitude, longitude, user_geom, traffic_source, created_at |
| `orders` | 124,452 | order_id, user_id, status, gender, created_at, shipped_at, delivered_at, returned_at, num_of_item |
| `order_items` | 180,746 | id, order_id, user_id, product_id, status, sale_price, created_at, shipped_at, delivered_at, returned_at |
| `products` | 29,120 | id, name, brand, category, department, cost, retail_price, sku |

Consequences:
- The data has **no concept of "churn"**, "revenue" or "underspending". These need explicit definitions in a versioned semantic layer (`config/semantic_layer.yaml`) so the LLM does not invent them per question. Example: revenue = `SUM(order_items.sale_price)` excluding Cancelled/Returned items; churned customer = bought in period N-1 and not in period N (window configurable).
- The product-to-user link is `order_items`. Product attributes (brand, category, department) live only in `products`.

## 2. "Each user can only analyze data on products related to him"

thelook has no ownership model, so access is defined by configuration:

- `config/user_access.yaml` maps `user_id -> {brands: [...], departments: [...]}`. In production this lives in the identity/IAM layer and arrives as signed claims; the agent never trusts a scope supplied through chat.
- **Enforcement is in SQL, not in the prompt.** The validator parses generated SQL with `sqlglot` and replaces every reference to a base table with a scoped subquery under the same alias:
  - `products` -> products matching the user's brands/departments
  - `order_items` -> items whose `product_id` is in the scoped products
  - `orders` -> orders that contain at least one scoped item
  - `users` -> customers that bought at least one scoped item
- Known limitation (documented, not hidden): `orders.num_of_item` counts the whole order, including out-of-scope items. The semantic layer directs item counts to scoped `order_items` instead.

## 3. PII policy

| Class | Columns | Rule |
|---|---|---|
| Direct identifiers | first_name, last_name, email, street_address, postal_code | Blocked: query validator rejects them; never returned to the LLM or user |
| Precise location | latitude, longitude, user_geom | Blocked |
| Quasi-identifiers | city, age | Allowed only inside aggregates (GROUP BY); small groups suppressed |
| Safe dimensions | state, country, gender, traffic_source, dates | Allowed |
| Pseudonymous keys | users.id / user_id | Shown only as a salted-hash token (e.g. `cust_9f2a1c`), so "top customers" works without exposing identity |

Defense in depth: (1) schema exposed to the LLM omits blocked columns, (2) SQL validator rejects them, (3) result-frame masking before the LLM sees data, (4) output scanner on the final text (emails and phone-number patterns; the production design adds Cloud Sensitive Data Protection for names and addresses).

## 4. Golden Bucket ("Trios")

Marked theoretical in the specification, so this is a **design-only** item: the prototype does not read a golden bucket. The assumed schema is:

```json
{"id": "...", "question": "...", "sql": "...", "report": "...",
 "tags": ["revenue", "monthly"], "tables": ["order_items", "products"],
 "quality_score": 0.0, "source": "analyst|agent_promoted", "created_at": "...", "schema_version": 1}
```

- Retrieval: embed the question, top-k by cosine similarity, filtered by tags/tables and `quality_score`, injected as few-shot examples. Retrieved SQL is treated as **examples, not truth**: it still goes through validation and scoping.
- Update loop (production design): interactions rated positively or accepted by an analyst enter a curation queue, are deduplicated by embedding similarity, reviewed, then promoted with a quality score.

## 5. Scale and platform

- Platform: GCP (BigQuery is mandated; Gemini preferred).
- Load: ~100-200 executive users, ~2-5k queries/day, ~10 concurrent at peak. Cost control matters more than throughput: per-query byte cap via BigQuery dry-run, capped self-correction retries, caching.

## 6. Saved reports and destructive operations

- Reports are per-user; a user can only list, search and delete their own.
- Report record: `report_id, owner_id, conversation_id, title, body, mentioned_entities, tags, created_at, deleted_at`.
- Deletion is **two-phase**: (1) resolve matches for requests like "delete all reports mentioning Client X" or "delete the reports from this conversation" and show an exact preview; (2) the user confirms in natural language ("yes", "go ahead"); only then execute. Execution re-checks ownership and the confirmed id set, so the set cannot grow between preview and confirm.
- Soft delete plus an audit log entry, so an accidental confirmation is recoverable.

## 7. LLM

- Gemini through `google-genai`, behind a thin client (`src/llm.py`) providing retries with backoff, an ordered model-fallback chain, a shared per-model circuit breaker, per-call timeouts and a total deadline, structured output and token telemetry.
- Observed in practice: a 503 "high demand" on the first smoke test, which is the reason for the fallback design.
- Model names are configuration. Primary and fallbacks are chosen from the models the API key can access (`scripts/probe_models.py`).

## 8. Out of scope for the prototype (covered in the design document)

Email delivery, web search, multi-tenant IAM integration, the real Golden Bucket pipeline, and Vertex AI Vector Search. None of these are implemented in the prototype; `docs/ARCHITECTURE.md` explains how each would plug in (tool registry, confirmation gate for side effects).
