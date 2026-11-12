# Retail Data Analysis Chat Assistant

A command-line chat agent that lets non-technical executives ask questions about sales, products and customers in plain
English. It writes and runs BigQuery SQL on `bigquery-public-data.thelook_ecommerce`, protects personal data, scopes every user to
their own products, manages a saved-reports library with a confirmation-gated delete, and recovers from model and warehouse failures.

* **Design and technical explanation:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (production HLD with diagrams, service choices,
  data flow, error handling, and an explanation of every requirement)
* **Assumptions:** [docs/assumptions.md](docs/assumptions.md)

## What the prototype implements

| Requirement | Status | How |
|---|---|---|
| Safety and PII masking | Built | SQL parsed with `sqlglot`; only single read-only statements; table and column allowlists; every table rewritten to a per-user scoped view; result masking and an output scanner |
| High-stakes oversight | Built | Saved reports; delete is two-phase (preview, then a plain "yes" parsed by code, never by the LLM); soft delete, audit log, "undo" for an hour |
| Resilience and graceful errors | Built | Self-correction loop capped at 3 attempts; BigQuery dry run and byte caps; LLM retries, timeouts, deadline, model fallback and circuit breaker |
| Observability | Built | One JSON event per step, keyed by a trace id; `python -m src.trace_report` for metrics and per-turn timelines |
| Persona management | Built | `config/persona.md` is re-read on every turn; edit it and the next answer changes, no restart |
| Golden Bucket, learning loop, QA harness | Design only | Explained in the architecture document, as the brief specifies |

## Quick start

Prerequisites: Python 3.11 or newer, a Google account, and the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).
Everything used here has a free tier; no credit card is needed for BigQuery sandbox mode.

**1. Get the code and create a virtual environment**

```bash
git clone https://github.com/mbsoroush/DataAnalysisChatAssistant.git
cd DataAnalysisChatAssistant
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Google Cloud access for BigQuery**

1. Create a project at [console.cloud.google.com](https://console.cloud.google.com) and enable the BigQuery API
   (`gcloud services enable bigquery.googleapis.com`).
2. Authenticate and select the project:

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

Queries against the public dataset run in your project and count against the free 1 TB per month.

**3. Gemini API key**

Create a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

**4. Configure**

```bash
cp .env.example .env                 # Windows: copy .env.example .env
```

Edit `.env` and replace the three `CHANGE_ME` values: `GEMINI_API_KEY`, `GCP_PROJECT_ID`, and `PII_SALT` (any long random string).
Never commit `.env`; it is git-ignored.

**5. Check that everything is connected**

```bash
python scripts/smoke_test.py         # Gemini generation and embeddings, BigQuery access, table schemas
python scripts/probe_models.py gemini-3.1-flash-lite gemini-3.5-flash gemini-flash-latest
```

Model names and free-tier limits change. The probe shows which models respond reliably for *your* key; put the good ones in
`GEMINI_MODEL_FAST`, `GEMINI_MODEL_STRONG` and `GEMINI_FALLBACKS` in `.env`.

**6. Run the tests** (no network or credentials needed)

```bash
pytest                               # 131 tests
```

**7. Chat**

```bash
python -m src.cli --user exec_brands --show-sql
```

If `python` or `pytest` resolves to another environment (for example conda's `base`), call the virtual environment explicitly:
`./.venv/bin/python -m pytest`, `./.venv/bin/python -m src.cli`.

## Using it

Commands inside the chat: `/help`, `/sql` (toggle showing the SQL the agent wrote and its self-correction feedback), `/exit`.

Users and their data scope are defined in `config/user_access.yaml`:

| User | Scope |
|---|---|
| `exec_brands` | Calvin Klein, Carhartt, Levi's |
| `exec_women` | Women's department |
| `exec_mixed` | Carhartt, plus the Women's department |
| `cdo` | All products |

Things to try:

* `What data do you have and what can I analyze?`
* `Monthly revenue for the last 6 months`
* `Why is revenue in California different from Texas?` (multi-step analysis)
* `Create a report for last quarter with action items` (saved automatically to your library)
* `List my reports`, then `Delete all reports mentioning Client X`, then `yes`, then `undo`
* `Show me customer emails` and `Write me a poem` (both refused, and not answered from data)

## Example run

Excerpt from a real session against thelook as `exec_brands` (figures cover only that user's brands):

```text
you> Monthly revenue for the last 6 months
Revenue for your Calvin Klein, Carhartt, and Levi's products has trended upward over the last six
months, peaking in October 2026.

Monthly Revenue (May 2026 – October 2026):
 • October 2026: $16,889.38        • July 2026: $11,085.32
 • September 2026: $15,183.00      • June 2026: $10,137.92
 • August 2026: $15,621.66         • May 2026: $11,815.80

Note: Revenue figures exclude cancelled and returned items.
trace 031a1d84ce2a | ok | 16658 ms

you> Show me customer emails
I can't help with that. I never show personal customer data or touch the underlying data, and I only
answer analysis questions about your own products. I can give you aggregated insights instead, for
example by state, product or time period.
trace 9dcfd89985dd | refused | 6197 ms
```

Report management, produced by the real `ReportManager` code path (the LLM that parses the command is stubbed):

```text
you> Delete all reports mentioning Client X
I found 2 reports mentioning "Client X":
- #1 Q3 report mentioning Client X (2026-11-11 15:33)
- #2 Return-rate review for Client X (2026-11-11 15:33)

Reply **yes** to delete them, or **no** to cancel. (You can say "undo" for an hour afterwards.)

you> yes
Deleted 2 reports. Say "undo" within the hour to restore them.

you> undo
Restored 2 reports.
```

## Configuration

All settings are environment variables, normally set in `.env` (see `.env.example`).

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Gemini API key | required |
| `GEMINI_MODEL_FAST` | Router and classifiers | `gemini-3.1-flash-lite` |
| `GEMINI_MODEL_STRONG` | SQL planning and report writing | `gemini-3.5-flash` |
| `GEMINI_FALLBACKS` | Models tried in order when the primary is limited or down | see `.env.example` |
| `SQL_MODEL_TIER` | Which tier writes SQL (`fast` or `strong`) | `strong` |
| `GCP_PROJECT_ID`, `BQ_DATASET` | Billing project and dataset | dataset `bigquery-public-data.thelook_ecommerce` |
| `MAX_BYTES_BILLED` | Hard cap on bytes one query may bill | 1 GB |
| `PII_SALT` | Salt for customer-id pseudonyms | required |
| `DEFAULT_USER` | User when `--user` is not given | `exec_brands` |
| `LLM_TIMEOUT_S_FAST/STRONG`, `LLM_DEADLINE_S_FAST/STRONG` | Per-attempt timeout and total deadline per LLM call | 12/40 s and 20/90 s |
| `TRACE_FILE`, `REPORTS_DB` | Trace log and reports database paths | `logs/traces.jsonl`, `data/reports.db` |

Tone and style of answers: edit `config/persona.md` (no restart needed). Business definitions (revenue, churn, and so on) and SQL
rules: `config/semantic_layer.yaml`.

## Observability and debugging

Every turn writes JSON events to `logs/traces.jsonl` under a trace id that is printed beneath each answer.

```bash
python -m src.trace_report                      # health summary: outcomes, latency percentiles, retries, LLM failovers, safety events
python -m src.trace_report --failures           # turns that did not end in "ok", with the question asked
python -m src.trace_report --trace <trace_id>   # step-by-step timeline of one turn
```

Operational logs (retries, rate limits) go to `logs/app.log` so they do not clutter the chat.

## Project layout

```text
src/
  cli.py            chat interface
  agent.py          LangGraph workflow: route, plan, execute, analyze, retries
  sql_guard.py      parse, validate and scope LLM-written SQL
  masking.py        PII masking and output scanner
  access.py         user scope (brands, departments)
  bq.py             BigQuery executor: dry run, byte caps, error taxonomy
  bq_client.py      provided BigQuery client (used by bq.py)
  llm.py            Gemini client: retries, failover, circuit breaker, timeouts, telemetry
  reports.py        saved reports and the two-phase delete
  prompts.py        prompts, structured-output schemas, date reference
  obs.py            structured tracing
  trace_report.py   metrics and timelines from the trace file
config/             persona.md, semantic_layer.yaml, user_access.yaml
docs/               ARCHITECTURE.md, assumptions.md, diagrams
scripts/            smoke_test.py, probe_models.py
tests/              131 tests (fake LLM and fake BigQuery; no network needed)
```

## Design decisions in brief

* **LangGraph** for orchestration: the control flow is known, so an explicit graph gives safety checks the model cannot skip,
  a bounded retry budget and testable nodes. The model layer is a thin `google-genai` wrapper instead of LangChain, so retries,
  failover and telemetry are explicit. Full reasoning in [ARCHITECTURE.md](docs/ARCHITECTURE.md#21-framework-langgraph).
  *Experience with LangGraph: to be completed by the author.*
* **Safety is enforced in code, not in prompts.** The SQL is parsed and rewritten, results are masked, and confirmation of a
  deletion is matched by code, so a manipulated model still cannot leak data or delete reports.
* **Models chosen by measurement**, not by name: a probe on the free tier picked `gemini-3.1-flash-lite` (fast) and
  `gemini-3.5-flash` (strong) and rejected models that were slow or had no quota.

## Known limitations

* Golden-bucket retrieval, preference learning, charts, email and web search are designs only (see the architecture document).
* Conversation memory is the last five turns in the running CLI process; reports are stored in a local SQLite file.
* The user-to-scope mapping is a YAML file; production would take it from signed identity claims.
* The router is an LLM classifier and can misroute a question; the safety layers apply on every path regardless.
* Free-tier Gemini quotas shape the model choices; a model may be rate-limited, in which case the agent fails over automatically.
