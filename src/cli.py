"""CLI chat interface.   python -m src.cli --user exec_brands [--show-sql]"""
import argparse
import logging
import os
import sys
import uuid
import warnings
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

HELP = ("Commands: /help  /sql (toggle SQL display)  /exit\n"
        "Reports: ask for a report to save it; try \"list my reports\" or \"delete all reports mentioning X\".")


def build_agent():
    from .access import load_access
    from .agent import DataAgent
    from .bq import BQExecutor
    from .bq_client import BigQueryRunner
    from .llm import LLMClient
    from .obs import Tracer
    from .reports import ReportStore

    dataset = os.getenv("BQ_DATASET", "bigquery-public-data.thelook_ecommerce")
    runner = BigQueryRunner(project_id=os.getenv("GCP_PROJECT_ID") or None, dataset_id=dataset)
    return DataAgent(
        llm_fast=LLMClient.from_env("fast"),
        llm_sql=LLMClient.from_env(os.getenv("SQL_MODEL_TIER", "fast")),
        llm_strong=LLMClient.from_env("strong"),
        executor=BQExecutor(runner, max_bytes=int(os.getenv("MAX_BYTES_BILLED", 1_000_000_000))),
        access_loader=load_access,
        dataset=dataset,
        salt=os.getenv("PII_SALT", "dev-salt"),
        tracer=Tracer(os.getenv("TRACE_FILE", "logs/traces.jsonl")),
        report_store=ReportStore(os.getenv("REPORTS_DB", "data/reports.db")),
    )


def main() -> int:
    load_dotenv()
    warnings.filterwarnings("ignore", message=".*BigQuery Storage module not found.*")
    Path("logs").mkdir(exist_ok=True)  # keep operational logs out of the chat UI
    logging.basicConfig(filename="logs/app.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Retail data analysis assistant")
    ap.add_argument("--user", default=os.getenv("DEFAULT_USER", "exec_brands"), help="user id from config/user_access.yaml")
    ap.add_argument("--show-sql", action="store_true", help="print the SQL the agent wrote")
    args = ap.parse_args()
    console = Console()

    missing = [k for k in ("GEMINI_API_KEY", "GEMINI_MODEL_FAST", "GEMINI_MODEL_STRONG") if not os.getenv(k)]
    if missing:
        console.print(f"[red]Missing in .env: {', '.join(missing)}[/red]")
        return 1
    try:
        agent = build_agent()
        agent._access(args.user)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Startup failed: {type(e).__name__}: {e}[/red]")
        return 1

    console.print(Panel(f"Signed in as [bold]{args.user}[/bold]. Ask about sales, products, customers or trends.\n{HELP}",
                        title="Retail Data Assistant"))
    history: list[dict] = []
    conversation_id = uuid.uuid4().hex  # one per CLI session
    show_sql = args.show_sql
    while True:
        try:
            q = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/exit", "/quit"):
            break
        if q == "/help":
            console.print(HELP)
            continue
        if q == "/sql":
            show_sql = not show_sql
            console.print(f"SQL display {'on' if show_sql else 'off'}")
            continue
        with console.status("Analysing..."):
            res = agent.handle_turn(args.user, q, history, conversation_id=conversation_id)
        console.print(Markdown(res.answer))
        if show_sql and res.errors:
            console.print(Panel("\n".join(f"attempt {i}: {e}" for i, e in enumerate(res.errors, 1)),
                                title="Self-correction feedback", style="dim yellow"))
        if show_sql and res.sql:
            console.print(Panel("\n\n".join(res.sql), title="SQL (before scoping)", style="dim"))
        console.print(f"[dim]trace {res.trace_id} | {res.status} | {res.latency_ms} ms[/dim]\n")
        history.append({"q": q, "a": res.answer})
        history = history[-5:]
    console.print("Goodbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
