"""BigQuery execution with cost and failure controls, built on the provided BigQueryRunner client.

* dry-run first: catches syntax/semantic errors for free and enforces a bytes-scanned cap
* maximum_bytes_billed on the real job: server-side hard stop even if the estimate was off
* error taxonomy:
    QueryError           -> the SQL is wrong/too expensive; message goes back to the LLM to self-correct
    BQUnavailableError   -> infrastructure problem (outage, quota, permissions); never retried by the LLM loop
* transient errors (503/500/429/timeouts) are retried here with exponential backoff
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from google.api_core import exceptions as gexc
from google.cloud import bigquery

log = logging.getLogger(__name__)

TRANSIENT = (
    gexc.ServiceUnavailable, gexc.InternalServerError, gexc.TooManyRequests,
    gexc.DeadlineExceeded, gexc.GatewayTimeout, ConnectionError, TimeoutError,
)


class QueryError(Exception):
    """Fixable by rewriting the SQL."""


class QueryTooExpensive(QueryError):
    pass


class BQUnavailableError(Exception):
    """BigQuery cannot serve us right now."""


@dataclass
class QueryResult:
    df: pd.DataFrame
    bytes_processed: int


class BQExecutor:
    def __init__(self, runner: Any, *, max_bytes: int = 1_000_000_000, timeout_s: float = 60,
                 retries: int = 2, base_delay_s: float = 1.0, sleep: Callable[[float], None] = time.sleep):
        self.client = runner.client
        self.max_bytes = max_bytes
        self.timeout_s = timeout_s
        self.retries = retries
        self.base_delay_s = base_delay_s
        self._sleep = sleep

    @staticmethod
    def _clean(e: Exception) -> str:
        msg = getattr(e, "message", None) or str(e)
        return " ".join(msg.split())[:400]

    def _run_once(self, sql: str) -> QueryResult:
        dry = self.client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
        est = dry.total_bytes_processed or 0
        if est > self.max_bytes:
            raise QueryTooExpensive(
                f"Query would scan {est / 1e9:.2f} GB (limit {self.max_bytes / 1e9:.2f} GB). "
                "Aggregate more, filter by date, or select fewer columns."
            )
        job = self.client.query(sql, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=self.max_bytes))
        df = job.result(timeout=self.timeout_s).to_dataframe(create_bqstorage_client=False)
        return QueryResult(df=df, bytes_processed=est)

    def run(self, sql: str) -> QueryResult:
        for attempt in range(self.retries + 1):
            try:
                return self._run_once(sql)
            except QueryError:
                raise
            except (gexc.BadRequest, gexc.NotFound) as e:
                raise QueryError(self._clean(e)) from e
            except gexc.Forbidden as e:
                raise BQUnavailableError(f"BigQuery denied the request: {self._clean(e)}") from e
            except TRANSIENT as e:
                log.warning("bq.transient_error attempt=%d err=%s", attempt + 1, type(e).__name__)
                if attempt == self.retries:
                    raise BQUnavailableError(f"BigQuery unavailable after retries: {type(e).__name__}") from e
                self._sleep(self.base_delay_s * (2 ** attempt))
        raise BQUnavailableError("unreachable")  # pragma: no cover
