import pandas as pd
import pytest
from google.api_core import exceptions as gexc

from src.bq import BQExecutor, BQUnavailableError, QueryError, QueryTooExpensive


class Job:
    def __init__(self, bytes_=0, df=None):
        self.total_bytes_processed, self._df = bytes_, df

    def result(self, timeout=None):
        return self

    def to_dataframe(self, **kw):
        return self._df


class FakeClient:
    def __init__(self, dry=(), real=()):
        self.dry, self.real, self.configs = list(dry), list(real), []

    def query(self, sql, job_config=None):
        self.configs.append(job_config)
        o = (self.dry if job_config.dry_run else self.real).pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def make(client, **kw):
    runner = type("R", (), {"client": client})()
    return BQExecutor(runner, sleep=lambda s: None, **kw)


DF = pd.DataFrame({"a": [1]})


def test_success_sets_server_side_byte_cap():
    c = FakeClient(dry=[Job(500)], real=[Job(df=DF)])
    r = make(c, max_bytes=1000).run("SELECT 1")
    assert r.df.equals(DF) and r.bytes_processed == 500
    assert c.configs[1].maximum_bytes_billed == 1000


def test_syntax_error_caught_at_dry_run_costs_nothing():
    c = FakeClient(dry=[gexc.BadRequest("Syntax error at [1:8]")])
    with pytest.raises(QueryError):
        make(c).run("SELEC")
    assert c.real == []  # never executed


def test_too_expensive_query_is_rejected_as_fixable():
    c = FakeClient(dry=[Job(5_000_000_000)])
    with pytest.raises(QueryTooExpensive):
        make(c, max_bytes=1_000_000_000).run("SELECT *")


def test_transient_error_is_retried():
    c = FakeClient(dry=[gexc.ServiceUnavailable("503"), Job(10)], real=[Job(df=DF)])
    assert make(c).run("SELECT 1").df.equals(DF)


def test_persistent_outage_raises_unavailable():
    c = FakeClient(dry=[gexc.ServiceUnavailable("503")] * 3)
    with pytest.raises(BQUnavailableError):
        make(c, retries=2).run("SELECT 1")


def test_permission_error_is_not_retried():
    c = FakeClient(dry=[gexc.Forbidden("denied"), Job(1)])
    with pytest.raises(BQUnavailableError):
        make(c).run("SELECT 1")
    assert len(c.dry) == 1  # second outcome untouched
