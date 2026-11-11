"""LangGraph agent for executive data analysis.

Why a graph and not a free-roaming ReAct loop: the control flow is known (route -> plan -> guard ->
execute -> analyze), every safety check is a deterministic node the model cannot skip, and the
retry budget is explicit, which bounds cost. The LLM only does what needs language: classify,
write SQL, explain results.

    route --analysis/report--> plan --> execute --ok--> analyze --> END
      |                          ^         |
      |                          +--retry--+ (guard / BigQuery / empty result, capped at max_attempts)
      +--schema / chitchat / refuse / manage_reports --> END
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph

from . import prompts
from .access import UnknownUserError, UserAccess
from .bq import BQUnavailableError, QueryError
from .llm import LLMPermanentError, LLMUnavailableError
from .masking import mask_frame, scan_text
from .obs import Tracer, new_trace_id
from .reports import ReportManager, ReportStore
from .sql_guard import SQLGuardError, guard_sql

log = logging.getLogger(__name__)

MSG_OFF_TOPIC = ("That is outside what I can help with. I analyse sales, product and customer performance, "
                 "and can explain what data is available. What would you like to know?")
MSG_REFUSED = ("I can't help with that. I never show personal customer data or touch the underlying data, "
               "and I only answer analysis questions about your own products. I can give you aggregated "
               "insights instead, for example by state, product or time period.")
MSG_NO_ACCESS = "No data access is configured for your account. Please contact your administrator."
MSG_UNAVAILABLE = ("A service this analysis depends on is temporarily unavailable. Nothing was lost; "
                   "please try again in a minute.")
MSG_FAILED = ("I couldn't build a reliable analysis for that after several attempts. Could you rephrase it or "
              "narrow it down (for example a specific product, state or time period)?")
MSG_NO_DATA = ("I ran the analysis but found no data matching that request within your product scope. "
               "Try a wider time period or different filters.")
MSG_MANAGE = "Saved-report management is not enabled in this build."
MSG_REPORT_SAVE_FAILED = "\n\n(I could not save this report to your library, but the analysis above is complete.)"
MSG_UNEXPECTED = "Something went wrong on my side. Your question was not lost; please try again."


class State(TypedDict, total=False):
    trace_id: str
    user_id: str
    conversation_id: str
    confirm: Optional[str]
    question: str
    history: list
    prefs: dict
    intent: str
    queries: list
    results: list
    feedback: Optional[str]
    last_failure: Optional[str]
    attempts: int
    answer: str
    status: str
    errors: list


@dataclass
class TurnResult:
    answer: str
    status: str
    intent: str = ""
    trace_id: str = ""
    attempts: int = 0
    sql: list = field(default_factory=list)
    latency_ms: int = 0
    errors: list = field(default_factory=list)  # feedback given to the model on each failed attempt


class DataAgent:
    def __init__(self, *, llm_fast, llm_sql, llm_strong, executor, access_loader, dataset: str,
                 config_dir: str = "config", salt: str = "dev-salt", tracer: Optional[Tracer] = None,
                 max_attempts: int = 3, max_rows_for_llm: int = 40,
                 report_store: Optional[ReportStore] = None):
        self.llm_fast, self.llm_sql, self.llm_strong = llm_fast, llm_sql, llm_strong
        self.executor = executor
        self.access_loader = access_loader
        self.dataset = dataset
        self.salt = salt
        self.tracer = tracer or Tracer(None)
        self.max_attempts = max_attempts
        self.max_rows_for_llm = max_rows_for_llm
        self.reports = report_store
        self._manager: Optional[ReportManager] = None
        cfg = Path(config_dir)
        self.schema_text = prompts.render_schema(yaml.safe_load((cfg / "semantic_layer.yaml").read_text()))
        self.persona_path = cfg / "persona.md"
        self._access_cache: dict[str, UserAccess] = {}
        if report_store is not None:
            self._manager = ReportManager(report_store, llm_fast, scrub=lambda t: scan_text(t)[0])
        from . import llm as _llm_mod, obs as _obs  # route LLM telemetry into the trace file
        _llm_mod.set_event_hook(
            lambda name, data: self.tracer.event(_obs.current_trace_id.get() or "-", name, **data))
        self.graph = self._build()

    # ---------------------------------------------------------------- helpers
    def _access(self, user_id: str) -> UserAccess:
        if user_id not in self._access_cache:
            self._access_cache[user_id] = self.access_loader(user_id)
        return self._access_cache[user_id]

    def _persona(self) -> str:
        try:  # re-read every turn: edits to persona.md apply immediately, no redeploy
            return self.persona_path.read_text()
        except OSError:
            return "Professional, concise, plain business English."

    # ---------------------------------------------------------------- graph
    def _build(self):
        g = StateGraph(State)
        g.add_node("route", self.n_route)
        g.add_node("plan", self.n_plan)
        g.add_node("execute", self.n_execute)
        g.add_node("analyze", self.n_analyze)
        g.add_node("give_up", self.n_give_up)
        g.add_node("schema_answer", self.n_schema_answer)
        g.add_node("chitchat", self.n_chitchat)
        g.add_node("refuse", self.n_refuse)
        g.add_node("manage_reports", self.n_manage_reports)

        g.add_edge(START, "route")
        g.add_conditional_edges("route", self.next_after_route, {
            "plan": "plan", "schema_answer": "schema_answer", "manage_reports": "manage_reports",
            "chitchat": "chitchat", "refuse": "refuse"})
        g.add_edge("plan", "execute")
        g.add_conditional_edges("execute", self.next_after_execute, {
            "plan": "plan", "analyze": "analyze", "give_up": "give_up", "end": END})
        for node in ("analyze", "give_up", "schema_answer", "chitchat", "refuse", "manage_reports"):
            g.add_edge(node, END)
        return g.compile()

    # ---------------------------------------------------------------- nodes
    def n_route(self, s: State) -> dict:
        tid = s["trace_id"]
        if self._manager is not None:
            # events from the manager (including discard_pending) belong to THIS turn
            self._manager.event = lambda name, **kw: self.tracer.event(tid, name, **kw)
            # A plain yes/no while a deletion awaits confirmation is handled without the LLM:
            # the model can neither confirm a deletion nor be talked into one, and it works during LLM outages.
            reply = self._manager.pending_reply(s["user_id"], s["conversation_id"], s["question"])
            if reply:
                self.tracer.event(tid, "intent", intent="manage_reports", model="deterministic")
                return {"intent": "manage_reports", "confirm": reply, "attempts": 0}
        with self.tracer.span(tid, "route"):
            res = self.llm_fast.generate(prompts.router_prompt(s.get("history", []), s["question"]),
                                         system=prompts.ROUTER_SYSTEM, schema=prompts.RouteDecision,
                                         max_output_tokens=200)
        intent = res.parsed.intent
        self.tracer.event(tid, "intent", intent=intent, model=res.model)
        if self._manager is not None and intent != "manage_reports":
            self._manager.discard_pending(s["user_id"], s["conversation_id"])  # anything else cancels it
        return {"intent": intent, "confirm": None, "attempts": 0}

    def next_after_route(self, s: State) -> str:
        return {"analysis": "plan", "report": "plan", "schema_question": "schema_answer",
                "manage_reports": "manage_reports", "chitchat": "chitchat",
                "off_topic": "refuse", "malicious": "refuse"}[s["intent"]]

    def n_refuse(self, s: State) -> dict:
        if s["intent"] == "malicious":
            self.tracer.event(s["trace_id"], "blocked_request", reason="malicious_or_pii")
            return {"answer": MSG_REFUSED, "status": "refused"}
        return {"answer": MSG_OFF_TOPIC, "status": "refused"}

    def n_manage_reports(self, s: State) -> dict:
        if self._manager is None:
            return {"answer": MSG_MANAGE, "status": "ok"}
        tid = s["trace_id"]
        self._manager.event = lambda name, **kw: self.tracer.event(tid, name, **kw)
        with self.tracer.span(tid, "manage_reports"):
            answer = self._manager.handle(s["user_id"], s["conversation_id"], s["question"],
                                          s.get("history", []), confirm=s.get("confirm"))
        return {"answer": answer, "status": "ok"}

    def n_chitchat(self, s: State) -> dict:
        res = self.llm_fast.generate(s["question"], system=prompts.CHITCHAT_SYSTEM, max_output_tokens=120)
        return {"answer": scan_text(res.text)[0], "status": "ok"}

    def n_schema_answer(self, s: State) -> dict:
        res = self.llm_fast.generate(prompts.schema_answer_prompt(self.schema_text, s["question"]),
                                     max_output_tokens=700)
        return {"answer": scan_text(res.text)[0], "status": "ok"}

    def n_plan(self, s: State) -> dict:
        tid = s["trace_id"]
        attempt = s.get("attempts", 0) + 1
        queries: list = []
        try:
            with self.tracer.span(tid, "plan", attempt=attempt):
                res = self.llm_sql.generate(
                    prompts.plan_user(s["question"], s.get("history", []), s.get("feedback")),
                    system=prompts.plan_system(self.schema_text), schema=prompts.QueryPlan,
                    max_output_tokens=1500)
            queries = [q.model_dump() for q in res.parsed.queries][:3] if res.parsed else []
        except ValueError:  # model returned JSON that does not match the schema: treat as a fixable error
            self.tracer.event(tid, "plan_invalid_output", attempt=attempt)
        return {"queries": queries, "attempts": attempt}

    def n_execute(self, s: State) -> dict:
        tid = s["trace_id"]
        access = self._access(s["user_id"])
        results: list = []
        feedback: Optional[str] = None
        failure: Optional[str] = None

        if not s.get("queries"):
            fb = "Your previous reply contained no valid queries. Return 1-3 queries in the requested structure."
            return {"results": [], "feedback": fb, "last_failure": "sql_error",
                    "errors": list(s.get("errors", [])) + [fb]}

        for i, q in enumerate(s["queries"], 1):
            try:
                guarded = guard_sql(q["sql"], access, dataset=self.dataset)
            except SQLGuardError as e:
                self.tracer.event(tid, "guard_reject", query=i, code=e.code)
                if e.code == "no_scope":
                    return {"results": [], "status": "no_access", "answer": MSG_NO_ACCESS}
                feedback, failure = f"Query {i} was rejected by the SQL validator: {e}", "sql_error"
                break
            try:
                with self.tracer.span(tid, "bq_query", query=i):
                    r = self.executor.run(guarded.sql)
            except QueryError as e:
                feedback, failure = f"Query {i} failed in BigQuery: {e}", "sql_error"
                break
            except BQUnavailableError:
                return {"results": [], "status": "unavailable", "answer": MSG_UNAVAILABLE}

            masked, notes = mask_frame(r.df, self.salt)
            if notes:
                self.tracer.event(tid, "masked", query=i, notes=notes)
            results.append({
                "purpose": q["purpose"], "sql": q["sql"], "rows": len(masked),
                "truncated": len(masked) > self.max_rows_for_llm,
                "csv": masked.head(self.max_rows_for_llm).to_csv(index=False, float_format="%.2f"),
            })

        if feedback is None and all(r["rows"] == 0 for r in results):
            feedback, failure = ("All queries returned zero rows. Re-check filters (status values, date range "
                                 "anchoring, spelling) and try a broader query."), "empty"
        if feedback:
            self.tracer.event(tid, "retry_needed", failure=failure, attempt=s.get("attempts", 0))
        errors = list(s.get("errors", [])) + ([feedback] if feedback else [])
        return {"results": results, "feedback": feedback, "last_failure": failure, "errors": errors}

    def next_after_execute(self, s: State) -> str:
        if s.get("status") in ("unavailable", "no_access"):
            return "end"
        if s.get("feedback"):
            return "plan" if s.get("attempts", 0) < self.max_attempts else "give_up"
        return "analyze"

    def n_give_up(self, s: State) -> dict:
        if s.get("last_failure") == "empty":
            return {"answer": MSG_NO_DATA, "status": "no_data"}
        return {"answer": MSG_FAILED, "status": "failed"}

    def n_analyze(self, s: State) -> dict:
        tid = s["trace_id"]
        blocks = "\n\n".join(
            f"### Query {i}: {r['purpose']}\nSQL used: {r['sql'][:1500]}\nRows: {r['rows']}"
            f"{' (truncated to the first %d)' % self.max_rows_for_llm if r['truncated'] else ''}\n{r['csv']}"
            for i, r in enumerate(s["results"], 1))
        note = ""
        if any(r["rows"] == 0 for r in s["results"]):
            note = "\nNote: some queries returned no rows; say so rather than guessing."
        system = prompts.analyst_system(self._persona(), s.get("intent") == "report", s.get("prefs") or {},
                                        self._access(s["user_id"]).describe())
        with self.tracer.span(tid, "analyze"):
            res = self.llm_strong.generate(
                prompts.analyst_prompt(s["question"], s.get("history", []), blocks, note),
                system=system, temperature=0.3, max_output_tokens=2500)
        clean, findings = scan_text(res.text)
        if findings:
            self.tracer.event(tid, "pii_output_redacted", findings=findings)
        if s.get("intent") == "report" and self.reports is not None:
            try:  # a storage failure must never lose the analysis the user just waited for
                saved = self.reports.save(s["user_id"], s["conversation_id"], s["question"], clean)
                clean += f"\n\n_Saved as report #{saved.id}._"
                self.tracer.event(tid, "report_saved", report_id=saved.id)
            except Exception as e:  # noqa: BLE001
                log.exception("could not save report")
                self.tracer.event(tid, "report_save_failed", error=type(e).__name__)
                clean += MSG_REPORT_SAVE_FAILED
        return {"answer": clean, "status": "ok"}

    # ---------------------------------------------------------------- entry point
    def handle_turn(self, user_id: str, question: str, history: Optional[list] = None,
                    prefs: Optional[dict] = None, conversation_id: str = "default") -> TurnResult:
        tid, t0 = new_trace_id(), time.monotonic()
        self.tracer.event(tid, "turn_start", user_id=user_id, question=question[:500])
        from . import obs as _obs_ctx
        _obs_ctx.current_trace_id.set(tid)
        final: dict[str, Any]
        try:
            self._access(user_id)  # unknown user -> UnknownUserError (fail closed)
            final = self.graph.invoke(
                {"trace_id": tid, "user_id": user_id, "conversation_id": conversation_id, "question": question,
                 "history": history or [], "prefs": prefs or {}},
                config={"recursion_limit": 25})
        except UnknownUserError:
            final = {"status": "no_access", "answer": MSG_NO_ACCESS}
        except LLMUnavailableError as e:
            self.tracer.event(tid, "llm_unavailable", attempts=e.attempts)
            final = {"status": "unavailable", "answer": MSG_UNAVAILABLE}
        except LLMPermanentError as e:
            self.tracer.event(tid, "llm_permanent_error", error=str(e)[:200])
            final = {"status": "failed", "answer": MSG_UNEXPECTED}
        except Exception as e:  # noqa: BLE001 - the UI must never crash
            log.exception("unexpected error in turn %s", tid)
            self.tracer.event(tid, "unexpected_error", error=type(e).__name__)
            final = {"status": "failed", "answer": MSG_UNEXPECTED}

        ms = int((time.monotonic() - t0) * 1000)
        result = TurnResult(
            answer=final.get("answer", MSG_UNEXPECTED), status=final.get("status", "failed"),
            intent=final.get("intent", ""), trace_id=tid, attempts=final.get("attempts", 0),
            sql=[q["sql"] for q in final.get("queries", [])], latency_ms=ms, errors=final.get("errors", []))
        self.tracer.event(tid, "turn_end", status=result.status, intent=result.intent,
                          attempts=result.attempts, ms=ms)
        return result
