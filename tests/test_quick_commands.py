"""Bare 'undo' and stale yes/no are answered in code: no LLM call, no confusion with new requests."""
from types import SimpleNamespace

from src.reports import MSG_NOTHING_PENDING, ReportManager, is_undo


class FakeStore:
    def __init__(self, pending=None, undo_result=("restored", 1)):
        self.pending, self.undo_result, self.undo_calls = pending, undo_result, 0

    def get_pending(self, user_id, conversation_id):
        return self.pending

    def undo_last_delete(self, user_id):
        self.undo_calls += 1
        return self.undo_result


def _manager(store=None, events=None):
    # llm=object(): any attempt to call the LLM raises AttributeError, so these tests prove none happens
    m = ReportManager(store or FakeStore(), object())
    if events is not None:
        m.event = lambda name, **kw: events.append((name, kw))
    return m


def test_is_undo_matches_plain_requests_only():
    for text in ["undo", "Undo!", "  undo that. ", "UNDO IT", "bring them back", "Restore them"]:
        assert is_undo(text), text
    for text in ["undo my last revenue query", "what does undo do?", "delete all reports", "yes", ""]:
        assert not is_undo(text), text


def test_bare_undo_is_deterministic_and_restores():
    events = []
    store = FakeStore()
    m = _manager(store, events)
    assert m.quick_command("u", "c", "undo", []) == "undo"
    answer = m.handle("u", "c", "undo", [], confirm="undo")
    assert answer.startswith("Restored") and store.undo_calls == 1
    assert events == [("delete_undone", {"status": "restored", "count": 1})]


def test_undo_with_nothing_to_undo_says_so():
    m = _manager(FakeStore(undo_result=("none", 0)))
    assert m.handle("u", "c", "undo", [], confirm="undo") == "There is no recent deletion to undo."


def test_stale_yes_or_no_with_nothing_pending():
    m = _manager()
    history = [{"q": "what can you do?", "a": "I provide analytical insights. Ask me about sales."}]
    for text in ["yes", "No", "go ahead", "cancel"]:
        assert m.quick_command("u", "c", text, history) == "nothing_pending", text
    assert m.handle("u", "c", "yes", history, confirm="nothing_pending") == MSG_NOTHING_PENDING


def test_yes_answers_the_assistants_follow_up_question():
    m = _manager()
    history = [{"q": "revenue by state", "a": "California leads. Would you like me to break this down by month?"}]
    assert m.quick_command("u", "c", "yes", history) is None  # goes to the normal router


def test_yes_while_a_deletion_is_pending_is_left_to_pending_reply():
    m = _manager(FakeStore(pending=SimpleNamespace(report_ids=[1])))
    assert m.quick_command("u", "c", "yes", []) is None


def test_normal_requests_are_not_intercepted():
    m = _manager()
    for text in ["list my reports", "show me revenue for California", "undo my last query please"]:
        assert m.quick_command("u", "c", text, []) is None, text
