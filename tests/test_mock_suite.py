"""
tests/test_mock_suite.py — Mocked unit tests for all 6 agent modules + api.py.

No real OpenAI API calls. All LLM interactions are mocked.
Safe to run in CI or during development with no credentials.

Run with: pytest tests/test_mock_suite.py -v
"""

# Import under private aliases so pytest does not collect them as tests
from agents.planner        import test_planner        as _planner
from agents.librarian      import test_librarian      as _librarian
from agents.data_scientist import test_data_scientist as _data_scientist
from agents.router         import test_router         as _router
from agents.synthesizer    import test_synthesizer    as _synthesizer
from agents.auditor        import test_auditor        as _auditor
from api                   import test_api            as _api


def test_planner_mock():
    _planner()


def test_librarian_mock():
    _librarian()


def test_data_scientist_mock():
    _data_scientist()


def test_router_mock():
    _router()


def test_synthesizer_mock():
    _synthesizer()


def test_auditor_mock():
    _auditor()


def test_api_mock():
    _api()
