def test_agents_imports():
    from agents.planner import planner_node
    from agents.librarian import librarian_worker
    from agents.data_scientist import data_scientist_worker
    from agents.router import router_node
    from agents.synthesizer import synthesizer_node
    from agents.auditor import auditor_node
    from agents.chat import chat_node

    from core.registry import WORKER_REGISTRY
    from agents.librarian import librarian_worker as _lib
    from agents.data_scientist import data_scientist_worker as _ds

    assert WORKER_REGISTRY["librarian"]      is _lib, "registry still points to placeholder"
    assert WORKER_REGISTRY["data_scientist"] is _ds,  "registry still points to placeholder"
