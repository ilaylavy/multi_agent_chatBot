def test_core_imports():
    from core.state import AgentState
    from core.llm_config import get_llm
    from core.manifest import get_manifest_index, get_manifest_detail
    from core.registry import WORKER_REGISTRY
