"""
core/llm_config.py — Per-agent LLM loader.

Agents never instantiate their own LLM client.
They call get_llm(agent_name) and receive a fully configured ChatOpenAI instance.
Swapping a model = one line in config.yaml. Agent code never changes.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Bootstrap — load .env once at import time
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_config() -> dict:
    """Load and cache config.yaml from the project root."""
    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _agent_config(agent_name: str) -> dict:
    """Return the llm config block for a single agent."""
    config = _load_config()
    agents = config["llm"]["agents"]
    if agent_name not in agents:
        valid = list(agents.keys())
        raise ValueError(
            f"Unknown agent '{agent_name}'. Valid names: {valid}"
        )
    return agents[agent_name]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_llm(agent_name: str) -> ChatOpenAI:
    """
    Return a configured ChatOpenAI client for the named agent.

    Parameters
    ----------
    agent_name : str
        One of: chat, planner, router, librarian, data_scientist,
        synthesizer, auditor.

    Returns
    -------
    ChatOpenAI
        A LangChain ChatOpenAI instance with the model and temperature
        specified in config.yaml for that agent.

    Raises
    ------
    ValueError
        If agent_name is not present in config.yaml.
    EnvironmentError
        If OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    cfg = _agent_config(agent_name)
    return ChatOpenAI(
        model=cfg["model"],
        temperature=cfg["temperature"],
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Isolated test — run with: python core/llm_config.py
# ---------------------------------------------------------------------------

def test_llm_config():
    agents = [
        "chat",
        "planner",
        "router",
        "librarian",
        "data_scientist",
        "synthesizer",
        "auditor",
    ]

    config = _load_config()
    agent_cfg = config["llm"]["agents"]

    print(f"{'Agent':<16} {'Model':<12} {'Temperature'}")
    print("-" * 38)

    for name in agents:
        llm = get_llm(name)
        expected_model = agent_cfg[name]["model"]
        expected_temp  = agent_cfg[name]["temperature"]

        assert llm.model_name == expected_model, (
            f"{name}: expected model '{expected_model}', got '{llm.model_name}'"
        )
        assert llm.temperature == expected_temp, (
            f"{name}: expected temperature {expected_temp}, got {llm.temperature}"
        )

        print(f"{name:<16} {llm.model_name:<12} {llm.temperature}")

    # Unknown agent must raise ValueError
    try:
        get_llm("nonexistent_agent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Missing key must raise EnvironmentError
    saved_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        get_llm("chat")
        assert False, "Should have raised EnvironmentError"
    except EnvironmentError:
        pass
    finally:
        if saved_key:
            os.environ["OPENAI_API_KEY"] = saved_key

    print("\nPASS: all 7 agents return correctly configured ChatOpenAI clients")


if __name__ == "__main__":
    test_llm_config()
